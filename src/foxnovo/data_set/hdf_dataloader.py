import gc
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from alphabase.io.hdf import HDF_File
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from foxnovo.constants import HYDROGEN_MASS, ISOTOPE_MASS_DIFF, PROTON_MASS
from foxnovo.model.config import MOD_TO_AA_TOKEN

logger = logging.getLogger(__name__)


def _remove_precursor_numpy(mz: np.ndarray, remove_mz: np.ndarray, tol: float):
    """
    numpy remove precursor peaks
    """
    mask = np.ones(len(mz), dtype=bool)
    for rm in remove_mz:
        mask &= np.abs(mz - rm) >= tol
    return mask


mod_to_aa_token = MOD_TO_AA_TOKEN
ignore_mod = []

allow_ignore_mod_num = 1


def filter_ignore_mods(psm_df, mods_column, ignore_mod):
    initial_count = psm_df.shape[0]

    def has_more_than_one_ignore(mods):
        if pd.isna(mods):
            return False
        count = 0
        for mod in ignore_mod:
            if mod in mods:
                count += 1
                if count > 1:
                    return True
        return False

    filtered_df = psm_df[~psm_df[mods_column].apply(has_more_than_one_ignore)]
    deleted_count = initial_count - filtered_df.shape[0]

    return filtered_df, deleted_count


# def alpha_raw_reader(file_path, file_type):
#     "简单的多类型原始数据MS2加载器"
#     if file_type == 'hdf5':
#         f = HDF_File(file_name=file_path, read_only=True)
#         spectrum_df = f.psm.psm_df.values
#         peak_df = f.ms_data.peak_df.values
#     elif file_type == 'raw':
#         raw_data = ThermoRawData()
#         raw_data.import_raw(file_path)
#         spectrum_df = raw_data.spectrum_df
#         spectrum_df.rename(columns={'precursor_charge': 'charge'}, inplace=True)
#         peak_df = raw_data.peak_df
#     elif file_type == 'mzml':
#         mzml_reader = ms_reader_provider.get_reader("mzml")
#         mzml_reader.import_raw(file_path)
#         spectrum_df = mzml_reader.spectrum_df
#         spectrum_df.rename(columns={'precursor_charge': 'charge'}, inplace=True)
#         peak_df = mzml_reader.peak_df
#     else:
#         print('Unknow raw data type, please make sure your raw data is Thermo raw/hdf5/mzml!')
#     return spectrum_df, peak_df


def build_mod_seq(sequences, mods_list, sites_list, mod_dict):
    result = []
    mass_dict = {k: "+" + v.split("+")[-1] for k, v in mod_dict.items()}
    for seq, mods, sites in zip(sequences, mods_list, sites_list, strict=False):
        if not mods or pd.isna(mods) or mods.strip() == "":
            result.append(seq)
            continue

        mods_split = mods.split(";")
        sites_split = list(map(int, sites.split(";")))

        mod_map = {}
        for mod, site in zip(mods_split, sites_split, strict=False):
            if mod in ignore_mod:
                continue
            tag = mass_dict.get(mod, "")
            if tag:
                mod_map[site - 1] = tag

        parts = []
        for i, aa in enumerate(seq):
            parts.append(aa)
            if i in mod_map:
                parts.append(mod_map[i])
        result.append("".join(parts))
    return result


class HDFParser:
    def __init__(self, hdf5_path: str):
        self.data = {}
        self.hdf5_path = hdf5_path

        self.charge_arr = None
        self.start_arr = None
        self.end_arr = None
        self.mz_arr = None
        self.int_arr = None
        self.prec_mz_arr = None
        self.spec_idx_arr = None
        self.n = 0

    def load_data(self):
        f = HDF_File(file_name=self.hdf5_path, read_only=True)
        try:
            psm_df = f.ms_data.spectrum_df.values[
                [
                    "charge",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "spec_idx",
                    "ms_level",
                ]
            ]
            psm_df = psm_df[psm_df["ms_level"] == 2]

        except Exception:
            logger.exception("Failed to extract MS2 spectrum dataframe from f.ms_data.spectrum_df")
            raise

        psm_df = psm_df[(psm_df["charge"] > 0) & (psm_df["charge"] <= 5)]
        peak_df = f.ms_data.peak_df.values

        self.charge_arr = psm_df["charge"].values.astype(int)
        self.start_arr = psm_df["peak_start_idx"].values.astype(int)
        self.end_arr = psm_df["peak_stop_idx"].values.astype(int)
        self.prec_mz_arr = psm_df["precursor_mz"].values.astype(float)
        self.spec_idx_arr = psm_df["spec_idx"].values.astype(int)

        self.mz_arr = peak_df["mz"].values
        self.int_arr = peak_df["intensity"].values
        self.n = len(psm_df)

    def get_spectrum(self, idx: int):
        s = self.start_arr[idx]
        e = self.end_arr[idx]
        mz = self.mz_arr[s:e]
        intensity = self.int_arr[s:e]
        prec_mz = self.prec_mz_arr[idx]
        charge = self.charge_arr[idx]
        spec_idx = self.spec_idx_arr[idx]
        return mz, intensity, prec_mz, charge, spec_idx


class AnnotatedHDFParser:
    def __init__(self, hdf5_folder: str):
        self.data = {}
        self.hdf5_folder = hdf5_folder

        self.global_index_arr = None
        self.file_map = {}

        self.tensors = {}
        self.all_peptides = []
        self.all_raw_names = []
        self.all_charges = []

    def load_data(self):
        folder_path = Path(self.hdf5_folder)
        self.files = sorted(folder_path.glob("*.hdf5"))
        temp_global_index = []

        charges, p_mzs, s_indices, e_indices, scores, spec_ids = [], [], [], [], [], []

        for file_idx, file_path in enumerate(self.files):
            self.file_map[file_idx] = file_path
            f = HDF_File(file_name=file_path, read_only=True)
            df = f.psm.psm_df.values[
                [
                    "charge",
                    "mods",
                    "mod_sites",
                    "nAA",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "sequence",
                    "raw_name",
                    "spec_idx",
                    "score",
                ]
            ]
            peak_df = f.ms_data.peak_df.values
            mod_seqs = build_mod_seq(
                df["sequence"], df["mods"], df["mod_sites"], mod_dict=mod_to_aa_token
            )

            charges.extend(df["charge"].values.astype(np.int8))
            p_mzs.extend(df["precursor_mz"].values.astype(np.float32))
            s_indices.extend(df["peak_start_idx"].values.astype(np.int64))
            e_indices.extend(df["peak_stop_idx"].values.astype(np.int64))
            scores.extend(df["score"].values.astype(np.float32))
            spec_ids.extend(df["spec_idx"].values.astype(np.int32))

            mz_t = torch.from_numpy(peak_df["mz"].to_numpy().astype(np.float32))
            int_t = torch.from_numpy(peak_df["intensity"].to_numpy().astype(np.float32))

            self.all_peptides.extend(mod_seqs)
            self.all_raw_names.extend(df["raw_name"].values)
            self.all_charges.extend(df["charge"].values.astype(np.int8))
            self.data[file_path] = [mz_t, int_t]

            current_len = len(df)
            file_indices = np.empty((current_len, 2), dtype=np.int32)
            file_indices[:, 0] = file_idx
            file_indices[:, 1] = np.arange(current_len)
            temp_global_index.append(file_indices)

        self.tensors["charge"] = torch.tensor(charges)
        self.tensors["precursor_mz"] = torch.tensor(p_mzs)
        self.tensors["peak_start_idx"] = torch.tensor(s_indices)
        self.tensors["peak_stop_idx"] = torch.tensor(e_indices)
        self.tensors["score"] = torch.tensor(scores)
        self.tensors["spec_idx"] = torch.tensor(spec_ids)

        self.global_index_arr = np.vstack(temp_global_index)
        self.all_peptides = np.array(self.all_peptides, dtype="S50")
        self.all_raw_names = np.array(self.all_raw_names, dtype="S100")
        self.all_charges = np.array(self.all_charges, dtype=np.int8)

        gc.collect()
        gc.freeze()

        composite_keys = [
            f"{seq.decode('ascii')}_{charge}"
            for seq, charge in zip(self.all_peptides, self.all_charges, strict=False)
        ]
        freq_counter = Counter(composite_keys)
        weights = [1.0 / np.sqrt(freq_counter[key]) for key in composite_keys]
        self.peptide_weights = torch.tensor(weights, dtype=torch.float32)

    def get_spectrum(self, idx: int):
        file_idx = self.global_index_arr[idx, 0]
        file_path = self.file_map[file_idx]

        start_idx = self.tensors["peak_start_idx"][idx].item()
        stop_idx = self.tensors["peak_stop_idx"][idx].item()

        mz_t, int_t = self.data[file_path]
        mz = mz_t[start_idx:stop_idx].numpy()
        intensity = int_t[start_idx:stop_idx].numpy()

        precursor_mz = self.tensors["precursor_mz"][idx].item()
        precursor_charge = self.tensors["charge"][idx].item()
        sequence = self.all_peptides[idx].decode("ascii")
        raw_name = self.all_raw_names[idx].decode("ascii")
        spec_idx = self.tensors["spec_idx"][idx].item()
        score = self.tensors["score"][idx].item()

        return mz, intensity, precursor_mz, precursor_charge, sequence, raw_name, spec_idx, score


class HDFSpectrumDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        n_peaks: int = 150,
        min_intensity: float = 0.01,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        remove_precursor_tol: float = 2.0,
        random_state: int | None = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity
        self.dataparser = HDFParser(hdf5_path)
        self.dataparser.load_data()
        self.n_spectra = self.dataparser.n
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol

    def __len__(self):
        return self.n_spectra

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> np.ndarray:
        min_mz = self.min_mz
        max_mz = self.max_mz
        min_intensity = self.min_intensity
        n_peaks = self.n_peaks
        tol = self.remove_precursor_tol

        adduct_mass = HYDROGEN_MASS
        c_mass_diff = ISOTOPE_MASS_DIFF
        isotope = 0

        mz = mz_array.astype(np.float64, copy=False)
        intensity = int_array.astype(np.float32, copy=False)

        try:
            mask = (mz >= min_mz) & (mz <= max_mz)
            if not np.any(mask):
                raise ValueError

            neutral_mass = (precursor_mz - adduct_mass) * precursor_charge

            for charge in range(precursor_charge, 0, -1):
                base = neutral_mass / charge + adduct_mass
                for iso in range(isotope + 1):
                    rm = base + iso * (c_mass_diff / charge)
                    mask &= np.abs(mz - rm) >= tol

            if not np.any(mask):
                raise ValueError

            masked_intensity = intensity[mask]
            if masked_intensity.size == 0:
                raise ValueError

            threshold = masked_intensity.max() * min_intensity
            mask &= intensity >= threshold

            if not np.any(mask):
                raise ValueError

            mz = mz[mask]
            intensity = intensity[mask]

            k = min(n_peaks, mz.shape[0])
            if mz.shape[0] > k:
                idx = np.argpartition(intensity, -k)[-k:]
                mz = mz[idx]
                intensity = intensity[idx]

            order = np.argsort(mz)
            mz = mz[order]
            intensity = intensity[order]

            intensity = np.sqrt(intensity, dtype=np.float32)
            norm = np.linalg.norm(intensity)
            if norm <= 1e-12:
                raise ValueError

            intensity /= norm

            return np.stack((mz, intensity), axis=1).astype(np.float32, copy=False)

        except ValueError:
            return np.array([[0.0, 1.0]], dtype=np.float32)

    def __getitem__(self, idx):
        mz_array, int_array, precursor_mz, precursor_charge, spec_idx = (
            self.dataparser.get_spectrum(idx)
        )
        spectrum = self._process_peaks(mz_array, int_array, precursor_mz, precursor_charge)
        return spectrum, precursor_mz, precursor_charge, spec_idx


class AnnotatedHDFSpectrumDataset(Dataset):
    def __init__(
        self,
        hdf5_folder: str,
        n_peaks: int = 150,
        min_intensity: float = 0.01,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        remove_precursor_tol: float = 2.0,
        random_state: int | None = None,
    ):
        super().__init__()
        self.hdf5_folder = hdf5_folder
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity
        self.dataparser = AnnotatedHDFParser(hdf5_folder)
        self.dataparser.load_data()
        self.n_spectra = len(self.dataparser.global_index_arr)
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return self.n_spectra

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> np.ndarray:
        min_mz = self.min_mz
        max_mz = self.max_mz
        min_intensity = self.min_intensity
        n_peaks = self.n_peaks
        tol = self.remove_precursor_tol

        adduct_mass = HYDROGEN_MASS
        c_mass_diff = ISOTOPE_MASS_DIFF
        isotope = 0

        mz = mz_array.astype(np.float64, copy=False)
        intensity = int_array.astype(np.float32, copy=False)

        try:
            mask = (mz >= min_mz) & (mz <= max_mz)
            if not np.any(mask):
                raise ValueError
            neutral_mass = (precursor_mz - adduct_mass) * precursor_charge

            for charge in range(precursor_charge, 0, -1):
                base = neutral_mass / charge + adduct_mass
                for iso in range(isotope + 1):
                    rm = base + iso * (c_mass_diff / charge)
                    mask &= np.abs(mz - rm) >= tol

            if not np.any(mask):
                raise ValueError
            masked_intensity = intensity[mask]
            if masked_intensity.size == 0:
                raise ValueError

            threshold = masked_intensity.max() * min_intensity
            mask &= intensity >= threshold

            if not np.any(mask):
                raise ValueError
            mz = mz[mask]
            intensity = intensity[mask]
            k = min(n_peaks, mz.shape[0])
            if mz.shape[0] > k:
                idx = np.argpartition(intensity, -k)[-k:]
                mz = mz[idx]
                intensity = intensity[idx]
            order = np.argsort(mz)
            mz = mz[order]
            intensity = intensity[order]
            intensity = np.sqrt(intensity, dtype=np.float32)
            norm = np.linalg.norm(intensity)
            if norm <= 1e-12:
                raise ValueError

            intensity /= norm

            return np.stack((mz, intensity), axis=1).astype(np.float32, copy=False)

        except ValueError:
            return np.array([[0.0, 1.0]], dtype=np.float32)

    def __getitem__(self, idx):
        mz_array, int_array, precursor_mz, precursor_charge, peptide, raw_name, spec_idx, score = (
            self.dataparser.get_spectrum(idx)
        )
        spectrum = self._process_peaks(mz_array, int_array, precursor_mz, precursor_charge)
        return spectrum, precursor_mz, precursor_charge, peptide, raw_name, spec_idx, score


def prepare_batch(batch):
    first_element = batch[0]
    is_annotated = len(first_element) == 7
    if is_annotated:
        spectra, precursor_mzs, precursor_charges, peptides, raw_names, spec_idx, score = list(
            zip(*batch, strict=False)
        )
    else:
        spectra, precursor_mzs, precursor_charges, spec_idx = list(zip(*batch, strict=False))

    spectra = [torch.from_numpy(s) for s in spectra]
    spectra = pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges, precursor_mzs]).T.float()

    if is_annotated:
        return (
            spectra,
            precursors,
            np.array(peptides, dtype=object),
            np.array(raw_names, dtype=object),
            np.array(spec_idx),
            np.array(score),
        )
    else:
        return spectra, precursors, np.array(spec_idx)


class ChunkedWeightedSampler(torch.utils.data.Sampler):
    """Chunked weighted sampler for large datasets that cannot fit weights in memory
    Ori WeightedRandomSampler can't handel dataset with more than 1800w sample"""

    def __init__(self, weights, num_samples, chunk_size=10000000, replacement=True):
        self.weights = weights
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.replacement = replacement

        self.n = len(weights)

        self.chunks = []
        for start in range(0, self.n, chunk_size):
            end = min(start + chunk_size, self.n)
            self.chunks.append((start, end))

    def __iter__(self):
        samples = []

        for start, end in self.chunks:
            w = self.weights[start:end]

            k = int(self.num_samples * (end - start) / self.n)

            if k == 0:
                continue

            idx = torch.multinomial(
                w,
                num_samples=k,
                replacement=self.replacement,
            )

            samples.append(idx + start)

        if len(samples) == 0:
            return iter([])

        samples = torch.cat(samples)

        if len(samples) < self.num_samples:
            extra = torch.randint(0, self.n, (self.num_samples - len(samples),))
            samples = torch.cat([samples, extra])

        perm = torch.randperm(len(samples))
        samples = samples[perm]

        return iter(samples.tolist())

    def __len__(self):
        return self.num_samples


class DeNovoDataModule:
    def __init__(
        self,
        train_folder: str | None = None,
        val_folder: str | None = None,
        test_path: str | None = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1024,
        n_peaks: int | None = 150,
        min_mz: float = 150,
        max_mz: float = 2500,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: int | None = 0,  # win 0, linux 16
        random_state: int | None = 454,
        annotated=True,
        eval_subset_ratio: float = 0.1,
        weighted_sample: bool = True,
        chunked_weighted_sample: bool = False,
    ):
        super().__init__()
        self._seed = random_state
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_path = test_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers
        self.rng = np.random.default_rng(random_state)
        self.annotated = annotated
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.weighted_sample = weighted_sample
        self.chunked_weighted_sample = chunked_weighted_sample
        self.eval_subset_ratio = eval_subset_ratio
        self.eval_subset_seed = random_state
        self.train_eval_dataset = None
        self.sampler = None
        self.dataset_kwargs = {
            "n_peaks": n_peaks,
            "min_mz": min_mz,
            "max_mz": max_mz,
            "min_intensity": min_intensity,
            "remove_precursor_tol": remove_precursor_tol,
            "random_state": random_state,
        }

    def setup(self):
        if self._seed is not None:
            random.seed(self._seed)
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

        dataset_cls = AnnotatedHDFSpectrumDataset if self.annotated else HDFSpectrumDataset

        if self.train_folder is not None:
            self.train_dataset = dataset_cls(self.train_folder, **self.dataset_kwargs)
            self._create_train_eval_subset()
            if self.weighted_sample:
                logger.info("Sample_weighted based on charge_modified_sequence......")
                self.sampler = WeightedRandomSampler(
                    self.train_dataset.dataparser.peptide_weights,
                    num_samples=len(self.train_dataset.dataparser.peptide_weights),
                    replacement=True,
                )
            elif self.chunked_weighted_sample:
                logger.info("Chunked_sample_weighted based on charge_modified_sequence......")
                self.sampler = ChunkedWeightedSampler(
                    self.train_dataset.dataparser.peptide_weights,
                    num_samples=len(self.train_dataset.dataparser.peptide_weights),
                    chunk_size=10000000,
                    replacement=True,
                )
            logger.info("Training dataset initialized with %s spectra", len(self.train_dataset))
            logger.info(
                "Fixed train eval subset created with %s spectra", len(self.train_eval_dataset)
            )

        if self.val_folder is not None:
            self.valid_dataset = dataset_cls(self.val_folder, **self.dataset_kwargs)
            logger.info("Validation dataset initialized with %s spectra", len(self.valid_dataset))

        if self.test_path is not None:
            self.test_dataset = dataset_cls(self.test_path, **self.dataset_kwargs)
            logger.info("Test dataset initialized with %s spectra", len(self.test_dataset))

    def get_loader(self, dataset, batch_size, shuffle=False, sampler=None):
        def worker_init_fn(worker_id):
            if self._seed is not None:
                worker_seed = self._seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if sampler is None else False),
            collate_fn=prepare_batch,
            sampler=sampler,
            pin_memory=True,
            num_workers=self.n_workers,
            worker_init_fn=worker_init_fn if self._seed is not None else None,
        )

    def get_train_loader(self):
        if self.sampler:
            return self.get_loader(
                self.train_dataset, self.train_batch_size, shuffle=False, sampler=self.sampler
            )
        else:
            return self.get_loader(self.train_dataset, self.train_batch_size, shuffle=True)

    def get_val_loader(self):
        return self.get_loader(self.valid_dataset, self.eval_batch_size, shuffle=False)

    def get_test_loader(self):
        return self.get_loader(self.test_dataset, self.eval_batch_size, shuffle=False)

    def get_train_eval_loader(self):
        return self.get_loader(
            self.train_eval_dataset, self.eval_batch_size, shuffle=False, sampler=None
        )

    def _create_train_eval_subset(self):
        rng = np.random.default_rng(self.eval_subset_seed)
        total_size = len(self.train_dataset)
        eval_size = int(total_size * self.eval_subset_ratio)

        eval_indices = rng.choice(total_size, size=eval_size, replace=False)
        eval_indices = sorted(eval_indices)

        self.train_eval_dataset = torch.utils.data.Subset(self.train_dataset, eval_indices)
        self.eval_indices = eval_indices
