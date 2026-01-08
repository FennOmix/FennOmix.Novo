from alphabase.io.hdf import HDF_File
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import pandas as pd
import spectrum_utils.spectrum as sus
from typing import Optional
import os
from pathlib import Path
from alpharaw.thermo import ThermoRawData
from alpharaw.ms_data_base import ms_reader_provider
from collections import Counter
'''训练集，验证集：经过dataset_splitter.py 脚本划分的hdf文件'''
'''测试机提供路径，遍历加载'''

'''todo:
1. 添加无注释谱图读取 √
2. 添加RT> 已添加rt_norm ✔
3. 判断加载是否正确： MS2等 ✔
4. 处理至固定长度序列：只会填充至batch内最大长度，不会填充至max_peaks ✔
5.多文件hdf处理 √： 目前直接加载所有数据至内存
6. 写入完整模型 ✔
7. 增加异常处理
8. 其余类型数据读取：先转为hdf，再接入hdf读取或直接alpharaw读取'''


mode2mass = {
    "Carbamidomethyl@C": "C+57.021",
    "Oxidation@M": "M+15.995",
    "Deamidated@N": "N+0.984",
    "Deamidated@Q": "Q+0.984",
    "Acetyl@Protein N-term": "+42.011",
    "AEBS@Y": "Y+183.035",
    "AEBS@K": "K+183.035",
    "Glu->pyro-Glu@E^Any_N-term": "-18.011", #before nterm E
    "Gln->pyro-Glu@Q^Any_N-term": "-17.027", #before nterm Q
    "Cysteinyl@C": "C+119.004"
}
ignore_mod = ["Deamidated@N", "Deamidated@Q", "AEBS@Y", "AEBS@K", "Glu->pyro-Glu@E^Any_N-term", "Gln->pyro-Glu@Q^Any_N-term"] #调整数据集：保留数据，但认为不存在该修饰(序列层面)
allow_ignore_mod_num = 1 #带有ignore_mod的sequence会变为纯序列，但带有多于allow_ignore_mod_num的序列会直接删除

def filter_ignore_mods(psm_df, mods_column, ignore_mod):
    initial_count = psm_df.shape[0]

    # 定义函数，判断每个mods字符串中包含的ignore_mod数量是否大于1
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

    # 筛选出不满足条件的行（即保留mods中包含ignore_mod数量小于等于1的行）
    filtered_df = psm_df[~psm_df[mods_column].apply(has_more_than_one_ignore)]

    # 统计删除的行数
    deleted_count = initial_count - filtered_df.shape[0]

    return filtered_df, deleted_count
def alpha_raw_reader(file_path, file_type):
    "简单的多类型原始数据MS2加载器"
    "todo: 添加异常处理，支持mgf/timstof data"
    if file_type == 'hdf5':
        f = HDF_File(file_name=file_path, read_only=True)
        spectrum_df = f.psm.psm_df.values
        peak_df = f.ms_data.peak_df.values
    elif file_type == 'raw':
        raw_data = ThermoRawData()
        raw_data.import_raw(file_path)
        spectrum_df = raw_data.spectrum_df
        spectrum_df.rename(columns={'precursor_charge': 'charge'}, inplace=True) #与hdf文件统一
        peak_df = raw_data.peak_df
    elif file_type == 'mzml':
        mzml_reader = ms_reader_provider.get_reader("mzml")
        mzml_reader.import_raw(file_path)
        spectrum_df = mzml_reader.spectrum_df
        spectrum_df.rename(columns={'precursor_charge': 'charge'}, inplace=True)
        peak_df = mzml_reader.peak_df
    else:
        print('Unknow raw data type, please make sure your raw data is Thermo raw/hdf5/mzml!')
    return spectrum_df, peak_df


def build_mod_seq(sequences, mods_list, sites_list, mod_dict):
    """构建带修饰的肽段序列"""
    #todo: mods_split 和 sites_split 长度一致校验
    result = []
    mass_dict = {k: '+' + v.split('+')[-1] for k, v in mod_dict.items()}

    for seq, mods, sites in zip(sequences, mods_list, sites_list):
        if not mods or pd.isna(mods) or mods.strip() == '':
            result.append(seq)
            continue

        mods_split = mods.split(';')
        sites_split = list(map(int, sites.split(';')))

        # 预构造：每个位置可能要插入的 mass
        mod_map = {}  # site: mass_tag
        for mod, site in zip(mods_split, sites_split):
            if mod in ignore_mod: #过滤ignore mod
                continue
            tag = mass_dict.get(mod, '')
            if tag:
                mod_map[site - 1] = tag

        # 构造修饰序列（一次性拼接）
        parts = []
        for i, aa in enumerate(seq):
            parts.append(aa)
            if i in mod_map:
                parts.append(mod_map[i])
        result.append(''.join(parts))
    return result

class HDFParser:
    "test_data"
    "针对单个hdf，无标签"
    def __init__(self, hdf5_path: str):
        self.data = {}
        self.hdf5_path = hdf5_path
    def load_data(self):
        f = HDF_File(file_name=self.hdf5_path, read_only=True)
        try:
            psm_df = f.ms_data.spectrum_df.values[['charge','peak_start_idx','peak_stop_idx','precursor_mz','spec_idx', 'ms_level']] #此处为ms.data 可能也会是raw.data
        except:
            psm_df = f.psm.psm_df.values[['charge','peak_start_idx','peak_stop_idx','precursor_mz','spec_idx']]
        psm_df = psm_df[psm_df['ms_level'] == 2]
        psm_df = psm_df.astype({'charge': int, 'peak_start_idx': int, 'peak_stop_idx': int, 'spec_idx': int})
        peak_df = f.ms_data.peak_df.values
        mz_array = peak_df['mz'].to_numpy()
        intensity_array = peak_df['intensity'].to_numpy()

        self.data = {
            "psm_df": psm_df,
            "mz_array":mz_array,
            "intensity_array": intensity_array
        }

    def get_spectrum(self, idx: int):
        "可能的问题：效率可能低"
        spectrum_info = self.data['psm_df'].iloc[idx]
        start_idx_in_peak_df = int(spectrum_info['peak_start_idx'])
        stop_idx_in_peak_dx = int(spectrum_info['peak_stop_idx'])

        mz = self.data["mz_array"][start_idx_in_peak_df:stop_idx_in_peak_dx]
        intensity = self.data["intensity_array"][start_idx_in_peak_df:stop_idx_in_peak_dx]

        precursor_mz = spectrum_info['precursor_mz']
        precursor_charge = int(spectrum_info['charge'])
        spec_idx = int(spectrum_info['spec_idx'])
        return mz, intensity, precursor_mz, precursor_charge, spec_idx




class AnnotatedHDFParser:
    "train/val_data"
    "针对完整文件夹(多个hdf)，有标签"

    def __init__(self, hdf5_folder: str):
        self.data = {}
        self.hdf5_folder = hdf5_folder
        self.global_index = [] #(file_name & self.data key value, spec_idx) #构造一个idx2data函数
    "加载给定标签hdf"
    def load_data(self):
        folder_path = Path(self.hdf5_folder)
        self.files = sorted(folder_path.glob("*.hdf5"))
        all_sequences = []  # 保存所有样本的 peptide 序列，用于统计频率
        delete_multiple_ignore_mod_count = 0
        for file_idx, file_path in enumerate(self.files):
            f = HDF_File(file_name=file_path, read_only=True)
            psm_df = f.psm.psm_df.values[['charge','mods','mod_sites','nAA','peak_start_idx','peak_stop_idx','precursor_mz','sequence','raw_name','spec_idx','score']]
            peak_df = f.ms_data.peak_df.values
            psm_df['modified_sequence'] = build_mod_seq(
                psm_df['sequence'],
                psm_df['mods'],
                psm_df['mod_sites'],
                mod_dict=mode2mass
            )

            mz_array = peak_df['mz'].to_numpy()
            intensity_array = peak_df['intensity'].to_numpy()

            for scan_idx, spec_idx in enumerate(psm_df['spec_idx']):
                self.global_index.append((file_path, scan_idx))
                all_sequences.append(psm_df.iloc[scan_idx]['modified_sequence'])
            self.data[file_path] = [psm_df, mz_array, intensity_array]
        # === 统计频率并生成 weights ===
        freq_counter = Counter(all_sequences)
        weights = []
        for seq in all_sequences:
            f = freq_counter[seq]
            w = 1.0 / (np.sqrt(f))
            weights.append(w)
        self.peptide_weights = np.array(weights, dtype=np.float32)

    def get_spectrum(self, idx: int):
        "给定索引，读取一个ms2，返回数据"
        "可能的问题：效率可能低"
        spectrum_info = self.data[self.global_index[idx][0]][0].iloc[self.global_index[idx][1]]
        start_idx_in_peak_df = spectrum_info['peak_start_idx']
        stop_idx_in_peak_dx = spectrum_info['peak_stop_idx']

        mz = self.data[self.global_index[idx][0]][1][start_idx_in_peak_df:stop_idx_in_peak_dx]
        intensity = self.data[self.global_index[idx][0]][2][start_idx_in_peak_df:stop_idx_in_peak_dx]

        precursor_mz = spectrum_info['precursor_mz']
        precursor_charge = spectrum_info['charge']
        sequence = spectrum_info['modified_sequence']
        raw_name = spectrum_info['raw_name']
        spec_idx = spectrum_info['spec_idx']
        score = spectrum_info['score']

        return mz, intensity, precursor_mz, precursor_charge, sequence, raw_name, spec_idx, score # item: 8


class HDFSpectrumDataset(Dataset):
    "test_data"
    def __init__(
            self,
            hdf5_path: str,
            n_peaks: int = 150,
            min_intensity: float = 0.01,
            min_mz: float = 140.0,
            max_mz: float = 2500.0,
            remove_precursor_tol: float = 2.0,
            random_state: Optional[int] = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity
        self.dataparser = HDFParser(hdf5_path)
        self.dataparser.load_data()
        self.n_spectra = len(self.dataparser.data['psm_df'])
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol

    def __len__(self):
        return self.n_spectra

    def _process_peaks(
            self,
            mz_array: np.ndarray,
            int_array: np.ndarray,
            precursor_mz: float,
            precursor_charge: int
    ) -> torch.Tensor:
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            # print('谱峰异常，报错于process_peaks模块')
            return torch.tensor([[0, 1]]).float()

    def __getitem__(self, idx):
        mz_array, int_array, precursor_mz, precursor_charge, spec_idx = self.dataparser.get_spectrum(idx)
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )

        return spectrum, precursor_mz, precursor_charge, spec_idx

class AnnotatedHDFSpectrumDataset(Dataset):
    "train/val"
    def __init__(
            self,
            hdf5_folder:str,
            n_peaks: int = 150,
            min_intensity: float = 0.01,
            min_mz: float = 140.0,
            max_mz: float = 2500.0,
            remove_precursor_tol: float = 2.0,
            random_state: Optional[int] = None,
    ):
        super().__init__()
        self.hdf5_folder = hdf5_folder
        self.n_peaks = n_peaks
        self.min_intensity = min_intensity
        self.dataparser = AnnotatedHDFParser(hdf5_folder)
        self.dataparser.load_data() #此处执行IO操作，后期注意是否会频繁IO
        self.n_spectra = len(self.dataparser.global_index)
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return self.n_spectra

    def _process_peaks(
            self,
            mz_array: np.ndarray,
            int_array: np.ndarray,
            precursor_mz: float,
            precursor_charge: int
    ) -> torch.Tensor:
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            return torch.tensor([[0, 1]]).float()

    def __getitem__(self,idx):
        mz_array, int_array, precursor_mz, precursor_charge, peptide, raw_name, spec_idx, score = self.dataparser.get_spectrum(idx)
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )

        return spectrum, precursor_mz, precursor_charge, peptide, raw_name, spec_idx, score


def prepare_batch(batch):
    #根据输入
    first_element = batch[0]
    is_annotated = len(first_element) == 7
    if is_annotated:
        spectra, precursor_mzs, precursor_charges, peptides, raw_names, spec_idx, score = list(zip(*batch))
    else:
        spectra, precursor_mzs, precursor_charges, spec_idx = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    if is_annotated:
        return spectra, precursors, np.array(peptides,dtype=object), np.array(raw_names, dtype=object), np.array(spec_idx), np.array(score)
    else:
        return spectra, precursors, np.array(spec_idx)


class DeNovoDataModule():
    #训练、验证集传入文件夹：一次性加载所有hdf5
    #测试集循环传入路径，逐次处理hdf5
    def __init__(
            self,
            train_folder: Optional[str] = None,  # 添加路径参数
            val_folder: Optional[str] = None,
            test_path: Optional[str] = None, #path, not folder
            train_batch_size:int = 128,
            eval_batch_size:int = 1024,
            n_peaks: Optional[int] = 150,
            min_mz: float = 150,
            max_mz: float = 2500,
            min_intensity: float = 0.01,
            remove_precursor_tol: float = 2.0,
            n_workers: Optional[int] = 1,
            random_state: Optional[int] = None,
            annotated=True):
        super().__init__()
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_path = test_path

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_workers = n_workers if n_workers is not None else os.cpu_count() // 4
        self.rng = np.random.default_rng(random_state)
        self.annotated = annotated
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.weighted_sample = True
        self.dataset_kwargs = dict(
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            random_state=random_state,
        )

    def setup(self):
        dataset_cls = AnnotatedHDFSpectrumDataset if self.annotated else HDFSpectrumDataset
        try:
            if self.train_folder is not None:
                self.train_dataset = dataset_cls(self.train_folder, **self.dataset_kwargs)
                if self.weighted_sample:
                    self.sampler = WeightedRandomSampler(self.train_dataset.dataparser.peptide_weights,
                                                         num_samples=len(self.train_dataset.dataparser.peptide_weights), replacement=True)
                print("Training dataset initialized with", len(self.train_dataset), "spectra")

            if self.val_folder is not None:
                self.valid_dataset = dataset_cls(self.val_folder, **self.dataset_kwargs)
                print("Validation dataset initialized with", len(self.valid_dataset), "spectra")

            if self.test_path is not None:
                self.test_dataset = dataset_cls(self.test_path, **self.dataset_kwargs)
                print("Test dataset initialized with", len(self.test_dataset), "spectra")
        except:
            raise ValueError("Failed to initialize datasets. Please check the file paths and dataset parameters.")

    def get_loader(self, dataset, batch_size, shuffle=False, sampler=None):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if sampler is None else False),
            collate_fn=prepare_batch,
            sampler=sampler,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def get_train_loader(self):
        if self.sampler:
            return self.get_loader(self.train_dataset, self.train_batch_size, shuffle=True, sampler=self.sampler)
        else:
            return self.get_loader(self.train_dataset, self.train_batch_size, shuffle=True)

    def get_val_loader(self):
        return self.get_loader(self.valid_dataset, self.eval_batch_size, shuffle=False)

    def get_test_loader(self):
        return self.get_loader(self.test_dataset, self.eval_batch_size, shuffle=False)




