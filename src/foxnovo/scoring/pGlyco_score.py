import logging
import re
import warnings

import numba as nb
import numpy as np
import pandas as pd
from alphabase.io.hdf import HDF_File
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from alphabase.peptide.precursor import refine_precursor_df
from peptdeep.mass_spec.match import match_one_raw_with_numba

from foxnovo.constants import WATER_MASS
from foxnovo.model.config import AA_MASS, MOD_TO_AA_TOKEN

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@nb.jit(nopython=True, parallel=True)
def calculate_score_numba(
    matched_intensities,
    matched_mz_err_ppm,
    frag_start_idxes,
    frag_stop_idxes,
    ppm,
    gamma,
    is_b_col,
):
    n_sequences = len(frag_start_idxes)
    n_frag_types = matched_intensities.shape[1]
    scores = np.zeros(n_sequences, dtype=np.float64)
    matched_counts = np.zeros(n_sequences, dtype=np.int32)
    matched_ratios = np.zeros(n_sequences, dtype=np.float64)
    b_matched_counts = np.zeros(n_sequences, dtype=np.int32)
    y_matched_counts = np.zeros(n_sequences, dtype=np.int32)
    b_ratios = np.zeros(n_sequences, dtype=np.float64)
    y_ratios = np.zeros(n_sequences, dtype=np.float64)

    for i in nb.prange(n_sequences):
        start = frag_start_idxes[i]
        stop = frag_stop_idxes[i]
        seq_intensities = matched_intensities[start:stop]
        seq_errs = matched_mz_err_ppm[start:stop]
        n_frag = stop - start

        if n_frag == 0:
            continue

        row_has_any = np.zeros(n_frag, dtype=np.bool_)
        row_has_b = np.zeros(n_frag, dtype=np.bool_)
        row_has_y = np.zeros(n_frag, dtype=np.bool_)

        total_score = 0.0
        for row in range(n_frag):
            for col in range(n_frag_types):
                if seq_intensities[row, col] > 0 and np.abs(seq_errs[row, col]) <= ppm:
                    err = seq_errs[row, col]
                    intensity = seq_intensities[row, col]
                    err_ratio = np.abs(err) / ppm
                    penalty = 1 - (err_ratio**4)
                    total_score += np.log(intensity) * penalty

                    row_has_any[row] = True
                    if is_b_col[col]:
                        row_has_b[row] = True
                    else:
                        row_has_y[row] = True

        matched_count = np.sum(row_has_any)
        b_count = np.sum(row_has_b)
        y_count = np.sum(row_has_y)

        matched_ratio = matched_count / n_frag if n_frag > 0 else 0.0
        b_ratio = b_count / n_frag if n_frag > 0 else 0.0
        y_ratio = y_count / n_frag if n_frag > 0 else 0.0

        total_score *= matched_ratio**gamma

        scores[i] = total_score
        matched_counts[i] = matched_count
        matched_ratios[i] = matched_ratio
        b_matched_counts[i] = b_count
        y_matched_counts[i] = y_count
        b_ratios[i] = b_ratio
        y_ratios[i] = y_ratio

    return (
        scores,
        matched_counts,
        matched_ratios,
        b_matched_counts,
        y_matched_counts,
        b_ratios,
        y_ratios,
    )


def filter_mass_match(
    df,
    mass_tol,
    mass_error,
    sequence_mass_col="modified_sequence mass",
    precursor_mass_col="precursor_mass",
):
    """Keep only rows whose peptide mass matches the precursor within tolerance."""
    df_copy = df.copy(deep=True)
    required_cols = [sequence_mass_col, precursor_mass_col]
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_cols)}")
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            raise TypeError(f"Column {col} must be numeric (int/float), got {df_copy[col].dtype}")
    offsets = np.arange(-mass_tol, mass_tol + 1)
    precursor_masses = df_copy[precursor_mass_col].values[:, None]
    shifted_masses = precursor_masses + offsets
    sequence_masses = df_copy[sequence_mass_col].values[:, None]
    ppm_errors = np.abs(shifted_masses - sequence_masses) / sequence_masses * 1e6
    df_copy["mass_match"] = (ppm_errors <= mass_error).any(axis=1)
    result_df = df_copy[df_copy["mass_match"]].reset_index(drop=True)

    return result_df


class DenovoSequenceScoring:
    def __init__(self, sequences_df, hdf_path):
        f = HDF_File(file_name=hdf_path, read_only=True)
        try:
            spectrum_df = f.ms_data.spectrum_df.values
            if "charge" not in spectrum_df.columns and "precursor_charge" in spectrum_df.columns:
                spectrum_df["charge"] = spectrum_df["precursor_charge"]
            self.spectra_df = spectrum_df[
                [
                    "charge",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "spec_idx",
                    "ms_level",
                ]
            ]
            self.spectra_df = self.spectra_df[self.spectra_df["ms_level"] == 2]

        except Exception:
            logger.exception("Failed to extract MS2 spectrum dataframe from f.ms_data.spectrum_df")
            raise

        self.peak_df = f.ms_data.peak_df.values
        self.mode2mass = MOD_TO_AA_TOKEN.copy()
        self.aa2mass = AA_MASS.copy()
        self.WATER_MW = WATER_MASS
        self.sorted_aas = sorted(self.aa2mass.keys(), key=len, reverse=True)
        self.max_aa_len = max(len(aa) for aa in self.aa2mass)
        self.mass2mod = {}
        for mod, mass_str in self.mode2mass.items():
            for mass in re.findall(r"[+-]\d+\.\d+", mass_str):
                self.mass2mod[mass] = mod
        self.top_k = sequences_df.shape[1] - 1
        self.sequences_df = sequences_df
        self.sequences_df["modified_sequence"] = self.sequences_df["modified_sequence"].replace(
            "", np.nan
        )
        self.sequences_df = self.sequences_df.dropna(subset=["modified_sequence"]).reset_index(
            drop=True
        )

        self.parse_modified_sequences()
        self.mass_tol = 2
        self.ppm = 20.0

    def calculate_peptide_mw(self, peptide: str) -> float:
        """Calculate peptide molecular weight from FoxNovo residue tokens."""
        total_mass = 0.0
        pos = 0
        peptide_len = len(peptide)

        while pos < peptide_len:
            matched = False
            max_check_len = min(self.max_aa_len, peptide_len - pos)

            for check_len in range(max_check_len, 0, -1):
                aa_candidate = peptide[pos : pos + check_len]
                if aa_candidate in self.aa2mass:
                    total_mass += self.aa2mass[aa_candidate]
                    pos += check_len
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Peptide '{peptide}' contains an undefined residue near "
                    f"'{peptide[pos:pos + 3]}...'"
                )
        return total_mass + self.WATER_MW

    def add_mw_to_df(
        self,
        df: pd.DataFrame,
        peptide_col: str = "modified_sequence",
        mw_col: str = "modified_sequence mass",
    ):
        """Add a molecular-weight column to a DataFrame."""
        df[mw_col] = df[peptide_col].apply(self.calculate_peptide_mw).round(2).astype(np.float64)
        return df

    def calculate_precursor_mass(
        self, df: pd.DataFrame, precursor_mz_col: str = "precursor_mz", charge_col: str = "charge"
    ):
        df["precursor_mass"] = (
            (df[precursor_mz_col] * df[charge_col] - df[charge_col] * 1.00728)
            .round(2)
            .astype(np.float64)
        )
        return df

    def parse_modified_sequences(self):
        """Split modified sequences into base sequence, mods, and mod sites."""

        def parse_seq(seq):
            pure = []
            mods = []
            positions = []
            i = 0
            while i < len(seq):
                match = re.match(r"^([+-]\d+\.\d+)", seq[i:])
                if match:
                    mass = match.group(1)
                    mods.append(self.mass2mod.get(mass, f"Unknown{mass}"))
                    positions.append(str(len(pure)))
                    i += len(mass)
                else:
                    pure.append(seq[i])
                    i += 1
            return "".join(pure), ";".join(mods), ";".join(positions)

        result = self.sequences_df["modified_sequence"].apply(lambda x: parse_seq(str(x)))
        self.sequences_df[["sequence", "mods", "mod_sites"]] = pd.DataFrame(
            result.tolist(), index=self.sequences_df.index
        )

    def sequence_spectra_match(self):
        self.sequences_df["spec_idx"] = self.sequences_df["spec_idx"].astype(str)
        self.spectra_df["spec_idx"] = self.spectra_df["spec_idx"].astype(str)
        self.sequences_df = pd.merge(
            self.sequences_df,
            self.spectra_df[
                ["charge", "peak_start_idx", "peak_stop_idx", "precursor_mz", "spec_idx"]
            ],
            on="spec_idx",
            how="inner",
        )
        psm_df = refine_precursor_df(self.sequences_df)
        psm_df["row_idx"] = range(len(psm_df))
        charged_frag_types = ["b_z1", "y_z1", "b_z2", "y_z2"]
        fragment_mz_df = create_fragment_mz_dataframe(psm_df, charged_frag_types)
        self.fragment_mz_df = fragment_mz_df
        psm_df["spec_idx"] = psm_df["spec_idx"].astype(np.int64)
        psm_df["frag_start_idx"] = psm_df["frag_start_idx"].astype(np.int64)
        psm_df["frag_stop_idx"] = psm_df["frag_stop_idx"].astype(np.int64)
        psm_df = self.add_mw_to_df(
            psm_df, peptide_col="modified_sequence", mw_col="modified_sequence mass"
        )
        psm_df = self.calculate_precursor_mass(
            psm_df, precursor_mz_col="precursor_mz", charge_col="charge"
        )
        all_spec_mzs = self.peak_df["mz"].values.astype(np.float64)
        all_spec_intensities = self.peak_df["intensity"].values.astype(np.float64)
        peak_start_idxes = self.sequences_df["peak_start_idx"].values.astype(np.int64)
        peak_stop_idxes = self.sequences_df["peak_stop_idx"].values.astype(np.int64)

        all_frag_mzs = fragment_mz_df.values.astype(np.float64)

        matched_intensities = np.zeros_like(all_frag_mzs, dtype=np.float64)
        matched_mz_errs = np.full_like(all_frag_mzs, np.inf, dtype=np.float64)
        match_one_raw_with_numba(
            spec_idxes=psm_df["row_idx"].values,
            frag_start_idxes=psm_df["frag_start_idx"].values,
            frag_stop_idxes=psm_df["frag_stop_idx"].values,
            all_frag_mzs=all_frag_mzs,
            all_spec_mzs=all_spec_mzs,
            all_spec_intensities=all_spec_intensities,
            peak_start_idxes=peak_start_idxes,
            peak_end_idxes=peak_stop_idxes,
            matched_intensities=matched_intensities,
            matched_mz_errs=matched_mz_errs,
            ppm=True,
            tol=self.ppm,
        )

        self.matched_intensity_df = pd.DataFrame(
            matched_intensities, columns=fragment_mz_df.columns
        )

        self.matched_mz_err_df = pd.DataFrame(matched_mz_errs, columns=fragment_mz_df.columns)

        self.matched_mz_err_ppm_df = pd.DataFrame(
            (matched_mz_errs / all_frag_mzs) * 1e6, columns=fragment_mz_df.columns
        )

    def filter_score_and_mass(self):
        """Filter low-score rows and drop temporary matching columns."""
        self.filtered_sequences_df = self.sequences_df.copy(deep=True)
        self.filtered_sequences_df = self.filtered_sequences_df[
            self.filtered_sequences_df["score"] >= 0.00001
        ]
        drop_cols = [
            "peak_start_idx",
            "peak_stop_idx",
            "row_idx",
            "frag_start_idx",
            "frag_stop_idx",
        ]
        existing_cols = [col for col in drop_cols if col in self.filtered_sequences_df.columns]
        if existing_cols:
            self.filtered_sequences_df.drop(columns=existing_cols, inplace=True)

    def pGlyco_scoring_numba(self, gamma=0.94):
        ppm = self.ppm
        matched_intensities = self.matched_intensity_df.values.astype(np.float64)
        matched_mz_err_ppm = self.matched_mz_err_ppm_df.values.astype(np.float64)
        frag_start_idxes = self.sequences_df["frag_start_idx"].values.astype(np.int64)
        frag_stop_idxes = self.sequences_df["frag_stop_idx"].values.astype(np.int64)

        cols = self.matched_intensity_df.columns
        is_b_col = np.array([col.startswith("b") for col in cols], dtype=np.bool_)

        (scores, matched_counts, matched_ratios, b_counts, y_counts, b_ratios, y_ratios) = (
            calculate_score_numba(
                matched_intensities,
                matched_mz_err_ppm,
                frag_start_idxes,
                frag_stop_idxes,
                ppm,
                gamma,
                is_b_col,
            )
        )

        self.sequences_df["score"] = scores.round(2).astype(np.float64)
        self.sequences_df["matched_ion_count"] = matched_counts
        self.sequences_df["matched_ion_ratio"] = matched_ratios.round(2).astype(np.float64)
        self.sequences_df["b_matched_ion_count"] = b_counts
        self.sequences_df["y_matched_ion_count"] = y_counts
        self.sequences_df["b_matched_ion_ratio"] = b_ratios.round(2).astype(np.float64)
        self.sequences_df["y_matched_ion_ratio"] = y_ratios.round(2).astype(np.float64)


def score_sequence(sequences_df, hdf_path):
    scoring_module = DenovoSequenceScoring(sequences_df, hdf_path)
    scoring_module.sequence_spectra_match()
    scoring_module.pGlyco_scoring_numba()
    scoring_module.filter_score_and_mass()
    return scoring_module.sequences_df, scoring_module.filtered_sequences_df
