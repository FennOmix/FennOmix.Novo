import re
import warnings

import numba as nb
import numpy as np
import pandas as pd
from alphabase.io.hdf import HDF_File
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from alphabase.peptide.precursor import refine_precursor_df
from peptdeep.mass_spec.match import match_one_raw_with_numba

warnings.filterwarnings("ignore", category=RuntimeWarning)


@nb.jit(nopython=True, parallel=True)
def calculate_score_numba(
    matched_intensities,
    matched_mz_err_ppm,
    frag_start_idxes,
    frag_stop_idxes,
    ppm,
    gamma,
    is_b_col,  # bool array of shape (n_frag_types,)
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

        # 最终分数（仍按总匹配比例加权）
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
    """
    根据mass_tol（Da偏移）和mass_error（ppm误差）过滤DataFrame，保留满足质量匹配的行

    参数说明：
        df (pd.DataFrame): 输入的原始DataFrame，需包含序列质量和母离子质量列
        mass_tol (int): Da水平的偏移容忍度（整数），会生成 [-mass_tol, ..., 0, ..., +mass_tol] 的偏移量
        mass_error (float): ppm水平的误差容忍度（浮点数），允许的最大ppm误差
        sequence_mass_col (str): 序列质量列的列名（默认："sequence_mass"）
        precursor_mass_col (str): 母离子质量列的列名（默认："precursor_mass"）

    返回值：
        pd.DataFrame: 仅保留质量匹配（mass_match=True）的行，包含原始列和新增的mass_match列
    """
    df_copy = df.copy(deep=True)
    required_cols = [sequence_mass_col, precursor_mass_col]
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列：{', '.join(missing_cols)}")
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            raise TypeError(f"{col}列必须是数值类型（int/float），当前类型：{df_copy[col].dtype}")
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
            self.spectra_df = f.ms_data.spectrum_df.values[
                [
                    "precursor_charge",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "spec_idx",
                    "ms_level",
                ]
            ]
            self.spectra_df["charge"] = self.spectra_df["precursor_charge"]
        except:  # noqa: E722
            self.spectra_df = f.psm.psm_df.values[
                ["charge", "peak_start_idx", "peak_stop_idx", "precursor_mz", "spec_idx"]
            ]

        self.peak_df = f.ms_data.peak_df.values
        self.mode2mass = {
            "Carbamidomethyl@C": "C+57.021",
            "Oxidation@M": "M+15.995",
            "Cysteinyl@C": "C+119.004",
        }
        "-------------肽段分子量计算------------"
        self.aa2mass = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C+57.021": 160.030649,
            "L": 113.084064,
            "I": 113.084064,
            "C": 103.009649,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            "M+15.995": 147.035400,
            "C+119.004": 222.014,
        }
        self.WATER_MW = 18.01056
        self.sorted_aas = sorted(self.aa2mass.keys(), key=len, reverse=True)
        self.max_aa_len = max(len(aa) for aa in self.aa2mass)
        "-------------肽段分子量计算------------"
        self.mass2mod = {}
        for mod, mass_str in self.mode2mass.items():
            for mass in re.findall(r"[+-]\d+\.\d+", mass_str):
                self.mass2mod[mass] = mod
        self.top_k = sequences_df.shape[1] - 1  # -1: spec_idx
        self.sequences_df = sequences_df
        self.sequences_df["modified_sequence"] = self.sequences_df["modified_sequence"].replace(
            "", np.nan
        )
        self.sequences_df = self.sequences_df.dropna(subset=["modified_sequence"]).reset_index(
            drop=True
        )

        self.parse_modified_sequences()  # 拆分修饰序列为sequence、mod、mod_site
        self.mass_tol = 2
        self.ppm = 20.0

    "------分子量计算------"

    def calculate_peptide_mw(self, peptide: str) -> float:
        """计算单个肽段分子量（逻辑简化，直观易懂）"""
        total_mass = 0.0
        pos = 0  # 当前匹配位置
        peptide_len = len(peptide)

        while pos < peptide_len:
            matched = False
            # 从最长缩写开始尝试匹配（避免短缩写覆盖长缩写）
            # 截取当前位置开始的最长可能子串（不超过max_aa_len）
            max_check_len = min(self.max_aa_len, peptide_len - pos)

            for check_len in range(max_check_len, 0, -1):
                aa_candidate = peptide[pos : pos + check_len]
                if aa_candidate in self.aa2mass:
                    total_mass += self.aa2mass[aa_candidate]
                    pos += check_len  # 跳过已匹配的字符
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"肽段 '{peptide}' 中存在未定义的氨基酸：{peptide[pos:pos + 3]}..."
                )
        return total_mass + self.WATER_MW

    def add_mw_to_df(
        self, df: pd.DataFrame, peptide_col: str = "肽段序列", mw_col: str = "肽段分子量"
    ):
        """给DataFrame添加分子量列"""
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

    "------分子量计算------"

    def parse_modified_sequences(self):
        """
        处理DataFrame中的修饰序列列，拆分出纯序列、修饰类型和修饰位点

        参数:
            df: 包含序列的DataFrame
            seq_col: 序列所在的列名
            mode2mass: 修饰模式到质量的映射字典

        返回:
            添加了三列的DataFrame: pure_sequence, mod_types, mod_positions
        """

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
        psm_df["row_idx"] = range(len(psm_df))  # 该函数中，peak_start_idx和df的行索引匹配
        charged_frag_types = ["b_z1", "y_z1", "b_z2", "y_z2"]
        fragment_mz_df = create_fragment_mz_dataframe(
            psm_df, charged_frag_types
        )  # 已核对：随便计算无误
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
        """过滤结果：
        1. 删除score≈0的行（score < 0.00001）
        2. 质量匹配过滤（mass_tol/ppm）
        3. 相同spec_idx保留score最高的行
        4. 删除冗余列
        """
        self.filtered_sequences_df = self.sequences_df.copy(deep=True)
        self.filtered_sequences_df = self.filtered_sequences_df[
            self.filtered_sequences_df["score"] >= 0.00001
        ]
        "mass filter去除，dp_decoder已做"
        # if not self.filtered_sequences_df.empty:
        #     self.filtered_sequences_df = filter_mass_match(
        #         self.filtered_sequences_df,
        #         mass_tol=self.mass_tol,
        #         mass_error=self.ppm,
        #         sequence_mass_col="modified_sequence mass",
        #         precursor_mass_col="precursor_mass"
        #     )
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
