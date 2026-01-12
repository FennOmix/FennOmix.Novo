import re

import numba as nb
import numpy as np
import pandas as pd
from alphabase.io.hdf import HDF_File
from alphabase.peptide.fragment import create_fragment_mz_dataframe
from alphabase.peptide.precursor import refine_precursor_df
from peptdeep.mass_spec.match import match_one_raw_with_numba


# 1. 定义Numba加速的核心评分函数（将最耗时的循环用Numba编译）
@nb.jit(nopython=True, parallel=True)
def calculate_score_numba(
    matched_intensities,  # 全量碎片匹配强度 (total_frags, n_frag_types)
    matched_mz_err_ppm,  # 全量碎片匹配误差 (total_frags, n_frag_types)
    frag_start_idxes,  # 每个序列的碎片起始索引 (n_sequences,)
    frag_stop_idxes,  # 每个序列的碎片结束索引 (n_sequences,)
    ppm,
    gamma,
):
    n_sequences = len(frag_start_idxes)
    scores = np.zeros(n_sequences, dtype=np.float64)
    matched_counts = np.zeros(n_sequences, dtype=np.int32)
    matched_ratios = np.zeros(n_sequences, dtype=np.float64)

    for i in nb.prange(n_sequences):
        start = frag_start_idxes[i]
        stop = frag_stop_idxes[i]
        seq_intensities = matched_intensities[start:stop]
        seq_errs = matched_mz_err_ppm[start:stop]
        n_frag = stop - start

        if n_frag == 0:
            scores[i] = 0.0
            matched_counts[i] = 0
            matched_ratios[i] = 0.0
            continue

        # 关键修正：将二维掩码转为一维索引（避免Numba类型推断失败）
        valid_indices = []
        for row in range(seq_intensities.shape[0]):
            for col in range(seq_intensities.shape[1]):
                if seq_intensities[row, col] > 0 and np.abs(seq_errs[row, col]) <= ppm:
                    valid_indices.append((row, col))

        total_score = 0.0
        matched_rows = set()  # 用集合记录已匹配的行（去重）

        if valid_indices:
            # 遍历有效索引计算分数（Numba对列表遍历支持更稳定）
            for row, col in valid_indices:
                err = seq_errs[row, col]
                intensity = seq_intensities[row, col]

                err_ratio = np.abs(err) / ppm
                penalty = 1 - (err_ratio**4)
                total_score += np.log(intensity) * penalty
                matched_rows.add(row)  # 标记该行为已匹配

        matched_count = len(matched_rows)
        matched_ratio = matched_count / n_frag if n_frag > 0 else 0.0
        total_score *= matched_ratio**gamma

        scores[i] = total_score
        matched_counts[i] = matched_count
        matched_ratios[i] = matched_ratio

    return scores, matched_counts, matched_ratios


# # -------------------------- Numba 加速核心函数 --------------------------
# @njit(parallel=True, fastmath=True)  # parallel=True 开启多线程，fastmath=True 加速浮点计算
# def calculate_mass_match_numba(precursor_masses, sequence_masses, mass_tol, mass_error):
#     """
#     Numba 加速的质量匹配计算
#     :param precursor_masses: 母离子质量数组 (n_samples,)
#     :param sequence_masses: 序列质量数组 (n_samples,)
#     :param mass_tol: Da偏移容忍度（int）
#     :param mass_error: ppm误差容忍度（float）
#     :return: 匹配结果数组 (n_samples,)，True=匹配，False=不匹配
#     """
#     n_samples = len(precursor_masses)
#     offsets = np.arange(-mass_tol, mass_tol + 1, dtype=np.float64)  # 偏移量数组
#     n_offsets = len(offsets)
#     match_results = np.zeros(n_samples, dtype=np.bool_)
#
#     # 并行遍历每个样本（prange 是 Numba 并行循环）
#     for i in prange(n_samples):
#         prec_mass = precursor_masses[i]
#         seq_mass = sequence_masses[i]
#
#         # 遍历所有偏移量，计算ppm误差
#         for j in range(n_offsets):
#             shifted_prec = prec_mass + offsets[j]
#             ppm = np.abs(shifted_prec - seq_mass) / seq_mass * 1e6
#             if ppm <= mass_error:
#                 match_results[i] = True
#                 break  # 找到匹配就跳出，减少计算
#
#     return match_results
#
#
# # -------------------------- 主函数（集成Numba加速） --------------------------
# def filter_mass_match(df, mass_tol, mass_error, sequence_mass_col="modified_sequence mass",
#                       precursor_mass_col="precursor_mass"):
#     """
#     根据mass_tol（Da偏移）和mass_error（ppm误差）过滤DataFrame，保留满足质量匹配的行
#     （Numba 加速版，比纯Numpy快5~20倍，数据量越大效果越明显）
#
#     参数说明：
#         df (pd.DataFrame): 输入的原始DataFrame，需包含序列质量和母离子质量列
#         mass_tol (int): Da水平的偏移容忍度（整数），会生成 [-mass_tol, ..., 0, ..., +mass_tol] 的偏移量
#         mass_error (float): ppm水平的误差容忍度（浮点数），允许的最大ppm误差
#         sequence_mass_col (str): 序列质量列的列名（默认："modified_sequence mass"）
#         precursor_mass_col (str): 母离子质量列的列名（默认："precursor_mass"）
#
#     返回值：
#         pd.DataFrame: 仅保留质量匹配（mass_match=True）的行，包含原始列和新增的mass_match列
#     """
#     df_copy = df.copy(deep=True)
#
#     # 1. 校验必要列和数据类型（保留原逻辑）
#     required_cols = [sequence_mass_col, precursor_mass_col]
#     missing_cols = [col for col in required_cols if col not in df_copy.columns]
#     if missing_cols:
#         raise ValueError(f"DataFrame缺少必要列：{', '.join(missing_cols)}")
#
#     for col in required_cols:
#         if not pd.api.types.is_numeric_dtype(df_copy[col]):
#             raise TypeError(f"{col}列必须是数值类型（int/float），当前类型：{df_copy[col].dtype}")
#
#     # 2. 提取数值数组（转为float64，适配Numba）
#     precursor_masses = df_copy[precursor_mass_col].values.astype(np.float64)
#     sequence_masses = df_copy[sequence_mass_col].values.astype(np.float64)
#
#     # 3. 调用Numba加速函数计算匹配结果
#     df_copy["mass_match"] = calculate_mass_match_numba(precursor_masses, sequence_masses, mass_tol, mass_error)
#
#     # 4. 过滤匹配行并重置索引
#     result_df = df_copy[df_copy["mass_match"]].reset_index(drop=True)
#
#     return result_df
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
    # 深拷贝避免修改原始数据
    df_copy = df.copy(deep=True)

    # 校验必要列是否存在
    required_cols = [sequence_mass_col, precursor_mass_col]
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        raise ValueError(f"DataFrame缺少必要列：{', '.join(missing_cols)}")

    # 校验质量列是否为数值类型
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            raise TypeError(f"{col}列必须是数值类型（int/float），当前类型：{df_copy[col].dtype}")

    # 生成偏移量（-mass_tol 到 +mass_tol 的整数，含两端）
    offsets = np.arange(-mass_tol, mass_tol + 1)

    # 向量化计算（高效处理大数据量）
    # 生成 (行数 × 偏移量个数) 的母离子质量偏移矩阵
    precursor_masses = df_copy[precursor_mass_col].values[:, None]  # 转换为列向量
    shifted_masses = precursor_masses + offsets

    # 计算ppm误差矩阵：|偏移后质量 - 序列质量| / 序列质量 × 1e6
    sequence_masses = df_copy[sequence_mass_col].values[:, None]  # 转换为列向量
    ppm_errors = np.abs(shifted_masses - sequence_masses) / sequence_masses * 1e6

    # 标记每行是否有任意一个偏移满足ppm误差要求
    df_copy["mass_match"] = (ppm_errors <= mass_error).any(axis=1)

    # 保留匹配成功的行，重置索引
    result_df = df_copy[df_copy["mass_match"]].reset_index(drop=True)

    return result_df


class DenovoSequenceScoring:
    def __init__(self, sequences_df, hdf_path):
        f = HDF_File(file_name=hdf_path, read_only=True)
        try:
            self.spectra_df = f.ms_data.spectrum_df.values[
                [
                    "charge",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "spec_idx",
                    "ms_level",
                ]
            ]  # 此处为ms.data 可能也会是raw.data
        except:
            self.spectra_df = f.psm.psm_df.values[
                ["charge", "peak_start_idx", "peak_stop_idx", "precursor_mz", "spec_idx"]
            ]
        self.peak_df = f.ms_data.peak_df.values
        self.mode2mass = {
            "Carbamidomethyl@C": "C+57.021",
            "Oxidation@M": "M+15.995",
            "Deamidated@N": "N+0.984",
            "Deamidated@Q": "Q+0.984",
            "Acetyl@Protein_N-term": "+42.011",
            "AEBS@Y": "Y+183.035",
            "AEBS@K": "K+183.035",
            "Glu->pyro-Glu@E^Any_N-term": "-18.011",  # before nterm E
            "Gln->pyro-Glu@Q^Any_N-term": "-17.027",  # before nterm Q
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
            "N+0.984": 115.026943,
            "Q+0.984": 129.042594,
            "+42.011": 42.010565,
            "Y+183.035": 346.099,
            "K+183.035": 311.130,
            "-18.011": -18.011,
            "-17.027": -17.027,
            "C+119.004": 222.014,
        }
        self.WATER_MW = 18.01056  # 加水分子量
        # 预处理：按氨基酸缩写长度降序排序（关键：优先匹配长缩写）
        self.sorted_aas = sorted(self.aa2mass.keys(), key=len, reverse=True)
        self.max_aa_len = max(len(aa) for aa in self.aa2mass.keys())  # 最长氨基酸缩写长度
        "-------------肽段分子量计算------------"
        self.mass2mod = {}
        for mod, mass_str in self.mode2mass.items():
            for mass in re.findall(r"[+-]\d+\.\d+", mass_str):
                self.mass2mod[mass] = mod
        self.top_k = sequences_df.shape[1] - 1  # -1: spec_idx
        self.sequences_df = pd.melt(
            sequences_df,
            id_vars=["spec_idx"],
            value_vars=sequences_df.columns.tolist()[1:],  # 取seq1到seq10 [1:] > 1
            var_name="sequence_rank",
            value_name="modified_sequence",
        )
        self.sequences_df["top"] = (
            self.sequences_df["sequence_rank"].str.extract(r"(\d+)").astype(int)
        )  # 用于记录模型原始输出排名
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
                # 截取子串尝试匹配氨基酸
                aa_candidate = peptide[pos : pos + check_len]
                if aa_candidate in self.aa2mass:
                    # 匹配成功，累加分子量
                    total_mass += self.aa2mass[aa_candidate]
                    pos += check_len  # 跳过已匹配的字符
                    matched = True
                    break

            if not matched:
                # 无匹配的氨基酸（可根据需求修改：抛错/忽略/设为0）
                raise ValueError(
                    f"肽段 '{peptide}' 中存在未定义的氨基酸：{peptide[pos:pos + 3]}..."
                )

        # 加水电分子量，返回结果
        return total_mass + self.WATER_MW

    def add_mw_to_df(
        self, df: pd.DataFrame, peptide_col: str = "肽段序列", mw_col: str = "肽段分子量"
    ):
        """给DataFrame添加分子量列（核心调用方法）"""
        # 批量计算（pandas.apply已优化，百万行足够快）
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

        # 解析单个序列的内部函数
        def parse_seq(seq):
            pure = []
            mods = []
            positions = []
            i = 0
            while i < len(seq):
                # 查找质量修饰(如+15.995)
                match = re.match(r"^([+-]\d+\.\d+)", seq[i:])
                if match:
                    mass = match.group(1)
                    # 记录修饰信息(位置从1开始)
                    mods.append(self.mass2mod.get(mass, f"Unknown{mass}"))
                    positions.append(str(len(pure)))  # 当前纯序列长度即修饰位置
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
        self.sequences_df = pd.merge(
            self.sequences_df,
            self.spectra_df[
                [
                    "charge",
                    "peak_start_idx",
                    "peak_stop_idx",
                    "precursor_mz",
                    "spec_idx",
                    "ms_level",
                ]
            ],
            on="spec_idx",
            how="inner",
        )  # 已核对：spec_idx与原始数据peak_start_stop一致
        psm_df = refine_precursor_df(self.sequences_df)
        psm_df["row_idx"] = range(len(psm_df))  # 该函数中，peak_start_idx和df的行索引匹配
        charged_frag_types = ["b_z1", "y_z1", "b_z2", "y_z2"]
        fragment_mz_df = create_fragment_mz_dataframe(
            psm_df, charged_frag_types
        )  # 已核对：随便计算无误
        self.fragment_mz_df = fragment_mz_df
        # ========== 关键修改: 显式类型转换 ==========
        # 确保所有数组都是正确的数值类型
        psm_df["spec_idx"] = psm_df["spec_idx"].astype(np.int64)
        psm_df["frag_start_idx"] = psm_df["frag_start_idx"].astype(np.int64)
        psm_df["frag_stop_idx"] = psm_df["frag_stop_idx"].astype(np.int64)
        psm_df = self.add_mw_to_df(
            psm_df, peptide_col="modified_sequence", mw_col="modified_sequence mass"
        )
        psm_df = self.calculate_precursor_mass(
            psm_df, precursor_mz_col="precursor_mz", charge_col="charge"
        )
        # 提取并转换谱图数据数组
        all_spec_mzs = self.peak_df["mz"].values.astype(np.float64)
        all_spec_intensities = self.peak_df["intensity"].values.astype(np.float64)

        # 提取并转换索引数组
        peak_start_idxes = self.sequences_df["peak_start_idx"].values.astype(np.int64)
        peak_stop_idxes = self.sequences_df["peak_stop_idx"].values.astype(np.int64)

        # 确保碎片 m/z 数组是浮点类型
        all_frag_mzs = fragment_mz_df.values.astype(np.float64)

        # 初始化结果数组
        matched_intensities = np.zeros_like(all_frag_mzs, dtype=np.float64)
        matched_mz_errs = np.full_like(all_frag_mzs, np.inf, dtype=np.float64)

        # 执行匹配
        match_one_raw_with_numba(
            spec_idxes=psm_df["row_idx"].values,  # √
            frag_start_idxes=psm_df["frag_start_idx"].values,  # √
            frag_stop_idxes=psm_df["frag_stop_idx"].values,  # √
            all_frag_mzs=all_frag_mzs,  # √
            all_spec_mzs=all_spec_mzs,  # √
            all_spec_intensities=all_spec_intensities,  # √
            peak_start_idxes=peak_start_idxes,  # √
            peak_end_idxes=peak_stop_idxes,  # √
            matched_intensities=matched_intensities,  # √
            matched_mz_errs=matched_mz_errs,  # √
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

        if not self.filtered_sequences_df.empty:
            self.filtered_sequences_df = filter_mass_match(
                self.filtered_sequences_df,
                mass_tol=self.mass_tol,
                mass_error=self.ppm,
                sequence_mass_col="modified_sequence mass",
                precursor_mass_col="precursor_mass",
            )
        if not self.filtered_sequences_df.empty:
            self.filtered_sequences_df = (
                self.filtered_sequences_df.sort_values("score", ascending=False)
                .drop_duplicates("spec_idx", keep="first")
                .reset_index(drop=True)  # 重置索引，避免原索引混乱
            )
        drop_cols = [
            "sequence_rank",
            "top",
            "peak_start_idx",
            "peak_stop_idx",
            "ms_level",
            "row_idx",
            "frag_start_idx",
            "frag_stop_idx",
        ]
        existing_cols = [col for col in drop_cols if col in self.filtered_sequences_df.columns]
        if existing_cols:
            self.filtered_sequences_df.drop(columns=existing_cols, inplace=True)

    def pGlyco_scoring_numba(self, gamma=0.94):
        # 1. 保留原始精度（用float64，避免精度丢失）
        ppm = self.ppm
        matched_intensities = self.matched_intensity_df.values.astype(np.float64)
        matched_mz_err_ppm = self.matched_mz_err_ppm_df.values.astype(np.float64)

        # 2. 提取每个序列的碎片分段索引（关键修正：传递正确的分段信息）
        frag_start_idxes = self.sequences_df["frag_start_idx"].values.astype(np.int64)
        frag_stop_idxes = self.sequences_df["frag_stop_idx"].values.astype(np.int64)

        # 3. 调用修正后的Numba函数
        scores, matched_counts, matched_ratios = calculate_score_numba(
            matched_intensities, matched_mz_err_ppm, frag_start_idxes, frag_stop_idxes, ppm, gamma
        )

        # 4. 批量赋值结果
        self.sequences_df["score"] = scores.round(2).astype(np.float64)
        self.sequences_df["matched_ion_count"] = matched_counts
        self.sequences_df["matched_ion_ratio"] = matched_ratios.round(2).astype(np.float64)


def score_sequence(sequences_df, hdf_path):
    scoring_module = DenovoSequenceScoring(sequences_df, hdf_path)
    scoring_module.sequence_spectra_match()
    scoring_module.pGlyco_scoring_numba()
    scoring_module.filter_score_and_mass()
    return scoring_module.sequences_df, scoring_module.filtered_sequences_df
