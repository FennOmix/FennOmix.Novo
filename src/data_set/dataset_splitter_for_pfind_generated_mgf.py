#读取所有hdf5文件，按unique seq划分训练集与验证集

#todo： 在最后期，可以不划分，全部数据用来训练
#但需要注意：1. 本代码中的修饰、标准氨基酸控制需假如mgf+search_result >  hdf的脚本中
mode2mass = {
    "Carbamidomethyl@C": "C+57.021",
    "Oxidation@M": "M+15.995",
    "Deamidated@NQ": "N+0.984 Q+0.984",
    "Acetyl@Protein N-term": "+42.011",
    "AEBS@Y": "Y+183.035",
    "AEBS@K": "K+183.035",
    "Glu->pyro-Glu@E^Any_N-term": "-18.011", #before nterm E
    "Gln->pyro-Glu@Q^Any_N-term": "-17.027", #before nterm Q
    "Cysteinyl@C": "C+119.004"
}
standard_amino_acids = {'H','R','K','I','F','L','W','A','M','P','C','N','V','G','S','Q','Y','D','E','T'}
import os
import glob
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from alphabase.io.hdf import HDF_File
from tqdm import tqdm
from collections import defaultdict


def build_mod_seq(sequences, mods_list, sites_list, mod_dict):
    """高性能构建带修饰的肽段序列"""
    result = []
    mass_dict = {k: '+' + v.split('+')[-1] for k, v in mod_dict.items() if '+' in v}
    for k,v in mod_dict.items():
        if '-' in v and v[0] == '-':
            mass_dict[k] = v


    for seq, mods, sites in zip(sequences, mods_list, sites_list):
        if not mods or pd.isna(mods) or mods.strip() == '':
            result.append(seq)
            continue
        mod_list_in_pep = mods.split(';')
        for i in mod_list_in_pep:
            if i not in mode2mass:
               break
        mods_split = mods.split(';')
        sites_split = list(map(int, sites.split(';')))

        # 预构造：每个位置可能要插入的 mass
        mod_map = {}  # site: mass_tag
        for mod, site in zip(mods_split, sites_split):
            tag = mass_dict.get(mod, '')
            if tag:
                mod_map[site - 1] = tag  # 关键修正

        # 构造修饰序列（一次性拼接）
        parts = []
        for i, aa in enumerate(seq):
            parts.append(aa)
            if i in mod_map:
                parts.append(mod_map[i])
        if -1 in mod_map:
            parts.insert(0,mod_map[-1])
        result.append(''.join(parts))

    return result

import pandas as pd
import numpy as np
from collections import Counter
def filter_and_remap_peaks(spectrum_df, peak_df):
    # 确保索引是整数
    start_indices = spectrum_df['peak_start_idx'].astype(int).values
    stop_indices = spectrum_df['peak_stop_idx'].astype(int).values

    # 存放新峰列表
    new_peaks = []
    new_start_idx = []
    new_stop_idx = []

    current_idx = 0  # 新 peak_df 的起始位置计数器

    for s, e in zip(start_indices, stop_indices):
        peaks = peak_df.iloc[s:e+1].copy()
        new_peaks.append(peaks)

        # 记录新的 start/stop idx
        n = len(peaks)
        new_start_idx.append(current_idx)
        new_stop_idx.append(current_idx + n - 1)
        current_idx += n

    # 拼接新 peak_df
    new_peak_df = pd.concat(new_peaks, ignore_index=True)

    # 更新 spectrum_df 中索引
    spectrum_df = spectrum_df.copy()
    spectrum_df['peak_start_idx'] = new_start_idx
    spectrum_df['peak_stop_idx'] = new_stop_idx

    return spectrum_df, new_peak_df


import matplotlib.pyplot as plt


def count_frequency(sequence_info, output_path):
    counts = [len(value) for value in sequence_info.values()]
    count_freq = Counter(counts)

    # 转换为 DataFrame
    df = pd.DataFrame({
        'Count of Elements in List': list(count_freq.keys()),
        'Frequency': list(count_freq.values())
    })

    # 保存为 CSV 文件
    df.to_csv(output_path+'/peptide_psm_frequency.csv', index=False)

    # 绘制条形图
    plt.bar(df['Count of Elements in List'], df['Frequency'], width=0.8, edgecolor='black')
    plt.title('Distribution of List Counts (Statistic Based)')
    plt.xlabel('Count of Elements in List')
    plt.ylabel('Frequency')
    plt.savefig(output_path+'/peptide_psm_frequency.png')
    plt.show()

class HDFDataSplitter:
    """Split HDF5 datasets based on unique sequences."""

    def __init__(
            self,
            hdf5_paths: List[str],
            val_ratio: float = 0.1,
            test_ratio: float = 0.0,
            random_state: Optional[int] = None,
            output_train: str = r'X:\chenzx\from_ssd_1029\chenzx\zheyi_data\training_dataset\batch11-14\train',
            output_val: str = r'X:\chenzx\from_ssd_1029\chenzx\zheyi_data\training_dataset\batch11-14\val',
            output_test: str = 'Z:/chenzx/zheyi_raw_data/MHC/ZheYi-LiverCancer/hdf_with_pfind_search_result_batch14/test_dataset',

    ):
        self.hdf5_paths = hdf5_paths
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = np.random.RandomState(random_state)
        self.standard_amino_acids = standard_amino_acids
        # 预加载所有序列信息
        self.output_train = output_train
        self.output_val = output_val
        self.output_test = output_test
        self.sequence_info = self._preload_sequences()
    def _is_standard_sequence(self, seq):
        """类内方法：检查序列是否全为标准氨基酸"""
        # 处理非字符串类型（如列表、数字）
        if not isinstance(seq, str):
            return False
        # 处理空字符串或纯空格
        seq = seq.strip()
        if len(seq) == 0:
            return False
        # 逐字符检查
        return all(aa in self.standard_amino_acids for aa in seq)

    def _has_valid_mods(self, mods):
        """检查修饰是否全部在mode2mass中"""
        # 空修饰视为有效
        if not mods or pd.isna(mods) or mods.strip() == '':
            return True
        # 拆分多个修饰（以分号分隔）
        mod_list = mods.split(';')
        # 检查每个修饰是否都在mode2mass中
        return all(mod in mode2mass for mod in mod_list)


    def _preload_sequences(self):

        sequence_info = {}
        for file_idx, path in enumerate(self.hdf5_paths):
            f = HDF_File(file_name=path, read_only=True)
            psm_df = f.psm.psm_df.values[['sequence', 'mods', 'mod_sites', 'precursor_mz', 'charge', 'spec_idx']]
            psm_df = psm_df[(psm_df['sequence'].str.len() >= 8) & (psm_df['sequence'].str.len() <= 14)]

            psm_df = psm_df[psm_df['sequence'].apply(self._is_standard_sequence)].copy() #要求氨基酸为标准氨基酸

            psm_df = psm_df[psm_df['mods'].apply(self._has_valid_mods)].copy() #要求修饰为设定修饰

            psm_df['modified_sequence'] = build_mod_seq(
                psm_df['sequence'],
                psm_df['mods'],
                psm_df['mod_sites'],
                mod_dict=mode2mass
            )

            for orig_idx, row in psm_df.iterrows():
                sequence = row.modified_sequence
                if sequence not in sequence_info:
                    sequence_info[sequence] = []
                # 存储 (file_idx, 原始索引)，而非筛选后的 local_idx
                sequence_info[sequence].append((file_idx, orig_idx, row['precursor_mz'], row['charge'], row['spec_idx']))

        # 绘制直方图
        count_frequency(sequence_info,output_path = self.output_train)


        return sequence_info

    def split_seq_group(self):

        unique_sequences = list(self.sequence_info.keys())
        self.rng.shuffle(unique_sequences)
        n_val = int(len(unique_sequences) * self.val_ratio)
        n_test = int(len(unique_sequences) * self.test_ratio)
        # all_seq_keys = list(self.sequence_info.keys()) # 改为 all_seq_keys = unique_sequences
        all_seq_keys = unique_sequences
        if n_test == 0:
            train_seq_keys = all_seq_keys[n_val:]
            val_seq_keys = all_seq_keys[:n_val]
            self.train_sequence_info = {k: self.sequence_info[k] for k in train_seq_keys}
            self.val_sequence_info = {k: self.sequence_info[k] for k in val_seq_keys}
        if n_test > 0:
            train_seq_keys = all_seq_keys[(n_val + n_test):] # [0.2+0.1：]
            val_seq_keys = all_seq_keys[:n_val] #[0:0.2]
            test_seq_keys = all_seq_keys[n_val: n_val + n_test] #[0.2:0.3]
            self.train_sequence_info = {k: self.sequence_info[k] for k in train_seq_keys}
            self.val_sequence_info = {k: self.sequence_info[k] for k in val_seq_keys}
            self.test_sequence_info = {k: self.sequence_info[k] for k in test_seq_keys}
            self.test_file_groups = defaultdict(list)
            for seq, pairs in self.test_sequence_info.items():
                for file_idx, orig_idx, prec_mz, charge, spec_idx in pairs:
                    self.test_file_groups[file_idx].append((orig_idx, seq, prec_mz, charge, spec_idx))
        #unique seq与对应的idx划分为train and val

        del self.sequence_info

        self.train_file_groups = defaultdict(list)
        self.val_file_groups = defaultdict(list)
        for seq, pairs in self.train_sequence_info.items():
            for file_idx, orig_idx, prec_mz, charge, spec_idx in pairs:
                self.train_file_groups[file_idx].append((orig_idx, seq, prec_mz, charge, spec_idx))

        for seq, pairs in self.val_sequence_info.items():
            for file_idx, orig_idx, prec_mz, charge, spec_idx in pairs:
                self.val_file_groups[file_idx].append((orig_idx, seq, prec_mz, charge, spec_idx))



    def build_peaks_column(self,spectrum_df, peak_df):
        # 将 mz 和 intensity 转为 ndarray，加速访问
        peak_array = peak_df[['mz', 'intensity']].values  # shape: (N, 2)

        start_indices = spectrum_df['peak_start_idx'].astype(int).values
        stop_indices = spectrum_df['peak_stop_idx'].astype(int).values

        result = []
        append = result.append  # 加快 for-loop 中函数调用

        for s, e in zip(start_indices, stop_indices):
            peaks = peak_array[s:e + 1]  # e+1 是右开区间
            append(list(map(tuple, peaks)))  # 转为 list[tuple]

        spectrum_df['peaks'] = result
        return  spectrum_df

    def hdf_generator(self, file_groups, output_folder):
        ''' match： raw idx + spec_idx + charge + precursor_mz(0.02 shift)'''
        for key, ms2_list in tqdm(file_groups.items()):  # 逐HDF读取
            current_handle = HDF_File(file_name=self.hdf5_paths[key], read_only=True)

            # 解包所有信息（保留你的原有解包）
            idxs, seqs, prec_mzs, charges, spec_idxs = zip(*file_groups[key])

            # 步骤1：读取原始PSM_df并筛选（保留原有逻辑）
            seq_psm_df = current_handle.psm.psm_df.values.iloc[list(idxs)].copy()

            # 步骤2：读取spectrum_df并预处理（核心修改1：放弃索引匹配，改为数值匹配）
            spectrum_df = current_handle.ms_data.spectrum_df.values.copy()
            # 确保spec_idx、charge、precursor_mz为有效数值
            spectrum_df = spectrum_df[
                (spectrum_df['spec_idx'].notna()) &
                (spectrum_df['charge'].notna()) &
                (spectrum_df['precursor_mz'].notna())
                ].copy()
            spectrum_df['spec_idx'] = spectrum_df['spec_idx'].astype(int)
            spectrum_df['charge'] = spectrum_df['charge'].astype(int)

            # 步骤3：为每个PSM匹配符合条件的光谱行（spec_idx+charge+mz协同匹配）
            valid_psm_mask = []  # 标记PSM是否找到匹配的光谱
            matched_spec_rows = []  # 存储匹配到的光谱行
            for idx in range(len(seq_psm_df)):
                psm_spec_idx = spec_idxs[idx]
                psm_charge = charges[idx]
                psm_prec_mz = prec_mzs[idx]

                # 第一步：筛选同spec_idx的光谱行
                same_spec_df = spectrum_df[spectrum_df['spec_idx'] == psm_spec_idx].copy()
                if same_spec_df.empty:
                    valid_psm_mask.append(False)
                    continue

                # 第二步：筛选charge一致的光谱行
                same_charge_df = same_spec_df[same_spec_df['charge'] == psm_charge].copy()
                if same_charge_df.empty:
                    valid_psm_mask.append(False)
                    continue

                # 第三步：计算mz差值，筛选≤0.02的行
                same_charge_df['mz_diff'] = abs(same_charge_df['precursor_mz'] - psm_prec_mz)
                valid_mz_df = same_charge_df[same_charge_df['mz_diff'] <= 0.02].copy()

                # 第四步：匹配逻辑（优先选mz差值最小的行，避免重复spec_idx多匹配）
                if valid_mz_df.empty:
                    valid_psm_mask.append(False)
                else:
                    # 取mz差值最小的那一行（解决spec_idx重复的核心）
                    best_match = valid_mz_df.loc[valid_mz_df['mz_diff'].idxmin()]
                    matched_spec_rows.append(best_match)
                    valid_psm_mask.append(True)

            # 步骤4：过滤无匹配的PSM
            if not any(valid_psm_mask):
                print(f"文件 {self.hdf5_paths[key]} 无有效数据（spec_idx+charge+mz协同匹配失败），跳过")
                continue

            # 应用PSM筛选
            seq_psm_df = seq_psm_df[valid_psm_mask].reset_index(drop=True)
            # 构建匹配后的spectrum_df
            spectrum_df = pd.DataFrame(matched_spec_rows).reset_index(drop=True)
            # 同步过滤其他列表
            prec_mzs = [mz for mz, valid in zip(prec_mzs, valid_psm_mask) if valid]
            charges = [c for c, valid in zip(charges, valid_psm_mask) if valid]
            spec_idxs = [idx for idx, valid in zip(spec_idxs, valid_psm_mask) if valid]
            seqs = [seq for seq, valid in zip(seqs, valid_psm_mask) if valid]

            # 步骤5：提取peak_df并重新映射（保留你的原有逻辑）
            peak_df = current_handle.ms_data.peak_df.values
            new_spectrum_df, new_peak_df = filter_and_remap_peaks(spectrum_df, peak_df)

            # 步骤6：合并peak索引到PSM_df（保留你的原有逻辑）
            peak_map = new_spectrum_df[['spec_idx', 'peak_start_idx', 'peak_stop_idx']]
            seq_psm_df = seq_psm_df.drop(columns=['peak_start_idx', 'peak_stop_idx'], errors='ignore')
            seq_psm_df = seq_psm_df.merge(peak_map, on='spec_idx', how='left')

            # 步骤7：验证修饰序列（保留你的原有逻辑，修复变量名错误）
            modified_pep = build_mod_seq(
                seq_psm_df['sequence'],
                seq_psm_df['mods'],
                seq_psm_df['mod_sites'],
                mod_dict=mode2mass
            )
            for pep in modified_pep:  # 修复原代码seqs变量名重复问题
                if pep not in seqs:
                    print(pep)

            # 步骤8：生成新HDF文件（保留你的原有逻辑，优化路径拼接）
            import os
            file_basename = os.path.basename(self.hdf5_paths[key])
            output_hdf_path = os.path.join(output_folder, file_basename)
            output_hdf = HDF_File(
                file_name=output_hdf_path,
                read_only=False,
                truncate=True,
                delete_existing=True
            )
            output_hdf.psm = {}
            output_hdf.ms_data = {}
            output_hdf.psm.psm_df = seq_psm_df
            output_hdf.ms_data.peak_df = new_peak_df


def test_data_loading_performance():
    """Test HDF data loading performance after optimization."""
    import time
    import torch
    from pathlib import Path

    # 1. 配置参数
    config = {
        "data_dir": r"X:\chenzx\from_ssd_1029\chenzx\zheyi_data\training_dataset\batch11-14\all",
        "batch_size": 32,
        "val_batch_size":1024,
        "num_workers": 0,  # Windows下使用0
        "cache_size": 4,
        "val_ratio": 0.1,
        "random_state": 42,
        "n_peaks": 150,
        "min_mz": 140.0,
        "max_mz": 2500.0,
        "min_intensity": 0.01,
        "remove_precursor_tol": 2.0
    }

    print("Starting data loading performance test...")

    hdf_files = list(Path(config["data_dir"]).glob("*.hdf5"))
    data_splitter = HDFDataSplitter(
        hdf5_paths = hdf_files
    )
    data_splitter.split_seq_group()
    print('train_hdf5_generator...')
    data_splitter.hdf_generator(data_splitter.train_file_groups,data_splitter.output_train)
    print('val_hdf5_generator...')
    data_splitter.hdf_generator(data_splitter.val_file_groups,data_splitter.output_val)
    if data_splitter.test_ratio > 0:
        print('test_hdf5_generator...')
        data_splitter.hdf_generator(data_splitter.test_file_groups,data_splitter.output_test)

#批量读取 hdf > 收集unique seq划分训练集与测试集> 存入训练集与验证集文件夹(hdf文件名不变)
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    test_data_loading_performance()