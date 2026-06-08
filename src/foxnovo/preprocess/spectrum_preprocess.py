import numpy as np

from foxnovo.constants import ISOTOPE_MASS_DIFF, PROTON_MASS


# 模拟AtomLib类（C#中的AtomLib.MassProton和MassIsotope）
class AtomLib:
    MassProton = PROTON_MASS  # 质子质量
    MassIsotope = ISOTOPE_MASS_DIFF  # 同位素质量差（C13-C12）


class Preprocess:
    MinMassErr: float = 0.005
    MaxPeakCharge: int = 3

    MonoLeftRelInten: float = 0.1
    FirstIsotopeLevelMz: float = 1000.0
    FirstIsotopeFactor: float = 0.7
    SecondIsotopeLevelMz: float = 2000.0
    SecondIsotopeFactor: float = 0.33

    @staticmethod
    def keep_local_top_k_peaks(
        peak_mzs: np.ndarray,
        peak_intens: np.ndarray,
        peaks_per_block: int,
        block_top_k: int,
        max_block_mz_range: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        按块保留局部Top-K峰（保持m/z升序）
        """
        n = len(peak_mzs)
        if peaks_per_block <= 0 or block_top_k <= 0:
            return peak_mzs.copy(), peak_intens.copy()

        constrain_mz_range = max_block_mz_range > 0.0
        selected = np.zeros(n, dtype=bool)

        i = 0
        while i < n:
            block_start = i
            j = block_start + 1

            # 扩展块边界（满足峰数量和m/z范围约束）
            while j < n:
                if j - block_start >= peaks_per_block:
                    break
                if constrain_mz_range and (
                    peak_mzs[j] - peak_mzs[block_start] > max_block_mz_range
                ):
                    break
                j += 1

            count = j - block_start
            if count <= 0:
                i += 1
                continue

            if count <= block_top_k:
                # 保留块内所有峰
                selected[block_start:j] = True
            else:
                # 保留块内强度Top-K的峰
                local_indices = np.arange(block_start, j)
                # 按强度降序排序
                local_indices = local_indices[np.argsort(peak_intens[local_indices])[::-1]]
                selected[local_indices[:block_top_k]] = True

            i = j

        # 收集选中的峰（保持原顺序）
        mask = selected
        out_mzs = peak_mzs[mask]
        out_intens = peak_intens[mask]

        return out_mzs, out_intens

    @staticmethod
    def keep_top_k_peaks(
        peak_mzs: np.ndarray, peak_intens: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        保留强度Top-K的峰（返回时按m/z升序排列）
        """
        n = len(peak_mzs)
        if top_k <= 0 or n <= top_k:
            return peak_mzs.copy(), peak_intens.copy()

        # 按强度降序排序的索引
        indices = np.argsort(peak_intens)[::-1]
        # 取Top-K索引
        top_indices = indices[:top_k]
        # 按m/z升序重新排序
        top_indices = top_indices[np.argsort(peak_mzs[top_indices])]

        top_mzs = peak_mzs[top_indices]
        top_ints = peak_intens[top_indices]

        return top_mzs, top_ints

    @staticmethod
    def deisotope_and_decharge(
        peak_mzs: np.ndarray, peak_intens: np.ndarray, rel_mass_err: float, prec_z: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        去同位素和去电荷处理
        """
        # 限制最大电荷数
        prec_z = min(prec_z, Preprocess.MaxPeakCharge)
        # 移除质子质量以便处理
        peak_masses = peak_mzs - AtomLib.MassProton

        # 获取峰电荷数和同位素索引
        peak_charges, isotope_idxes = Preprocess._get_peak_charges_and_isotopes(
            peak_masses, peak_intens, rel_mass_err, prec_z
        )

        # 移除同位素峰和多电荷峰
        ret_masses, ret_intens = Preprocess._remove_isotopes_and_charges(
            peak_masses, peak_intens, peak_charges, isotope_idxes
        )

        # 合并相邻峰
        merged_masses, merged_intens = Preprocess._merge_neighbor_peaks(
            ret_masses, ret_intens, rel_mass_err
        )

        # 加回质子质量
        merged_masses = merged_masses + AtomLib.MassProton

        return merged_masses, merged_intens

    @staticmethod
    def _merge_neighbor_peaks(
        peak_masses: np.ndarray, peak_intens: np.ndarray, rel_mass_err: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        合并质量误差范围内的相邻峰
        """
        if len(peak_masses) == 0:
            return np.array([]), np.array([])

        ret_masses: list[float] = []
        ret_intens: list[float] = []

        merged_peaks = [0]

        for i in range(1, len(peak_masses)):
            # 计算质量误差范围
            first_idx = merged_peaks[0]
            mass_err_left = rel_mass_err * peak_masses[first_idx]
            mass_err_right = rel_mass_err * peak_masses[i]
            mass_err_right = max(mass_err_right, Preprocess.MinMassErr)

            # 检查是否超出合并范围
            if peak_masses[i] - peak_masses[first_idx] > mass_err_left + mass_err_right:
                # 合并当前组并开始新组
                Preprocess._append_merged_peaks(
                    peak_masses, peak_intens, merged_peaks, ret_masses, ret_intens
                )
                merged_peaks.clear()
            merged_peaks.append(i)

        # 合并最后一组
        Preprocess._append_merged_peaks(
            peak_masses, peak_intens, merged_peaks, ret_masses, ret_intens
        )

        return np.array(ret_masses), np.array(ret_intens)

    @staticmethod
    def _get_peak_charges_and_isotopes(
        peak_masses: np.ndarray, peak_intens: np.ndarray, rel_mass_err: float, max_peak_charge: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        获取每个峰的电荷数和同位素索引
        """
        n = len(peak_masses)
        peak_charges = np.zeros(n, dtype=int)  # 0: 未处理, >0: 电荷数, <0: 同位素峰（编码母峰索引）
        # isotope_idxes: [峰索引, 电荷数-1] -> 下一个同位素峰索引
        isotope_idxes = np.zeros((n, max_peak_charge), dtype=int)

        for i in range(n):
            if peak_charges[i] != 0:
                continue  # 已处理（同位素峰）

            for charge in range(1, max_peak_charge + 1):
                iso_mass = AtomLib.MassIsotope / charge
                prev_peak = i

                for j in range(i + 1, n):
                    # 计算质量误差
                    mass_err = rel_mass_err * peak_masses[j]
                    mass_err = max(mass_err, Preprocess.MinMassErr)
                    # 计算质量差与理论同位素质量差的偏差
                    diff = peak_masses[j] - peak_masses[prev_peak] - iso_mass

                    if abs(diff) <= mass_err:
                        # 检查同位素强度合理性
                        is_mono = prev_peak == i
                        if Preprocess._check_possible_isotope(
                            peak_masses[prev_peak] * charge,
                            peak_intens[prev_peak],
                            peak_intens[j],
                            is_mono,
                        ):
                            if is_mono:
                                peak_charges[i] = charge  # 标记母峰电荷数

                            # 编码同位素峰的母峰索引（Python负数索引兼容）
                            peak_charges[j] = i - n
                            isotope_idxes[prev_peak, charge - 1] = j
                            prev_peak = j
                    elif diff > mass_err:
                        break  # m/z递增，后续峰偏差更大，提前退出

        return peak_charges, isotope_idxes

    @staticmethod
    def _remove_isotopes_and_charges(
        peak_masses: np.ndarray,
        peak_intens: np.ndarray,
        peak_charges: np.ndarray,
        isotope_idxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        移除同位素峰并处理多电荷峰（转换为单电荷质量）
        """
        kept_masses: list[float] = []
        kept_intens: list[float] = []
        intens = peak_intens.copy()

        # 倒序处理：先处理同位素峰，将强度累加到母峰
        for i in range(len(peak_masses) - 1, -1, -1):
            if peak_charges[i] < 0:
                # 解码母峰索引（peak_charges[i] = 母峰索引 - n）
                mono_idx = peak_charges[i] + len(peak_masses)
                if 0 <= mono_idx < len(peak_masses):
                    intens[mono_idx] += intens[i]
            elif peak_charges[i] == 0:
                # 未确定电荷的峰，尝试所有可能电荷
                max_charge = isotope_idxes.shape[1]
                for ch in range(1, max_charge + 1):
                    kept_masses.append(peak_masses[i] * ch)
                    kept_intens.append(intens[i])
            else:
                # 确定电荷的峰，转换为单电荷质量
                charge = peak_charges[i]
                kept_masses.append(peak_masses[i] * charge)
                kept_intens.append(intens[i])

                # 特殊情况：电荷为2时，检查是否可能是电荷1
                if charge == 2:
                    isotope_row = Preprocess._get_isotope_row(isotope_idxes, i)
                    if Preprocess._check_other_possible_charges(i, isotope_row, peak_intens):
                        kept_masses.append(peak_masses[i])  # 保留电荷1的情况
                        kept_intens.append(intens[i])

        # 转换为numpy数组并按质量排序
        kept_masses_arr = np.array(kept_masses)
        kept_intens_arr = np.array(kept_intens)

        if len(kept_masses_arr) > 0:
            sorted_indices = np.argsort(kept_masses_arr)
            sorted_masses = kept_masses_arr[sorted_indices]
            sorted_intens = kept_intens_arr[sorted_indices]
        else:
            sorted_masses = np.array([])
            sorted_intens = np.array([])

        return sorted_masses, sorted_intens

    @staticmethod
    def _check_possible_isotope(
        left_mass: float, left_inten: float, right_inten: float, is_mono: bool = False
    ) -> bool:
        """
        检查右侧峰是否为左侧峰的合理同位素峰
        """
        if is_mono and right_inten <= Preprocess.MonoLeftRelInten * left_inten:
            return False

        if left_mass <= Preprocess.FirstIsotopeLevelMz:
            return left_inten >= right_inten
        elif Preprocess.FirstIsotopeLevelMz < left_mass <= Preprocess.SecondIsotopeLevelMz:
            return left_inten >= Preprocess.FirstIsotopeFactor * right_inten
        else:
            return left_inten >= Preprocess.SecondIsotopeFactor * right_inten

    @staticmethod
    def _get_isotope_row(isotope_idxes: np.ndarray, mono_idx: int) -> np.ndarray:
        """
        获取指定母峰的同位素索引行
        """
        return isotope_idxes[mono_idx, :].copy()

    @staticmethod
    def _check_other_possible_charges(
        mono_idx: int, isotope_idxes: np.ndarray, spec_intens: np.ndarray
    ) -> bool:
        """
        检查母峰是否可能有其他电荷状态（主要针对电荷2是否可能为电荷1）
        """
        # 检查是否有足够的同位素峰且强度符合预期
        return (
            len(isotope_idxes) >= 2
            and isotope_idxes[0] > 0
            and isotope_idxes[1] > 0
            and spec_intens[isotope_idxes[1]] < spec_intens[mono_idx]
            and spec_intens[isotope_idxes[1]] < spec_intens[isotope_idxes[0]]
        )

    @staticmethod
    def _append_merged_peaks(
        peak_masses: np.ndarray,
        peak_intens: np.ndarray,
        merged_peaks: list[int],
        ret_masses: list[float],
        ret_intens: list[float],
    ) -> None:
        """
        合并一组峰并添加到结果中（加权平均质量，总强度）
        """
        indices = np.array(merged_peaks)
        intes = peak_intens[indices]
        sum_int = np.sum(intes)

        if sum_int <= 0.0:
            # 强度为0时，使用简单平均质量
            mean_mass = np.mean(peak_masses[indices])
            ret_masses.append(mean_mass)
            ret_intens.append(0.0)
        else:
            # 加权平均质量（质量*强度求和 / 总强度）
            sum_mass_int = np.sum(peak_masses[indices] * intes)
            weighted_mass = sum_mass_int / sum_int
            ret_masses.append(weighted_mass)
            ret_intens.append(sum_int)


# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_mzs = np.array([100.0, 101.0, 102.0, 200.0, 201.0, 300.0], dtype=np.float32)
    test_intens = np.array([1000.0, 500.0, 200.0, 800.0, 300.0, 1500.0], dtype=np.float32)

    # 1. 测试Top-K峰保留
    top_mzs, top_ints = Preprocess.keep_top_k_peaks(test_mzs, test_intens, 3)
    print("Top 3 Peaks:")
    print(f"MZ: {top_mzs}, Intens: {top_ints}\n")

    # 2. 测试局部Top-K峰保留
    local_mzs, local_ints = Preprocess.keep_local_top_k_peaks(
        test_mzs, test_intens, peaks_per_block=3, block_top_k=2
    )
    print("Local Top 2 per Block (3 peaks/block):")
    print(f"MZ: {local_mzs}, Intens: {local_ints}\n")

    # 3. 测试去同位素和去电荷
    deiso_mzs, deiso_ints = Preprocess.deisotope_and_decharge(
        test_mzs, test_intens, rel_mass_err=0.001, prec_z=2
    )
    print("Deisotoped and Decharged Peaks:")
    print(f"MZ: {deiso_mzs}, Intens: {deiso_ints}")
