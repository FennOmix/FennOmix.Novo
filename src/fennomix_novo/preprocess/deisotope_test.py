import unittest

import numpy as np

# 假设 Preprocess 和 AtomLib 与当前测试文件在同一目录，否则需调整导入路径
from fennomix_novo.preprocess import AtomLib, Preprocess  # 替换为实际的 Preprocess 所在文件名


class TestPreprocess(unittest.TestCase):
    def test_deisotope(self):
        # 构造测试数据（与C#一致）
        peak_mzs = np.array([0.5, 0.66, 1.0, 1.33, 1.5, 50, 100.5, 101.5, 600], dtype=np.float32)
        peak_mzs = peak_mzs + 99.5  # 对应C#的Select(mz => mz + 99.5f)
        peak_intens = np.ones_like(peak_mzs, dtype=np.float32)  # 所有强度为1.0f

        # 1. 测试GetPeakChargesAndIsotopes
        peak_charges, iso_idxes = Preprocess._get_peak_charges_and_isotopes(
            peak_mzs, peak_intens, rel_mass_err=2e-4, max_peak_charge=3
        )

        # 断言peak_charges（与C#预期结果一致）
        expected_charges = np.array([2, 3, -8, -8, -9, 0, 1, -3, 0], dtype=int)
        np.testing.assert_array_equal(peak_charges, expected_charges)

        # 2. 测试RemoveIsotopesAndCharges
        masses, intens = Preprocess._remove_isotopes_and_charges(
            peak_mzs, peak_intens, peak_charges, iso_idxes
        )

        # 断言强度（与C#预期一致）
        expected_intens = np.array([1, 2, 2, 1, 3, 1, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(intens, expected_intens)

        # 断言质量（允许微小浮点误差，与C#预期一致）
        expected_masses = np.array(
            [149.5, 200.0, 200.0, 299, 300.48, 448.5, 699.5, 1399, 2098.5], dtype=np.float32
        )
        np.testing.assert_allclose(masses, expected_masses, rtol=1e-5, atol=1e-5)

        # 3. 测试MergeNeighborPeaks
        merged_masses, merged_intens = Preprocess._merge_neighbor_peaks(
            masses, intens, rel_mass_err=2e-4
        )

        # 断言合并后的质量
        expected_merged_masses = np.array(
            [149.5, 200.0, 299, 300.48, 448.5, 699.5, 1399, 2098.5], dtype=np.float32
        )
        np.testing.assert_allclose(merged_masses, expected_merged_masses, rtol=1e-5, atol=1e-5)

        # 断言合并后的强度
        expected_merged_intens = np.array([1, 4, 1, 3, 1, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(merged_intens, expected_merged_intens)

        # 4. 测试完整的DeisotopeAndDecharge流程
        # 先加回质子质量（模拟C#测试中的peak_mzs = peak_mzs.Select(mz => mz + AtomLib.MassProton)）
        peak_mzs_with_proton = peak_mzs + AtomLib.MassProton
        # 执行完整去同位素去电荷
        final_masses, final_intens = Preprocess.deisotope_and_decharge(
            peak_mzs_with_proton, peak_intens, rel_mass_err=2e-4, prec_z=3
        )
        # 减去质子质量（与C#测试保持一致的断言基准）
        final_masses_no_proton = final_masses - AtomLib.MassProton

        # 断言最终结果
        np.testing.assert_allclose(
            final_masses_no_proton, expected_merged_masses, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(final_intens, expected_merged_intens)


if __name__ == "__main__":
    unittest.main()
