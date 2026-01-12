from pathlib import Path

import torch

from fennomix_novo.data_set.dataset_splitter_for_pfind_generated_mgf import HDFDataSplitter


def test_data_loading_performance():
    """Test HDF data loading performance after optimization."""
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
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 运行测试
    test_data_loading_performance()
