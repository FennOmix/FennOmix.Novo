import argparse
import logging
import time
from pathlib import Path
from typing import List

import pandas as pd
from foxnovo.model.config import Config, setup_runtime
from foxnovo.model.denovo import ModelRunner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_hdf5_files(folder: str) -> List[Path]:
    """获取文件夹中所有的hdf5文件"""
    folder_path = Path(folder)
    hdf5_files = list(folder_path.glob("*.hdf5")) + list(folder_path.glob("*.h5"))
    return sorted(hdf5_files)


def test_inference_speed(
    hdf_folder: str,
    model_weight: str,
    output_folder: str,
) -> None:
    """
    测试模型推理速度
    
    Args:
        hdf_folder: 包含hdf5文件的文件夹路径
        model_weight: 模型权重文件路径
        output_folder: 结果输出文件夹路径
    """
    # 初始化模型配置
    mconfig = Config()
    setup_runtime(mconfig)
    config = mconfig.config
    
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化模型runner
    runner = ModelRunner(config, model_weight)
    logger.info("Model initialized successfully")
    
    # 获取所有hdf5文件
    hdf5_files = get_hdf5_files(hdf_folder)
    
    if not hdf5_files:
        logger.warning(f"No HDF5 files found in {hdf_folder}")
        return
    
    logger.info(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # 存储每个文件的运行时间
    results = []
    total_start_time = time.time()
    
    for idx, file_path in enumerate(hdf5_files, 1):
        logger.info(f"Processing [{idx}/{len(hdf5_files)}]: {file_path.name}")
        
        try:
            # 记录单个文件的开始时间
            file_start_time = time.time()
            
            # 执行推理
            df = runner.predict_one_file(str(file_path))
            
            # 记录单个文件的结束时间
            file_end_time = time.time()
            elapsed_time = file_end_time - file_start_time
            
            # 保存结果
            num_predictions = len(df) if df is not None else 0
            
            result_record = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'num_predictions': num_predictions,
                'elapsed_time_seconds': elapsed_time,
                'elapsed_time_minutes': elapsed_time / 60,
                'status': 'success'
            }
            
            results.append(result_record)
            
            logger.info(
                f"✓ Completed: {file_path.name} | "
                f"Predictions: {num_predictions} | "
                f"Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)"
            )
            
        except Exception as e:
            logger.error(f"✗ Failed to process {file_path.name}: {str(e)}")
            result_record = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'num_predictions': 0,
                'elapsed_time_seconds': 0,
                'elapsed_time_minutes': 0,
                'status': f'failed: {str(e)}'
            }
            results.append(result_record)
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 添加汇总统计
    successful_count = (results_df['status'] == 'success').sum()
    total_time = results_df[results_df['status'] == 'success']['elapsed_time_seconds'].sum()
    avg_time = total_time / successful_count if successful_count > 0 else 0
    min_time = results_df[results_df['status'] == 'success']['elapsed_time_seconds'].min() if successful_count > 0 else 0
    max_time = results_df[results_df['status'] == 'success']['elapsed_time_seconds'].max() if successful_count > 0 else 0
    
    # 保存CSV文件
    csv_path = output_path / 'inference_speed_results.csv'
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to: {csv_path}")
    
    # 打印汇总信息
    logger.info("\n" + "="*80)
    logger.info("INFERENCE SPEED TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files: {len(hdf5_files)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {len(hdf5_files) - successful_count}")
    logger.info(f"\nTiming Statistics (successful files only):")
    logger.info(f"  Total time: {total_time:.2f}s ({total_time/60:.2f}min)")
    logger.info(f"  Average time: {avg_time:.2f}s ({avg_time/60:.2f}min)")
    logger.info(f"  Min time: {min_time:.2f}s ({min_time/60:.2f}min)")
    logger.info(f"  Max time: {max_time:.2f}s ({max_time/60:.2f}min)")
    logger.info(f"\nTotal test duration: {total_elapsed_time:.2f}s ({total_elapsed_time/60:.2f}min)")
    logger.info("="*80 + "\n")
    
    # 保存汇总统计
    summary = {
        'metric': ['Total Files', 'Successful', 'Failed', 'Total Time (s)', 'Total Time (min)', 
                   'Average Time (s)', 'Average Time (min)', 'Min Time (s)', 'Max Time (s)'],
        'value': [
            len(hdf5_files),
            successful_count,
            len(hdf5_files) - successful_count,
            f'{total_time:.2f}',
            f'{total_time/60:.2f}',
            f'{avg_time:.2f}',
            f'{avg_time/60:.2f}',
            f'{min_time:.2f}',
            f'{max_time:.2f}'
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_path = output_path / 'inference_speed_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test FoxNovo model inference speed on HDF5 files"
    )
    parser.add_argument(
        "--hdf_folder",
        type=str,
        required=True,
        help="Path to folder containing HDF5 files"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weight file (.ckpt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output folder for results"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting inference speed test...")
    logger.info(f"HDF folder: {args.hdf_folder}")
    logger.info(f"Model weight: {args.model}")
    logger.info(f"Output folder: {args.output}")
    
    test_inference_speed(
        hdf_folder=args.hdf_folder,
        model_weight=args.model,
        output_folder=args.output,
    )
    
    logger.info("Test completed!")
