import csv
import datetime
import os
import tempfile
from pathlib import Path

import neptune
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fennomix_novo.data_set import hdf_dataloader
from fennomix_novo.model.checkpoint import load_encoder_weight, load_model_weight
from fennomix_novo.model.config import Config, Modelconfig
from fennomix_novo.model.foxnovo import FoxNovoNARModel
from fennomix_novo.model.scheduler import CosineWarmupScheduler
from fennomix_novo.scoring import pGlyco_score


class ModelRunner:
    def __init__(
        self,
        config: Modelconfig,
        model_filename: None,
    ) -> None:
        self.config = config
        self.model_filename = model_filename

        self.device = torch.device(self.config.device)

        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        self.model_save_path = Path(self.config.model_save_path)
        self.min_valid_losses = 5.0
        self.max_recall = 0.0

        # ===================== 速度记录相关 =====================
        self.time_logs = [["hdf5文件名", "开始时间", "完成时间", "运行时长", "状态"]]
        self.output_root = ""

    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

        # ===================== 退出时保存速度日志 =====================
        if self.output_root and len(self.time_logs) > 1:
            csv_log_path = os.path.join(self.output_root, "predict_time_log.csv")
            with open(csv_log_path, "w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerows(self.time_logs)
            print(f"\n📊 速度计时日志已保存：{csv_log_path}")

    def train(
        self,
        train_folder: str,
        val_folder: str,
    ) -> None:
        self.initialize_model()
        run = neptune.init_run(
            project="FennOmix/FeNNetNovo",
            tags=["Task0_baseline"],
            dependencies="infer",
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
            monitoring_namespace="monitoring",
            mode="offline",
        )

        self.loaders = hdf_dataloader.DeNovoDataModule(
            train_folder=train_folder,
            val_folder=val_folder,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            n_peaks=self.config.n_peaks,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            random_state=self.config.random_seed,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            weighted_sample=self.config.use_weighted_sample,
            chunked_weighted_sample=self.config.use_chunked_weighted_sample,
        )
        self.loaders.setup()

        train_loader = self.loaders.get_train_loader()
        valid_loader = self.loaders.get_val_loader()
        train_eval_loader = self.loaders.get_train_eval_loader()

        num_epochs = self.config.max_epochs
        print("model_save_path:", self.model_save_path)
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.model.train()
            train_losses = []
            epoch_loss = 0
            progress_bar = tqdm(train_loader)
            for _step, batch in enumerate(progress_bar):
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                loss, peptide_recall = self.model.training_step(batch, mode="train")
                run["train/step_loss"].log(loss)
                run["train/step_pep_top1_recall"].log(peptide_recall)
                run["train/lr"].log(self.optimizer.param_groups[0]["lr"])
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                train_losses.append(loss.item())
                epoch_loss += loss.item()
                progress_bar.set_postfix(
                    train_loss=loss.item(), train_pep_top1_recall=peptide_recall
                )

            epoch_loss /= len(train_loader)
            self.model.eval()
            train_eval_loss, train_eval_recall, train_eval_dp_recall = 0, 0, 0
            with torch.no_grad():
                for batch in train_eval_loader:
                    batch = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                    )
                    loss, recall, dp_recall = self.model.training_step(batch, mode="val")
                    train_eval_loss += loss.item()
                    train_eval_recall += recall
                    train_eval_dp_recall += dp_recall if dp_recall is not None else 0

            train_eval_loss /= len(train_eval_loader)
            train_eval_recall /= len(train_eval_loader)
            train_eval_dp_recall /= len(train_eval_loader)

            val_loss, top1_val_peptide_recall, val_dp_recall = self.validate(valid_loader)

            if top1_val_peptide_recall > self.max_recall:
                self.max_recall = top1_val_peptide_recall
                model_save_path = (
                    str(self.model_save_path).replace(".ckpt", "")
                    + "_val1_recall_"
                    + f"{top1_val_peptide_recall:.3f}"
                    + ".ckpt"
                )
                torch.save(self.model.state_dict(), model_save_path)
            print(
                f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, "
                f"Train Eval Loss = {train_eval_loss:.4f}, Train Eval Top1 Recall = {train_eval_recall:.3f}, Train Eval Top10 Recall = {train_eval_dp_recall:.3f}, "
                f"Val Loss = {val_loss:.4f}, Val Top1 Recall = {top1_val_peptide_recall:.3f}, Val Top10 Recall = {val_dp_recall:.3f}"
            )
            run["train/epoch_loss"].log(epoch_loss)
            run["val/epoch_loss"].log(val_loss)
            run["val/epoch_pep_top1_recall"].log(top1_val_peptide_recall)
            run["train_eval_loss"].log(train_eval_loss)
            run["train_eval_recall"].log(train_eval_recall)
            run["train_eval/dp_top10_recall"].log(train_eval_dp_recall)
            run["val/dp_top10_recall"].log(val_dp_recall)

    def validate(self, valid_loader):
        self.model.eval()
        losses = []
        recalls = []
        dp_recalls = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                loss, top1_recall, dp_recall = self.model.training_step(batch, mode="val")

                losses.append(float(loss.item()) if not torch.isnan(loss) else 0.0)
                recalls.append(float(top1_recall) if top1_recall is not None else 0.0)
                dp_recalls.append(
                    float(dp_recall) if dp_recall is not None and not np.isnan(dp_recall) else 0.0
                )

        avg_loss = np.mean(losses) if losses else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_dp_recall = np.mean(dp_recalls) if dp_recalls else 0.0

        return avg_loss, avg_recall, avg_dp_recall

    def predict(self, predict_folder: str, out_put_folder: str):
        folder_path = Path(predict_folder)
        test_files = sorted(folder_path.glob("*.hdf5"))  # 保证顺序
        self.output_root = out_put_folder  # 保存日志路径
        self.initialize_model(mode="predict")

        success_count = 0
        skip_count = 0
        fail_count = 0

        print(f"\n🚀 开始预测，共 {len(test_files)} 个 .hdf5 文件")

        for test_file_path in tqdm(test_files, desc="预测进度", unit="文件"):
            file_name = test_file_path.name
            score_top_1_output_csv_path = os.path.join(
                out_put_folder, test_file_path.stem + "_result.csv"
            )

            # ===================== 跳过已存在文件 =====================
            if os.path.exists(score_top_1_output_csv_path):
                file_size_kb = os.path.getsize(score_top_1_output_csv_path) / 1024
                if file_size_kb > 10:  # 大于10kb跳过
                    tqdm.write(f"⏭️  已跳过：{file_name} ({file_size_kb:.2f}KB)")
                    self.time_logs.append([file_name, "-", "-", "-", "已跳过"])
                    skip_count += 1
                    continue

            # ===================== 计时开始 =====================
            start_time = datetime.datetime.now()
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            status = "成功"

            try:
                print(f"\n📄 处理：{test_file_path.name}")
                self.loaders = hdf_dataloader.DeNovoDataModule(
                    test_path=test_file_path,
                    eval_batch_size=self.config.eval_batch_size,
                    n_peaks=self.config.n_peaks,
                    min_mz=self.config.min_mz,
                    max_mz=self.config.max_mz,
                    min_intensity=self.config.min_intensity,
                    remove_precursor_tol=self.config.remove_precursor_tol,
                    annotated=False,
                )
                self.loaders.setup()
                predict_result = []
                predict_loader = self.loaders.get_test_loader()
                self.model.eval()

                with torch.no_grad():
                    for batch in tqdm(predict_loader, leave=False):
                        batch = tuple(
                            x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                        )
                        predict_batch_table = self.model.predict_step(batch)
                        predict_result.extend(predict_batch_table)

                # 保存结果
                if predict_result:
                    all_merged = np.vstack(predict_result)
                    columns = ["spec_idx", "modified_sequence", "nar_dp_score", "nar_dp_top"]
                    df = pd.DataFrame(all_merged, columns=columns)
                    scored_top1_df, filtered_scored_top1_df = pGlyco_score.score_sequence(
                        df, test_file_path
                    )
                    filtered_scored_top1_df.to_csv(score_top_1_output_csv_path, index=False)

                success_count += 1

            except Exception as e:
                print(f"❌ 处理失败：{file_name}, 错误：{str(e)}")
                status = "失败"
                fail_count += 1

            # ===================== 计时结束 =====================
            end_time = datetime.datetime.now()
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            duration = end_time - start_time
            duration_str = str(duration).split(".")[0]

            # 记录日志
            self.time_logs.append([file_name, start_str, end_str, duration_str, status])
            tqdm.write(f"✅ {status} | {file_name} | 耗时：{duration_str}")

        # ===================== 最终统计 =====================
        print("\n" + "=" * 60)
        print(f"📊 预测完成！成功：{success_count} | 跳过：{skip_count} | 失败：{fail_count}")
        print("=" * 60)

    def initialize_model(self, mode=None):
        self.model = FoxNovoNARModel(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            residues=self.config.residues,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            min_length=self.config.min_length,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            top_k=self.config.top_k,
            top_k_output=self.config.top_k_output,
            use_weighted_score=self.config.use_weighted_score,
            score_mean=self.config.score_mean,
            score_std=self.config.score_std,
            weight_min=self.config.weight_min,
            weight_max=self.config.weight_max,
            precursor_mass_ppm=self.config.precursor_mass_ppm,
        ).to(self.device)

        if mode == "predict":
            self.model = load_model_weight(self.model, self.model_filename)
        else:
            if not self.config.train_scratch:
                self.model = load_encoder_weight(self.model, self.model_filename)
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                self.lr_scheduler = CosineWarmupScheduler(
                    self.optimizer,
                    self.config.warmup_iters,
                    self.config.cosine_schedule_period_iters,
                )
            else:
                print("Training from scratch...")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                self.lr_scheduler = CosineWarmupScheduler(
                    self.optimizer,
                    self.config.warmup_iters,
                    self.config.cosine_schedule_period_iters,
                )


def train(
    train_folder: str,
    val_folder: str,
    model: str | None,
) -> None:
    mconfig = Config()
    config = mconfig.config
    with ModelRunner(config, model) as runner:
        print("Training model from:")
        print(f"  {train_folder}")

        print("Validating on:")
        print(f"  {val_folder}")
        runner.train(train_folder, val_folder)
    print("Training Done")


def predict(
    predict_folder: str,
    model: str | None,
    out_put_folder: str,
) -> None:
    mconfig = Config()
    config = mconfig.config
    with ModelRunner(config, model) as runner:
        print("Predicting model from:")
        print(f" {predict_folder}")
        runner.predict(predict_folder, out_put_folder)
    print("Predicting Done")
