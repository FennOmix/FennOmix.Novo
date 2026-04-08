import os
import tempfile
from pathlib import Path

import neptune
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fennomix_novo.data_set import hdf_dataloader
from fennomix_novo.scoring import pGlyco_score

from .checkpoint import load_encoder_weight, load_model_weight
from .config import Config, Modelconfig
from .foxnovo import FoxNovoNARModel
from .scheduler import CosineWarmupScheduler


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

    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

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

        train_loader = self.loaders.get_train_loader()  # train
        valid_loader = self.loaders.get_val_loader()  # val
        train_eval_loader = self.loaders.get_train_eval_loader()  # eval

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
                )  # spectra, precursors, peptides, raw_names, spec_idx, score
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
            "train_eval_dataset"
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
        test_files = folder_path.glob("*.hdf5")
        self.initialize_model(mode="predict")
        for test_file_path in test_files:
            score_top_1_output_csv_path = out_put_folder + "/" + test_file_path.stem + "_result.csv"
            file_path = Path(score_top_1_output_csv_path)
            if file_path.exists():
                continue
            print("Process:", test_file_path)
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
                for batch in tqdm(predict_loader):
                    batch = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                    )
                    # self.model.predict_step(batch)
                    predict_batch_table = self.model.predict_step(batch)
                    predict_result.extend(predict_batch_table)
            "测试速度时 跳过"
            if predict_result:
                all_merged = np.vstack(predict_result)
                columns = ["spec_idx", "modified_sequence", "nar_dp_score", "nar_dp_top"]
                df = pd.DataFrame(all_merged, columns=columns)
                scored_top1_df, filtered_scored_top1_df = pGlyco_score.score_sequence(
                    df, test_file_path
                )
                score_top_1_output_csv_path = (
                    out_put_folder + "/" + test_file_path.stem + "_result.csv"
                )
                filtered_scored_top1_df.to_csv(score_top_1_output_csv_path, index=False)

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
    out_put_folder=str,
) -> None:
    mconfig = Config()
    config = mconfig.config
    with ModelRunner(config, model) as runner:
        print("Predicting model from:")
        print(f" {predict_folder}")
        runner.predict(predict_folder, out_put_folder)
    print("Predicting Done")


predict(
    predict_folder=r"X:\chenzx\raw_Data\HLA\HLA_v1_2_all_data\HLA_v1_2_unseen\hdf_by_alpharaw\predict_result_v2_simple_mod_max_recall_0301\test_dp\hdf_test",
    out_put_folder=r"X:\chenzx\raw_Data\HLA\HLA_v1_2_all_data\HLA_v1_2_unseen\hdf_by_alpharaw\predict_result_v2_simple_mod_max_recall_0301\test_dp\hdf_test",
    model=r"X:\chenzx\raw_Data\HLA\HLA_v2_all_data\trained_weights\FeNNetNovo_HLA_v2_SOTA_simple_mod_500psm_max_recall_0301.ckpt",
)
