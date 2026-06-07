import contextlib
import os
import queue
import time
from pathlib import Path

import neptune
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from foxnovo.data_set import hdf_dataloader
from foxnovo.model.checkpoint import load_encoder_weight, load_model_weight
from foxnovo.model.config import Config, Modelconfig
from foxnovo.model.foxnovo import FoxNovoNARModel
from foxnovo.model.scheduler import CosineWarmupScheduler
from foxnovo.model.utils import worker_predict_step
from foxnovo.scoring import pGlyco_score

with contextlib.suppress(RuntimeError):
    mp.set_start_method("spawn", force=True)


class ModelRunner:
    def __init__(
        self,
        config: Modelconfig,
        model_filename: None,
    ) -> None:
        self.config = config
        self.model_filename = model_filename
        self.device = torch.device(self.config.device)
        self.trainer = None
        self.model = None
        self.loaders = None
        self.model_save_path = Path(self.config.model_save_path)
        self.max_recall = 0.0

    def train(
        self,
        train_folder: str,
        val_folder: str,
    ) -> None:
        if not self.config.model_save_path:
            raise ValueError("config.model_save_path must be provided for training.")
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

    def _create_predict_loader(self, hdf5_path: str):
        return hdf_dataloader.DeNovoDataModule(
            test_path=hdf5_path,
            eval_batch_size=self.config.eval_batch_size,
            n_peaks=self.config.n_peaks,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            annotated=False,
        )

    def predict_one_file(self, hdf5_path: str) -> pd.DataFrame:
        """
        input: one hdf5 file path
        output: one predicted dataframe with peak_ion_match_score
        """
        if self.model is None:
            self.initialize_model(mode="predict")
        datamodule = self._create_predict_loader(hdf5_path)
        datamodule.setup()
        loader = datamodule.get_test_loader()
        if self.device.type == "cpu" and self.config.cpu_process > 1:
            raw_results = self._predict_cpu_parallel(loader)
        else:
            raw_results = self._predict_single_device(loader)
        scored_results = self._score_predictions(raw_results, hdf5_path)
        return scored_results

    def predict_batch(self, folder: str, output_folder: str | None = None):
        """
        input: folder path with hdf5 files
        output: for each hdf5 file, one csv file with peak_ion_match_score
        """
        self.initialize_model(mode="predict")
        folder_path = Path(folder)
        hdf5_files = sorted(folder_path.glob("*.hdf5"))
        for file_path in hdf5_files:
            print(f"Processing file: {file_path.name}")
            df = self.predict_one_file(str(file_path))
            if output_folder:
                out_dir = Path(output_folder)
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / f"{file_path.stem}_result.csv", index=False)
        return

    def _predict_single_device(self, loader):
        """predict using single device (cpu or cuda)"""
        self.model.eval()
        predict_result = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference..."):
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                batch_result = self.model.predict_step(batch)
                predict_result.extend(batch_result)
        return predict_result

    def _predict_cpu_parallel(self, loader):  # noqa: C901
        n_process = self.config.cpu_process
        input_queue = mp.Queue(maxsize=n_process * 2)
        output_queue = mp.Queue()
        processes = []
        sentinels_sent = False

        for _ in range(n_process):
            p = mp.Process(
                target=worker_predict_step,
                args=(input_queue, output_queue, self.model, self.device),
                daemon=True,
            )
            p.start()
            processes.append(p)

        num_batches = len(loader)
        predict_result = []
        sent_count = 0
        received_count = 0
        loader_iter = iter(loader)

        try:
            with tqdm(
                total=num_batches, desc="predicting (CPU parallel)...", leave=False, unit="batch"
            ) as pbar:
                while received_count < num_batches:
                    while sent_count < num_batches and not input_queue.full():
                        try:
                            batch = next(loader_iter)
                            input_queue.put(batch)
                            sent_count += 1
                        except StopIteration:
                            break

                    if sent_count == num_batches and not sentinels_sent:
                        for _ in processes:
                            input_queue.put(None)
                        sentinels_sent = True

                    dead_workers = [
                        p.pid for p in processes if p.exitcode is not None and p.exitcode != 0
                    ]
                    if dead_workers:
                        raise RuntimeError(
                            f"Prediction worker exited unexpectedly before completion: {dead_workers}"
                        )

                    while received_count < sent_count:
                        try:
                            result = output_queue.get(timeout=0.01)
                        except queue.Empty:
                            if (
                                sentinels_sent
                                and not any(p.is_alive() for p in processes)
                                and received_count < sent_count
                            ):
                                raise RuntimeError(
                                    "Prediction workers exited before all queued batch results were received."
                                ) from None
                            break
                        predict_result.extend(result)
                        received_count += 1
                        pbar.update(1)
            for p in processes:
                p.join()
        except Exception:
            for p in processes:
                if p.is_alive():
                    p.terminate()
                p.join()
            raise

        return predict_result

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
            if self.device.type == "cpu":
                self.model.share_memory()
                self.model.eval()
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

    def _score_predictions(self, raw_results: list, hdf5_path: str) -> pd.DataFrame:
        """model_predicted_results -> scored_results"""
        if not raw_results:
            return pd.DataFrame()
        all_merged = np.vstack(raw_results)
        columns = ["spec_idx", "modified_sequence", "nar_dp_score", "nar_dp_top"]
        df = pd.DataFrame(all_merged, columns=columns)
        _, filtered_df = pGlyco_score.score_sequence(df, hdf5_path)
        return filtered_df


def train(
    train_folder: str,
    val_folder: str,
    model: str | None,
) -> None:
    mconfig = Config()
    config = mconfig.config
    runner = ModelRunner(config, model)
    print("Training model from:")
    print(f"  {train_folder}")
    print("Validating on:")
    print(f"  {val_folder}")
    runner.train(train_folder, val_folder)
    print("Training Done")


def predict(
    predict_folder: str,
    model: str | None,
    output_folder: str,
) -> None:
    mconfig = Config()
    config = mconfig.config
    runner = ModelRunner(config, model)
    print("Predicting model from:")
    print(f" {predict_folder}")
    runner.predict_batch(predict_folder, output_folder)
    print("Predicting Done")


if __name__ == "__main__":
    start_time = time.time()
    import torch.multiprocessing as mp

    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn", force=True)

    predict(
        predict_folder="./example_data/predict",
        model="./example_data/model.ckpt",
        output_folder="./example_output",
    )
    print("All prediction done!")
    end_time = time.time()
    print("Time cost:", end_time - start_time)
