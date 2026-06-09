import contextlib
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

AA_MASS = {
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

MOD_TO_AA_TOKEN = {
    "Carbamidomethyl@C": "C+57.021",
    "Oxidation@M": "M+15.995",
    "Cysteinyl@C": "C+119.004",
}


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


@dataclass
class ModelConfig:
    train_scratch: bool = True
    use_weighted_sample: bool = True  # useful for long tail data
    use_chunked_weighted_sample: bool = (
        False  # Set as True when dataset size large than 1800w, and set use_weighted_sample=False
    )
    use_weighted_score: bool = True  # useful for low quality data
    score_mean: float = 2.16
    score_std: float = 0.75
    weight_min: float = 0.5
    weight_max: float = 1.5

    random_seed: int = 454
    n_peaks: int = 150
    min_mz: float = 50.0
    max_mz: int = 2500
    min_intensity: float = 0.01
    remove_precursor_tol: float = 2.0
    max_charge: int = 10
    top_k: int = 10
    top_k_output: int = 10
    precursor_mass_tol: int = 2  # Da
    precursor_mass_ppm: int = 20  # ppm
    fragment_mass_ppm: int = 20  # ppm
    min_length: int = 8
    max_length: int = 14
    dim_model: int = 512
    n_head: int = 8
    dim_feedforward: int = 1024
    n_layers: int = 9
    dropout: float = 0.1
    dim_intensity: int | None = None
    residues: dict[str, float] = field(default_factory=AA_MASS.copy)
    train_label_smoothing: float = 0.01
    model_save_path: str = ""
    warmup_iters: int = 100_000
    cosine_schedule_period_iters: int = 600_000
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    train_batch_size: int = 64
    eval_batch_size: int = 256  # recommend 64 or 128
    max_epochs: int = 20
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )  # cpu or cuda
    cpu_threads: int = 3  # work if using cpu, threads every process, all = m x n
    cpu_process: int = 8  # work if using cpu


@dataclass
class Config:
    device: torch.device | None = None
    device_str: str | None = None
    config: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        try:
            self.device_str = self.config.device
            if self.device_str == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
                self.device_str = "cpu"
                self.config.device = "cpu"
            self.device = torch.device(self.device_str)
        except:  # noqa: E722
            self.device, self.device_str = get_default_device()
            self.config.device = self.device_str


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeding(seed: int) -> None:
    """Backward-compatible alias for seed_everything."""
    seed_everything(seed)


def setup_runtime(config: Config) -> None:
    if config.device_str == "cpu":
        threads = config.config.cpu_threads
        torch.set_num_threads(threads)
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = True
        torch.jit.enable_onednn_fusion(True)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.nn.functional.scaled_dot_product_attention  # noqa: B018
        except:  # noqa: E722
            pass
        torch.backends.openmp.enabled = True
        torch.set_flush_denormal(True)
    seed_everything(config.config.random_seed)


# Backward-compatible alias for older imports.
Modelconfig = ModelConfig
