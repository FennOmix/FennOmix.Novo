import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


@dataclass
class Modelconfig:
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
    residues: dict[str, float] = field(
        default_factory=lambda: {
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
    )
    train_label_smoothing: float = 0.01
    model_save_path: str = ""
    warmup_iters: int = 100_000
    cosine_schedule_period_iters: int = 600_000
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    train_batch_size: int = 64
    eval_batch_size: int = 128  # recommend 64 or 128
    max_epochs: int = 20
    device: str = "cpu"  # cpu or cuda, if None: try to get cuda


@dataclass
class Config:
    device: torch.device | None = None
    device_str: str | None = None
    config: Modelconfig = field(default_factory=Modelconfig)

    def __post_init__(self):
        try:
            self.device_str = self.config.device
            self.device = torch.device(self.device_str)
        except:  # noqa: E722
            self.device, self.device_str = get_default_device()


def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
