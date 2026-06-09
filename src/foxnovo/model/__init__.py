# model/__init__.py
"""FoxNovo - de novo peptide sequencing model

FoxNovo is a non-autoregressive deep learning model for peptide sequence
inference from mass spectrometry data.
"""

# Core model (FoxNovo)
from .checkpoint import load_encoder_weight, load_model_weight

# Configuration (`Modelconfig` is kept as a backward-compatible alias).
from .config import Config, ModelConfig, Modelconfig, get_default_device, seeding

# Training & Inference (de novo methodology)
from .denovo import ModelRunner, predict, train
from .foxnovo import FoxNovoNARModel
from .losses import WeightedCrossEntropyLoss
from .scheduler import CosineWarmupScheduler

# Utilities
from .utils import PeptideMass, batch_truncate_after_eos, pep_recall_evaluate

__all__ = [
    # Core model
    "FoxNovoNARModel",
    # Training framework
    "ModelRunner",
    "train",
    "predict",
    # Configuration
    "Config",
    "ModelConfig",
    "Modelconfig",
    "get_default_device",
    "seeding",
    # Utilities
    "PeptideMass",
    "batch_truncate_after_eos",
    "pep_recall_evaluate",
    "WeightedCrossEntropyLoss",
    "CosineWarmupScheduler",
    "load_model_weight",
    "load_encoder_weight",
]

__version__ = "0.1.0"
