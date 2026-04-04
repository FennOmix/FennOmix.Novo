"""
FennOmix Novo Dev - Unified Train and Predict API (Development Version)

This module provides a high-level API and CLI for training and prediction with the
DeNovo Dev peptide sequencing model (denovo_dev.py).

Key features:
- Weighted sampling for long-tail data
- Weighted scoring for low-quality data
- Dual decoder system (NAR + DP)
- Mass tolerance in both Da and ppm

Usage:
    # Train
    python -m fennomix_novo.api_denovo_dev train --train-folder /path/to/train --val-folder /path/to/val
    # Predict
    python -m fennomix_novo.api_denovo_dev predict --predict-folder /path/to/test --output-folder /path/to/output --model-weights /path/to/model.ckpt
"""

import argparse
import sys
import warnings
from pathlib import Path

from fennomix_novo.model.denovo_dev import Config, ModelRunner


class DeNovoDevAPI:
    """
    High-level API for FennOmix Novo Dev de novo peptide sequencing model.
    This is the development version with enhanced features:
    - Weighted sampling for long-tail data handling
    - Weighted scoring for low-quality spectrum handling
    - Dual decoder system (Non-Autoregressive + Dynamic Programming)
    - Improved mass tolerance handling (Da and ppm)
    This class provides simplified interfaces for:
    - Training the model on new data
    - Making predictions on MS/MS spectra
    Examples:
        >>> api = DeNovoDevAPI()
        >>> api.train(
        ...     train_folder="/path/to/train",
        ...     val_folder="/path/to/val",
        ... )
        >>> api.predict(
        ...     predict_folder="/path/to/test",
        ...     output_folder="/path/to/output",
        ...     model_weights="/path/to/model.ckpt",
        ... )
    """

    def __init__(self, model_weights: str | None = None):
        """
        Initialize the DeNovo Dev API.
        Args:
            model_weights (Optional[str]): Path to pre-trained model weights.
                If None, will initialize model from scratch or use default config.
        Examples:
            >>> # Initialize with default configuration
            >>> api = DeNovoDevAPI()
            >>> # Initialize with pre-trained weights
            >>> api = DeNovoDevAPI(
            ...     model_weights="/path/to/model.ckpt"
            ... )
        """
        self.model_weights = model_weights
        self.config = Config()
        self.runner = None

    def train(
        self,
        train_folder: str,
        val_folder: str,
        model_save_path: str | None = None,
        batch_size: int = 64,
        max_epochs: int = 20,
        learning_rate: float = 0.0001,
        train_scratch: bool = True,
        use_weighted_sample: bool = True,
        use_weighted_score: bool = True,
        **kwargs,
    ) -> None:
        """
        Train the DeNovo Dev model on new data.
        Args:
            train_folder (str): Path to training data folder containing HDF5 files.
            val_folder (str): Path to validation data folder containing HDF5 files.
            model_save_path (Optional[str]): Path where to save the trained model.
                If None, uses default path from config.
            batch_size (int): Training batch size. Default: 64.
            max_epochs (int): Maximum number of epochs to train. Default: 20.
            learning_rate (float): Initial learning rate. Default: 0.0001.
            train_scratch (bool): Train from scratch or fine-tune pre-trained encoder. Default: True.
            use_weighted_sample (bool): Use weighted sampling for long-tail data. Default: True.
            use_weighted_score (bool): Use weighted scoring for low-quality spectra. Default: True.
            **kwargs: Additional keyword arguments for model configuration.
        Returns:
            None
        Raises:
            FileNotFoundError: If train_folder or val_folder don't exist.
        Examples:
            >>> api = DeNovoDevAPI()
            >>> api.train(
            ...     train_folder="/data/train",
            ...     val_folder="/data/val",
            ...     model_save_path="/models/trained_model.ckpt",
            ...     batch_size=32,
            ...     max_epochs=50,
            ...     use_weighted_sample=True,
            ... )
        """
        # Validate input paths
        train_path = Path(train_folder)
        val_path = Path(val_folder)
        if not train_path.exists():
            raise FileNotFoundError(f"Training folder not found: {train_folder}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation folder not found: {val_folder}")
        # Update configuration
        self.config.config.train_batch_size = batch_size
        self.config.config.max_epochs = max_epochs
        self.config.config.learning_rate = learning_rate
        self.config.config.train_scratch = train_scratch
        self.config.config.use_weighted_sample = use_weighted_sample
        self.config.config.use_weighted_score = use_weighted_score
        if model_save_path:
            self.config.config.model_save_path = model_save_path
        # Update with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self.config.config, key):
                setattr(self.config.config, key, value)
            else:
                warnings.warn(f"Unknown configuration parameter: {key}", UserWarning, stacklevel=2)
        # Train model
        print("=" * 70)
        print("DENOVO DEV MODEL TRAINING")
        print("=" * 70)
        print(f"Training folder: {train_folder}")
        print(f"Validation folder: {val_folder}")
        print(f"Batch size: {batch_size}")
        print(f"Max epochs: {max_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Train from scratch: {train_scratch}")
        print(f"Use weighted sampling: {use_weighted_sample}")
        print(f"Use weighted scoring: {use_weighted_score}")
        if model_save_path:
            print(f"Model save path: {model_save_path}")
        print("=" * 70)
        with ModelRunner(self.config.config, self.model_weights) as runner:
            runner.train(train_folder, val_folder)
        print("=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

    def predict(
        self,
        predict_folder: str,
        output_folder: str,
        model_weights: str | None = None,
    ) -> None:
        """
        Run prediction on MS/MS spectra using the DeNovo Dev model.
        The DeNovo Dev model uses a dual decoder system (NAR + DP) and provides:
        - Modified sequence predictions
        - NAR-DP scores
        - Top-10 DP predictions
        - Filtered results based on scoring
        Args:
            predict_folder (str): Path to folder containing HDF5 files for prediction.
            output_folder (str): Path where to save prediction results as CSV files.
            model_weights (Optional[str]): Path to model weights to use for prediction.
                If None, uses weights provided during initialization.
        Returns:
            None
        Raises:
            FileNotFoundError: If predict_folder doesn't exist.
            ValueError: If no model weights are specified.
        Examples:
            >>> api = DeNovoDevAPI(model_weights="/models/trained_model.ckpt")
            >>> api.predict(
            ...     predict_folder="/data/test_spectra",
            ...     output_folder="/results/predictions",
            ... )
        """
        # Validate input path
        predict_path = Path(predict_folder)
        if not predict_path.exists():
            raise FileNotFoundError(f"Prediction folder not found: {predict_folder}")
        # Determine which model weights to use
        weights_to_use = model_weights or self.model_weights
        if not weights_to_use:
            raise ValueError(
                "No model weights specified. Please provide model_weights "
                "either during initialization or in the predict() call."
            )
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        # Run prediction
        print("=" * 70)
        print("DENOVO DEV MODEL PREDICTION")
        print("=" * 70)
        print(f"Prediction folder: {predict_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Model weights: {weights_to_use}")
        print("=" * 70)
        print("Output format: spec_idx, modified_sequence, nar_dp_score, nar_dp_top")
        print("=" * 70)
        with ModelRunner(self.config.config, weights_to_use) as runner:
            runner.predict(predict_folder, str(output_folder))
        print("=" * 70)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("=" * 70)


# Convenience functions for quick usage
def train(train_folder: str, val_folder: str, model_weights: str | None = None, **kwargs) -> None:
    """
    Quick function to train the model without creating an API instance.
    Args:
        train_folder (str): Path to training data folder.
        val_folder (str): Path to validation data folder.
        model_weights (Optional[str]): Path to pre-trained model weights.
        **kwargs: Additional arguments passed to DeNovoDevAPI.train()
    Examples:
        >>> from fennomix_novo.api_denovo_dev import train
        >>> train(
        ...     train_folder="/data/train",
        ...     val_folder="/data/val",
        ...     model_save_path="/models/my_model.ckpt",
        ... )
    """
    api = DeNovoDevAPI(model_weights=model_weights)
    api.train(train_folder, val_folder, **kwargs)


def predict(
    predict_folder: str,
    output_folder: str,
    model_weights: str,
) -> None:
    """
    Quick function to run prediction without creating an API instance.
    Args:
        predict_folder (str): Path to prediction data folder.
        output_folder (str): Path to save results.
        model_weights (str): Path to model weights.
    Examples:
        >>> from fennomix_novo.api_denovo_dev import predict
        >>> predict(
        ...     predict_folder="/data/test",
        ...     output_folder="/results",
        ...     model_weights="/models/trained_model.ckpt",
        ... )
    """
    api = DeNovoDevAPI(model_weights=model_weights)
    api.predict(predict_folder, output_folder)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="fennomix_novo_dev",
        description="FennOmix Novo Dev - DeNovo Peptide Sequencing Model (Development)",
        epilog="For more information, visit the project repository.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # ==================== TRAIN COMMAND ====================
    train_parser = subparsers.add_parser("train", help="Train the DeNovo Dev model")
    train_parser.add_argument(
        "--train-folder",
        required=True,
        type=str,
        help="Path to training data folder containing HDF5 files",
    )
    train_parser.add_argument(
        "--val-folder",
        required=True,
        type=str,
        help="Path to validation data folder containing HDF5 files",
    )
    train_parser.add_argument(
        "--model-weights",
        type=str,
        default=None,
        help="Path to pre-trained model weights (optional)",
    )
    train_parser.add_argument(
        "--model-save-path", type=str, default=None, help="Path to save the trained model"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size (default: 64)"
    )
    train_parser.add_argument(
        "--max-epochs", type=int, default=20, help="Maximum number of epochs (default: 20)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Initial learning rate (default: 0.0001)",
    )
    train_parser.add_argument(
        "--train-scratch",
        action="store_true",
        default=True,
        help="Train from scratch instead of fine-tuning (default: True)",
    )
    train_parser.add_argument(
        "--no-train-scratch",
        dest="train_scratch",
        action="store_false",
        help="Fine-tune pre-trained encoder instead of training from scratch",
    )
    train_parser.add_argument(
        "--use-weighted-sample",
        action="store_true",
        default=True,
        help="Use weighted sampling for long-tail data (default: True)",
    )
    train_parser.add_argument(
        "--no-weighted-sample",
        dest="use_weighted_sample",
        action="store_false",
        help="Disable weighted sampling",
    )
    train_parser.add_argument(
        "--use-weighted-score",
        action="store_true",
        default=True,
        help="Use weighted scoring for low-quality spectra (default: True)",
    )
    train_parser.add_argument(
        "--no-weighted-score",
        dest="use_weighted_score",
        action="store_false",
        help="Disable weighted scoring",
    )
    # ==================== PREDICT COMMAND ====================
    predict_parser = subparsers.add_parser("predict", help="Run prediction on MS/MS spectra")
    predict_parser.add_argument(
        "--predict-folder",
        required=True,
        type=str,
        help="Path to folder containing HDF5 files for prediction",
    )
    predict_parser.add_argument(
        "--output-folder",
        required=True,
        type=str,
        help="Path to save prediction results as CSV files",
    )
    predict_parser.add_argument(
        "--model-weights", required=True, type=str, help="Path to model weights for prediction"
    )
    return parser


def cli_train(args: argparse.Namespace) -> int:
    """Handle the train command."""
    try:
        api = DeNovoDevAPI(model_weights=args.model_weights)
        api.train(
            train_folder=args.train_folder,
            val_folder=args.val_folder,
            model_save_path=args.model_save_path,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            train_scratch=args.train_scratch,
            use_weighted_sample=args.use_weighted_sample,
            use_weighted_score=args.use_weighted_score,
        )
        return 0
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Training failed with error: {e}", file=sys.stderr)
        return 1


def cli_predict(args: argparse.Namespace) -> int:
    """Handle the predict command."""
    try:
        api = DeNovoDevAPI(model_weights=args.model_weights)
        api.predict(
            predict_folder=args.predict_folder,
            output_folder=args.output_folder,
        )
        return 0
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Prediction failed with error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "train":
        return cli_train(args)
    elif args.command == "predict":
        return cli_predict(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


# Example usage
if __name__ == "__main__":
    # ======================== CLI USAGE ========================
    # Uncomment to test as CLI:
    #
    # Train (from scratch):
    #   python api_denovo_dev.py train \
    #       --train-folder /path/to/train \
    #       --val-folder /path/to/val \
    #       --batch-size 64 \
    #       --max-epochs 20 \
    #       --train-scratch
    #
    # Train (fine-tune):
    #   python api_denovo_dev.py train \
    #       --train-folder /path/to/train \
    #       --val-folder /path/to/val \
    #       --model-weights /path/to/pretrained.ckpt \
    #       --no-train-scratch
    #
    # Predict:
    #   python api_denovo_dev.py predict \
    #       --predict-folder /path/to/test \
    #       --output-folder /path/to/output \
    #       --model-weights /path/to/model.ckpt
    #
    # ============================================================
    # Or use programmatically:
    # api = DeNovoDevAPI(model_weights="/path/to/model.ckpt")
    # api.train(train_folder="/path/to/train", val_folder="/path/to/val")
    # api.predict(predict_folder="/path/to/test", output_folder="/path/to/output")
    sys.exit(main())
