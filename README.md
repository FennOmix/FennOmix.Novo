# FennOmix.Novo (FoxNovo)

`foxnovo` provides training and batch prediction for de novo peptide sequencing from HDF5-based MS/MS data.

## Installation

Install the local runtime dependencies first:

```bash
pip install -r requirements.txt
```

Then install the package in editable mode from the repository root:

```bash
pip install -e .
```

`pip install -e .` installs the normal runtime dependencies needed for prediction and training.

If you want GPU acceleration, install a CUDA-enabled PyTorch build using the official PyTorch install command for your system before or after the editable install.

## Input Data

Training and prediction expect HDF5 inputs. The current CLI/API entrypoints do not read raw vendor files or mzML directly.

- Training expects folders containing annotated `.hdf5` files with peptide/PSM information.
- Prediction expects folders containing `.hdf5` files with spectra and peak tables.

## Training

`model_save_path` is required for training.

```bash
foxnovo train \
  --train-folder /path/to/train_hdf5 \
  --val-folder /path/to/val_hdf5 \
  --model-save-path /path/to/output/model.ckpt
```

## Prediction

```bash
foxnovo predict \
  --predict-folder /path/to/predict_hdf5 \
  --model-weights /path/to/model.ckpt \
  --output-folder /path/to/output_csv
```

## Local Smoke Test

This is a local-only check for manual validation before committing. It is not intended for pytest or CI. Local data and model weights are not included in the repository.

```bash
python scripts/local_smoke_predict.py \
  --predict-folder ./local_data \
  --model-weights ./local_data/model.ckpt \
  --output-folder ./local_data
```

## Python API

```python
from foxnovo.api import predict, train
```

## Output

Prediction writes one CSV per input HDF5 file. The output columns are:

- `spec_idx`
- `modified_sequence`
- `nar_dp_score`
- `nar_dp_top`

## CPU/GPU Note

GPU is recommended for normal model use. CPU prediction is supported, including the current multiprocessing prediction path, but it is slower.
