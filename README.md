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

## Input Data

Training and prediction expect folders containing `.hdf5` files.

- Training uses annotated HDF5 files with peptide/PSM information.
- Prediction uses HDF5 files containing spectra and peak tables.

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
  --output-folder /path/to/output_csv \
  --model-weights /path/to/model.ckpt
```

## Local Smoke Test

This is a local-only check for manual validation before committing. It is not intended for pytest or CI.

```bash
python scripts/local_smoke_predict.py \
  --predict-folder ./local_data/predict_hdf5 \
  --model-weights ./local_data/model.ckpt \
  --output-folder ./local_data/smoke_output
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

The runtime selects CPU or GPU from the current configuration. CPU multiprocessing prediction is supported; GPU prediction runs on a single device path.
