# FennOmix.Novo (FoxNovo)

`foxnovo` provides training and prediction for de novo peptide sequencing from HDF5-based MS/MS data, with optional one-step conversion from MGF, mzML, and Thermo RAW inputs at prediction time.

## Installation

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

This installs the package along with all runtime dependencies needed for prediction and training.

If you want GPU acceleration, install a CUDA-enabled PyTorch build using the official PyTorch install command for your system before or after the editable install.

## Input Data

Training expects HDF5 inputs. Prediction defaults to HDF5-only behavior unless you explicitly set `--data-format`.

- Training expects folders containing annotated `.hdf5` files with peptide/PSM information.
- Prediction without `--data-format` expects existing `.hdf` / `.hdf5` inputs.
- Prediction can also convert `.mgf`, `.mzML`, or Thermo `.raw` inputs when `--data-format` is set explicitly.

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

Single HDF file prediction also works:

```bash
foxnovo predict \
  --predict-folder /path/to/sample.hdf5 \
  --model-weights /path/to/model.ckpt \
  --output-folder /path/to/output_csv
```

Explicit MGF conversion and prediction:

```bash
foxnovo predict \
  --predict-folder /path/to/predict_mgf \
  --data-format mgf \
  --model-weights /path/to/model.ckpt \
  --output-folder /path/to/output_csv
```

Explicit mzML conversion and prediction:

```bash
foxnovo predict \
  --predict-folder /path/to/predict_mzml \
  --data-format mzml \
  --model-weights /path/to/model.ckpt \
  --output-folder /path/to/output_csv
```

Explicit Thermo RAW conversion and prediction:

```bash
foxnovo predict \
  --predict-folder /path/to/predict_raw \
  --data-format raw \
  --model-weights /path/to/model.ckpt \
  --output-folder /path/to/output_csv
```

When conversion is needed, FoxNovo writes cached HDF files under `<output_folder>/_foxnovo_hdf_cache/`. Thermo RAW support depends on AlphaRaw and its Python/.NET reader support in your local envi[...]

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
- `sequence`
- `mods`
- `mod_sites`
- `charge`
- `precursor_mz`
- `nAA`
- `modified_sequence mass`
- `precursor_mass`
- `score`
- `matched_ion_count`
- `matched_ion_ratio`
- `b_matched_ion_count`
- `y_matched_ion_count`
- `b_matched_ion_ratio`
- `y_matched_ion_ratio`

## CPU/GPU Note

GPU is recommended for normal model use. CPU prediction is supported, including the current multiprocessing prediction path, but it is slower.
Inference speed is about 2800+ spectra/s on single RTX 4090 and 190+ spectra/s on AMD EPYC 9554 64-Core Processor platform based on our large-scale benchmark.
