# FennOmix.Novo (FoxNovo)

FoxNovo is a deep learning and combinatorial framework for accurate and scalable de novo sequencing of HLA-I immunopeptides from tandem mass spectra.

FoxNovo integrates a dual-token m/z representation, a non-autoregressive transformer architecture, and dynamic programming-based precursor-mass constrained decoding for high-confidence peptide sequencing.

`foxnovo` provides training and prediction for de novo peptide sequencing from HDF5-based MS/MS data, with optional one-step conversion from MGF, mzML, and Thermo RAW inputs at prediction time.

## Installation

### Create a conda environment

FoxNovo requires Python 3.11. We recommend installing FoxNovo in a dedicated conda environment:

```bash
conda create -n foxnovo_env python=3.11
conda activate foxnovo_env
pip install -e .
```

This installs the package along with all runtime dependencies needed for prediction and training. Installation usually takes less than 10 minutes, excluding CUDA-enabled PyTorch installation.

If you want GPU acceleration, install a CUDA-enabled PyTorch build using the official PyTorch install command for your system before or after the editable install.

## Input Data

Training expects HDF5 inputs. Prediction defaults to HDF5-only behavior unless you explicitly set `--data-format`.

- Training expects folders containing annotated `.hdf5` files with peptide/PSM information.
- Prediction without `--data-format` expects existing `.hdf` / `.hdf5` inputs.
- Prediction can also convert `.mgf`, `.mzML`, or Thermo `.raw` inputs when `--data-format` is set explicitly.

## Model Checkpoint

The pretrained model checkpoint specific for HLA-I immunopeptides is available here:
[FoxNovo_HLAI_v1.0.ckpt](https://drive.google.com/file/d/1qsWbAUUT1FXr8qAIpgkTJS2hBDr-0fCP/view?usp=drive_link)

## Training

Training functionality is provided for users who want to train or fine-tune FoxNovo models on custom datasets.

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


## Python API

```python
from foxnovo.api import predict, train
```

## Output

The output CSV contains predicted peptide sequences, precursor information, decoding scores and fragment-ion matching statistics.
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
Performance depends on hardware and input data characteristics. Inference speed is about 2800+ spectra/s on single RTX 4090 and 190+ spectra/s on AMD EPYC 9554 64-Core Processor platform based on our large-scale benchmark.

## License

This project is released under the Apache License 2.0.
