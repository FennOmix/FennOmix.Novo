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
```

We recommend running FoxNovo on a GPU. Install PyTorch first using the command that matches your CUDA environment. To check your CUDA version:

```bash
nvidia-smi
```

Then choose the corresponding PyTorch command from the official selector: https://pytorch.org/get-started/locally/

Example for a CUDA 12.1 PyTorch build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install FoxNovo:

```bash
pip install -e .
```

`pip install -e .` installs FoxNovo and its non-PyTorch runtime dependencies. PyTorch is intentionally not installed automatically because CPU/CUDA wheels depend on the user's local environment.

## Input Data

Training expects HDF5 inputs. Prediction defaults to HDF5-only behavior unless you explicitly set `--data-format`.

- Training expects folders containing annotated `.hdf5` files with peptide/PSM information.
- Prediction without `--data-format` expects existing `.hdf` / `.hdf5` inputs.
- Prediction can also convert `.mgf`, `.mzML`, or Thermo `.raw` inputs when `--data-format` is set explicitly.

## Model Checkpoint

The pretrained model checkpoint for HLA-I immunopeptides is available upon request. Request information is collected solely for tracking usage and demand statistics.<br>
[Request access to FoxNovo_HLAI_v1.0.ckpt](https://fennomix.lab.westlake.edu.cn/register/)

## Training

Training functionality is provided for users who want to train or fine-tune FoxNovo models on custom datasets.
If `--model-weights` is provided, FoxNovo loads the full checkpoint and fine-tunes all model parameters. If it is omitted, training starts from scratch.

```bash
foxnovo train \
  --train-folder /path/to/train_hdf5 \
  --val-folder /path/to/val_hdf5 \
  --model-save-path /path/to/output/model.ckpt
```

Fine-tune from a pretrained checkpoint:

```bash
foxnovo train \
  --train-folder /path/to/train_hdf5 \
  --val-folder /path/to/val_hdf5 \
  --model-weights /path/to/FoxNovo_HLAI_v1.0.ckpt \
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

When conversion is needed, FoxNovo writes cached HDF files under `<output_folder>/_foxnovo_hdf_cache/`. Thermo RAW support depends on AlphaRaw and its Python/.NET reader support in your local environment.

## Configuration

FoxNovo includes an editable YAML configuration template at `config.yaml`. The `--config` option is optional; when it is omitted, FoxNovo uses the built-in defaults from `ModelConfig`. CLI arguments override overlapping YAML values when they are provided.

```bash
foxnovo train --config config.yaml --train-folder /path/to/train_hdf5 --val-folder /path/to/val_hdf5 --model-save-path /path/to/model.ckpt
foxnovo predict --config config.yaml --predict-folder /path/to/predict_hdf5 --model-weights /path/to/model.ckpt --output-folder /path/to/output_csv
```

## Test

The repository includes a small MGF example at `tests/test_data/test.mgf`. After downloading a model checkpoint, you can run a quick prediction test with:

```bash
foxnovo predict \
  --predict-folder tests/test_data/test.mgf \
  --data-format mgf \
  --model-weights /path/to/FoxNovo_HLAI_v1.0.ckpt \
  --output-folder tests/test_output
```

This command converts the MGF file to a temporary prediction HDF under `tests/test_output/_foxnovo_hdf_cache/` and writes the prediction CSV to `tests/test_output/`.

## Python API

```python
from foxnovo.api import predict, train

train(train_folder="/path/to/train_hdf5", val_folder="/path/to/val_hdf5", config_path="config.yaml")
predict(
    predict_folder="/path/to/predict_hdf5",
    output_folder="/path/to/output_csv",
    model_weights="/path/to/model.ckpt",
    config_path="config.yaml",
)
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

### Recommended peptide selection and quality control

For each MS2 spectrum, multiple candidate peptide sequences may be generated using DP-decoder. We recommend the following filtering strategy:

1. **Precursor mass filtering**

   First, filter candidate sequences based on precursor mass consistency. Candidates with incompatible peptide mass and precursor mass should be removed.

2. **Select the highest-ranked candidate using `nar_dp_top`**

   After mass filtering, select the candidate with the smallest `nar_dp_top` value as the most reliable prediction for each MS2 spectrum.

3. **Score and match-ion based filtering using `score` and `matched_ion_ratio`**

   Based on our evaluation, a score cutoff of **30** provides a reliable and balance quality threshold.
   Higher fragment-ion matching ratios indicate stronger agreement between the predicted peptide sequence and the experimental MS/MS spectrum. Therefore, `matched_ion_ratio` can be used for more stringent quality control.

## CPU/GPU Note

GPU is recommended for normal model use. CPU prediction is supported, including the current multiprocessing prediction path, but it is slower.

In our large-scale benchmark using HDF5 files in the native input format expected by FoxNovo, inference throughput exceeded ~2,800 spectra/s on a single NVIDIA RTX 4090 and 200 spectra/s on an AMD EPYC 9554 64-Core Processor platform.

## Citation

> **Accurate and ultra-fast de novo HLA-I immunopeptide sequencing with FoxNovo.**
> Chen, Z.-X., You, C.-R., Tarn, C., Zhou, X.-X., & Zeng, W.-F. (2026). *LangTaoSha Preprint Server*.
> doi: [https://doi.org/10.65215/LTSpreprints.2026.08.02.000299](https://doi.org/10.65215/LTSpreprints.2026.08.02.000299)


## License

This project is released under the Apache License 2.0.
