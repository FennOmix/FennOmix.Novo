from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from alphabase.io.hdf import HDF_File

logger = logging.getLogger(__name__)

SUPPORTED_PREDICTION_FORMATS = ("hdf5", "hdf", "raw", "mzml", "mgf", "auto")

_HDF_SUFFIXES = {".hdf", ".hdf5"}

_FORMAT_SUFFIXES = {
    "hdf5": _HDF_SUFFIXES,
    "hdf": _HDF_SUFFIXES,
    "raw": {".raw"},
    "mzml": {".mzml"},
    "mgf": {".mgf"},
}

_READER_TYPES = {
    "mzml": "mzml",
    "mgf": "mgf",
}

_CHARGE_RE = re.compile(r"[-+]?\d+")


def normalize_data_format(data_format: str) -> str:
    """Normalize and validate the requested input data format."""
    if data_format is None:
        data_format = "hdf5"

    fmt = str(data_format).lower()
    if fmt not in SUPPORTED_PREDICTION_FORMATS:
        raise ValueError(
            f"Unsupported data_format '{data_format}'. "
            f"Expected one of: {', '.join(SUPPORTED_PREDICTION_FORMATS)}"
        )

    return "hdf5" if fmt == "hdf" else fmt


def _infer_format_from_suffix(path: Path) -> str:
    """Infer prediction input format from a file suffix."""
    suffix = path.suffix.lower()

    if suffix in _HDF_SUFFIXES:
        return "hdf5"
    if suffix == ".raw":
        return "raw"
    if suffix == ".mzml":
        return "mzml"
    if suffix == ".mgf":
        return "mgf"

    raise ValueError(f"Unsupported input suffix for prediction: {path}")


def _collect_files_from_path(input_path: Path, data_format: str) -> list[Path]:
    """
    Collect prediction input files from either a single file or a folder.

    Default behavior is HDF-only. Auto-detection is used only when
    data_format='auto' is explicitly requested.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input path not found: {input_path}")

    if input_path.is_file():
        if data_format == "auto":
            _infer_format_from_suffix(input_path)
            return [input_path]

        expected_suffixes = _FORMAT_SUFFIXES[data_format]
        if input_path.suffix.lower() not in expected_suffixes:
            raise ValueError(f"Input file {input_path} does not match data_format='{data_format}'.")
        return [input_path]

    if data_format == "auto":
        supported_suffixes = {".raw", ".mzml", ".mgf", ".hdf", ".hdf5"}
        files = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in supported_suffixes
        )
    else:
        suffixes = _FORMAT_SUFFIXES[data_format]
        files = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        )

    if not files:
        raise ValueError(
            f"No prediction input files found for data_format='{data_format}' in {input_path}"
        )

    return files


def _get_alpharaw_reader(data_format: str) -> Any:
    """
    Get an AlphaRaw reader for the requested non-HDF format.

    RAW uses the previously validated Thermo DDA reader path:
        ms_reader_provider.get_reader("thermo", dda=True)

    This is intentionally different from a plain "thermo_raw" reader because
    the DDA reader can expose precursor_mz / precursor_charge list fields that
    are needed for FoxNovo prediction.
    """
    import alpharaw
    from alpharaw.ms_data_base import ms_reader_provider

    alpharaw.register_all_readers()

    if data_format == "raw":
        return ms_reader_provider.get_reader("thermo", dda=True)

    if data_format not in _READER_TYPES:
        raise ValueError(
            f"Cannot convert data_format='{data_format}' to HDF. "
            "Expected one of: raw, mzml, mgf."
        )

    return ms_reader_provider.get_reader(_READER_TYPES[data_format])


def _is_list_like(value: Any) -> bool:
    """Return True for list-like precursor fields, but not for strings."""
    return isinstance(value, list | tuple | np.ndarray | pd.Series)


def _to_list(value: Any) -> list[Any]:
    """Convert scalar or list-like value to a list."""
    if _is_list_like(value):
        return list(value)
    return [value]


def _is_missing_scalar(value: Any) -> bool:
    """Robust scalar missing-value check."""
    if _is_list_like(value):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _empty_list_to_nan(value: Any) -> Any:
    """Replace [] with NaN; keep other values unchanged."""
    if _is_list_like(value) and len(value) == 0:
        return np.nan
    return value


def _coerce_charge_value(value: Any) -> float:
    """
    Convert charge values to numeric form.

    Handles values such as:
    - 2
    - 2.0
    - "2+"
    - "+2"
    - ["2+"] after explode fallback, taking the first non-empty value
    """
    if _is_list_like(value):
        if len(value) == 0:
            return np.nan
        value = list(value)[0]

    if _is_missing_scalar(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.nan

        match = _CHARGE_RE.search(value)
        if match is None:
            return np.nan

        try:
            return float(abs(int(match.group(0))))
        except ValueError:
            return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values."""
    return pd.to_numeric(series, errors="coerce")


def _explode_paired_precursor_fields(  # noqa: C901
    df: pd.DataFrame,
    first_col: str,
    second_col: str,
    source_name: str,
) -> pd.DataFrame:
    """
    Expand paired list-like precursor fields into scalar rows.

    Used for:
    - MGF: precursor_mz + charge
    - RAW: precursor_mz + precursor_charge
    """
    if first_col not in df.columns or second_col not in df.columns:
        return df

    rows: list[pd.Series] = []

    for _, row in df.iterrows():
        first_values = _to_list(row[first_col])
        second_values = _to_list(row[second_col])

        first_len = len(first_values)
        second_len = len(second_values)
        max_len = max(first_len, second_len)

        if max_len == 0:
            new_row = row.copy()
            new_row[first_col] = np.nan
            new_row[second_col] = np.nan
            rows.append(new_row)
            continue

        if first_len == 0:
            first_values = [np.nan] * max_len
        elif first_len == 1 and max_len > 1:
            first_values = first_values * max_len
        elif first_len != max_len:
            raise ValueError(
                f"{first_col} list length does not match {second_col} list length in {source_name}."
            )

        if second_len == 0:
            second_values = [np.nan] * max_len
        elif second_len == 1 and max_len > 1:
            second_values = second_values * max_len
        elif second_len != max_len:
            raise ValueError(
                f"{second_col} list length does not match {first_col} list length in {source_name}."
            )

        for first_value, second_value in zip(first_values, second_values, strict=False):
            new_row = row.copy()
            new_row[first_col] = first_value
            new_row[second_col] = second_value
            rows.append(new_row)

    if not rows:
        return df.iloc[0:0].copy()

    return pd.DataFrame(rows).reset_index(drop=True)


def _ensure_charge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure both charge and precursor_charge columns exist when one of them exists."""
    df = df.copy()

    if "charge" not in df.columns and "precursor_charge" in df.columns:
        df["charge"] = df["precursor_charge"]

    if "precursor_charge" not in df.columns and "charge" in df.columns:
        df["precursor_charge"] = df["charge"]

    return df


def _normalize_raw_spectrum_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Normalize Thermo RAW spectrum_df following the previously validated raw2hdf script.

    Key points:
    - use DDA reader upstream
    - remove MS1
    - explode precursor_mz / precursor_charge list values
    - convert empty lists to NaN
    - drop rows where both precursor_mz and precursor_charge are missing
    """
    if "ms_level" in df.columns:
        df = df[df["ms_level"] != 1].reset_index(drop=True)

    df = _explode_paired_precursor_fields(
        df=df,
        first_col="precursor_mz",
        second_col="precursor_charge",
        source_name=source_name,
    )

    for col in ["precursor_mz", "precursor_charge"]:
        if col in df.columns:
            df[col] = df[col].apply(_empty_list_to_nan)

    if "precursor_mz" in df.columns and "precursor_charge" in df.columns:
        df = df.dropna(
            subset=["precursor_mz", "precursor_charge"],
            how="all",
        ).reset_index(drop=True)

    df = _ensure_charge_columns(df)
    return df


def _normalize_mgf_spectrum_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Normalize MGF spectrum_df following the previous MGF-to-HDF script.

    Key points:
    - remove MS1 if present
    - explode precursor_mz / charge list values
    - convert empty lists to NaN
    - keep only rows with usable precursor_mz and charge later
    """
    if "ms_level" in df.columns:
        ms_level = pd.to_numeric(df["ms_level"], errors="coerce")
        df = df[(ms_level != 1) | ms_level.isna()].copy()
        df["ms_level"] = pd.to_numeric(df["ms_level"], errors="coerce").fillna(2).astype("int32")
    else:
        df["ms_level"] = 2

    df = _ensure_charge_columns(df)

    df = _explode_paired_precursor_fields(
        df=df,
        first_col="precursor_mz",
        second_col="charge",
        source_name=source_name,
    )

    for col in ["precursor_mz", "charge"]:
        if col in df.columns:
            df[col] = df[col].apply(_empty_list_to_nan)

    if "charge" in df.columns:
        df["precursor_charge"] = df["charge"]

    df = _ensure_charge_columns(df)
    return df


def _normalize_mzml_spectrum_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mzML spectrum_df for prediction."""
    if "ms_level" in df.columns:
        ms_level = pd.to_numeric(df["ms_level"], errors="coerce")
        df = df[ms_level == 2].copy()
        df["ms_level"] = 2

    df = _ensure_charge_columns(df)
    return df


def _normalize_reader_spectrum_df(reader: Any, source_format: str, source_name: str) -> None:  # noqa: C901
    """
    Normalize reader.spectrum_df before saving AlphaRaw HDF.

    Responsibilities:
    - keep MS2 spectra
    - expand RAW/MGF list-like precursor fields
    - normalize charge and precursor_charge
    - drop invalid spectra
    - ensure no NA/inf values enter integer charge conversion
    """
    spectrum_df = getattr(reader, "spectrum_df", None)
    if spectrum_df is None:
        raise ValueError(f"Converted input has no spectrum dataframe: {source_name}")

    df = spectrum_df.copy()

    if df.empty:
        raise ValueError(f"Converted input contains no spectra: {source_name}")

    if source_format == "raw":
        df = _normalize_raw_spectrum_df(df, source_name)
    elif source_format == "mgf":
        df = _normalize_mgf_spectrum_df(df, source_name)
    elif source_format == "mzml":
        df = _normalize_mzml_spectrum_df(df)
    else:
        raise ValueError(f"Unsupported source format for conversion: {source_format}")

    if "ms_level" not in df.columns:
        df["ms_level"] = 2

    if df.empty:
        raise ValueError(f"No MS2 spectra were found in {source_name} after filtering.")

    df = _ensure_charge_columns(df)

    required_for_conversion = ["precursor_mz", "charge"]
    missing = [col for col in required_for_conversion if col not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction input {source_name} is missing required fields before HDF conversion: "
            f"{', '.join(missing)}"
        )

    df["precursor_mz"] = _coerce_numeric_series(df["precursor_mz"])

    for col in ["charge", "precursor_charge"]:
        if col in df.columns:
            df[col] = df[col].map(_coerce_charge_value)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["precursor_mz", "charge"]).copy()
    df = df[df["charge"] > 0].copy()

    if df.empty:
        raise ValueError(
            f"No valid MS2 spectra with precursor m/z and charge were found in {source_name}. "
            "FoxNovo requires precursor charge for prediction. "
            "For RAW input, the Thermo RAW reader may not expose charge annotations for this file; "
            "please provide mzML/MGF/HDF with charge information."
        )

    df["charge"] = df["charge"].astype("int32")

    if "precursor_charge" in df.columns:
        df["precursor_charge"] = df["precursor_charge"].fillna(df["charge"]).astype("int32")
    else:
        df["precursor_charge"] = df["charge"].astype("int32")

    for col in ["peak_start_idx", "peak_stop_idx", "spec_idx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_index_cols = ["peak_start_idx", "peak_stop_idx", "spec_idx"]
    missing_index_cols = [col for col in required_index_cols if col not in df.columns]
    if missing_index_cols:
        raise ValueError(
            f"Prediction input {source_name} is missing required spectrum index fields: "
            f"{', '.join(missing_index_cols)}"
        )

    df = df.dropna(subset=required_index_cols).copy()

    for col in required_index_cols:
        df[col] = df[col].astype("int64")

    df = df[df["peak_stop_idx"] > df["peak_start_idx"]].copy()

    if df.empty:
        raise ValueError(f"No spectra with valid peak ranges were found in {source_name}.")

    reader.spectrum_df = df.reset_index(drop=True)

    if hasattr(reader, "remove_unused_peaks"):
        reader.remove_unused_peaks()

    if hasattr(reader, "reset_spec_idxes"):
        reader.reset_spec_idxes()


def _ensure_charge_column(spectrum_df: pd.DataFrame) -> None:
    """
    Ensure a charge column exists in an HDF spectrum table when precursor_charge exists.
    This function mutates the provided DataFrame-like object in memory.
    """
    if "charge" not in spectrum_df.columns and "precursor_charge" in spectrum_df.columns:
        spectrum_df["charge"] = spectrum_df["precursor_charge"]


def _validate_prediction_schema(spectrum_df: pd.DataFrame, source_name: str) -> None:
    """Validate that a prediction HDF contains the fields and valid values needed by FoxNovo."""
    if spectrum_df is None:
        raise ValueError(f"Prediction input {source_name} has no spectrum table.")

    if len(spectrum_df) == 0:
        raise ValueError(f"Prediction input {source_name} contains no spectra.")

    required_columns = {"precursor_mz", "peak_start_idx", "peak_stop_idx", "spec_idx"}
    missing = sorted(col for col in required_columns if col not in spectrum_df.columns)
    if missing:
        raise ValueError(
            f"Prediction input {source_name} is missing required spectrum fields: "
            f"{', '.join(missing)}"
        )

    charge_col = None
    if "charge" in spectrum_df.columns:
        charge_col = "charge"
    elif "precursor_charge" in spectrum_df.columns:
        charge_col = "precursor_charge"

    if charge_col is None:
        raise ValueError(
            f"Prediction input {source_name} is missing required charge information: "
            "charge or precursor_charge"
        )

    precursor_mz = pd.to_numeric(spectrum_df["precursor_mz"], errors="coerce")
    valid_precursor_mz = precursor_mz.notna() & np.isfinite(precursor_mz)

    if int(valid_precursor_mz.sum()) == 0:
        raise ValueError(f"Prediction input {source_name} contains no valid precursor_mz values.")

    charge = pd.to_numeric(spectrum_df[charge_col], errors="coerce")
    valid_charge = charge.notna() & np.isfinite(charge) & (charge > 0)

    if int(valid_charge.sum()) == 0:
        raise ValueError(
            f"Prediction input {source_name} contains no valid precursor charge values. "
            "FoxNovo requires charge > 0 for prediction."
        )

    peak_start_idx = pd.to_numeric(spectrum_df["peak_start_idx"], errors="coerce")
    peak_stop_idx = pd.to_numeric(spectrum_df["peak_stop_idx"], errors="coerce")
    valid_peak_range = (
        peak_start_idx.notna()
        & peak_stop_idx.notna()
        & np.isfinite(peak_start_idx)
        & np.isfinite(peak_stop_idx)
        & (peak_stop_idx > peak_start_idx)
    )

    if int(valid_peak_range.sum()) == 0:
        raise ValueError(f"Prediction input {source_name} contains no valid peak ranges.")


def validate_prediction_hdf(hdf_path: str | Path) -> Path:
    """Validate an existing AlphaRaw HDF/HDF5 file for FoxNovo prediction."""
    hdf_path = Path(hdf_path)

    f = HDF_File(file_name=str(hdf_path), read_only=True)

    try:
        spectrum_df = f.ms_data.spectrum_df.values
    except Exception as exc:
        raise ValueError(
            f"Prediction HDF file does not contain ms_data.spectrum_df: {hdf_path}"
        ) from exc

    _ensure_charge_column(spectrum_df)
    _validate_prediction_schema(spectrum_df, str(hdf_path))

    return hdf_path


def convert_to_prediction_hdf(
    input_file: str | Path,
    output_hdf: str | Path,
    data_format: str,
) -> Path:
    """Convert RAW/mzML/MGF input into an AlphaRaw HDF file for prediction."""
    input_file = Path(input_file)
    output_hdf = Path(output_hdf)
    output_hdf.parent.mkdir(parents=True, exist_ok=True)

    reader = _get_alpharaw_reader(data_format)

    if reader is None:
        raise RuntimeError(
            f"AlphaRaw reader for data_format='{data_format}' is not available. "
            "For Thermo RAW, check AlphaRaw/Thermo/pythonnet installation."
        )

    with pd.option_context("mode.copy_on_write", False):
        reader.import_raw(str(input_file))

    _normalize_reader_spectrum_df(
        reader=reader,
        source_format=data_format,
        source_name=str(input_file),
    )

    spectrum_df = getattr(reader, "spectrum_df", None)
    _ensure_charge_column(spectrum_df)
    _validate_prediction_schema(spectrum_df, str(input_file))

    if output_hdf.exists():
        output_hdf.unlink()

    with pd.option_context("mode.copy_on_write", False):
        reader.save_hdf(str(output_hdf))

    validate_prediction_hdf(output_hdf)

    logger.info("Converted %s to cached HDF %s", input_file.name, output_hdf)

    return output_hdf


def collect_prediction_source_files(
    predict_path: str | Path,
    data_format: str = "hdf5",
) -> list[Path]:
    """Collect candidate prediction inputs without opening or converting them."""
    normalized_format = normalize_data_format(data_format)
    return _collect_files_from_path(Path(predict_path), normalized_format)


def prepare_single_prediction_hdf(
    input_file: str | Path,
    output_folder: str | Path,
    data_format: str = "hdf5",
) -> Path:
    """Validate or convert one prediction input into an HDF file."""
    normalized_format = normalize_data_format(data_format)
    input_file = Path(input_file)
    output_folder = Path(output_folder)

    source_format = (
        _infer_format_from_suffix(input_file) if normalized_format == "auto" else normalized_format
    )

    if source_format == "hdf5":
        return validate_prediction_hdf(input_file)

    cache_dir = output_folder / "_foxnovo_hdf_cache"
    cached_hdf = cache_dir / f"{input_file.name}.hdf"
    return convert_to_prediction_hdf(
        input_file=input_file,
        output_hdf=cached_hdf,
        data_format=source_format,
    )


def prepare_prediction_hdf_files(
    predict_path: str | Path,
    output_folder: str | Path,
    data_format: str = "hdf5",
) -> list[Path]:
    """
    Prepare prediction inputs and return HDF files ready for the existing prediction pipeline.

    Default behavior:
        data_format='hdf5'
        Only existing .hdf/.hdf5 files are processed.

    Explicit conversion:
        data_format='raw'  -> convert .raw files to cached HDF first
        data_format='mzml' -> convert .mzML/.mzml files to cached HDF first
        data_format='mgf'  -> convert .mgf files to cached HDF first
        data_format='auto' -> infer supported formats from file suffixes
    """
    source_files = collect_prediction_source_files(predict_path, data_format)

    prepared_files: list[Path] = []

    for source_file in source_files:
        prepared_files.append(
            prepare_single_prediction_hdf(source_file, output_folder, data_format)
        )

    if not prepared_files:
        raise ValueError(f"No prediction-ready HDF files were prepared from: {Path(predict_path)}")

    return prepared_files
