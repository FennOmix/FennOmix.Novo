import argparse
import csv
from pathlib import Path

from foxnovo.api import predict

EXPECTED_COLUMNS = {"spec_idx", "modified_sequence", "nar_dp_score", "nar_dp_top"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local-only FoxNovo prediction smoke test and validate CSV output."
    )
    parser.add_argument(
        "--predict-folder", required=True, help="Folder containing input HDF5 files"
    )
    parser.add_argument("--model-weights", required=True, help="Model checkpoint for prediction")
    parser.add_argument("--output-folder", required=True, help="Folder for prediction CSV output")
    return parser.parse_args()


def validate_outputs(output_folder: Path) -> None:
    if not output_folder.exists():
        raise FileNotFoundError(f"Output folder was not created: {output_folder}")

    csv_files = sorted(output_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were generated in: {output_folder}")

    first_csv = csv_files[0]
    with first_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {first_csv}")

        missing_columns = EXPECTED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV missing expected columns {sorted(missing_columns)}: {first_csv}")

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"CSV has no data rows: {first_csv}")


def main() -> int:
    args = parse_args()
    predict(
        predict_folder=args.predict_folder,
        output_folder=args.output_folder,
        model_weights=args.model_weights,
    )
    validate_outputs(Path(args.output_folder))
    print("Local smoke prediction passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
