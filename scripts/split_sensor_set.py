#!/usr/bin/env python3

"""
Split sensor_set.csv into sensor1.csv ... sensor9.csv by interleaving rows.

Rules (1-indexed rows, excluding header):
- Rows 1, 10, 19, ... -> sensor1.csv (rows 1..10 inside sensor1)
- Rows 2, 11, 20, ... -> sensor2.csv
- ...
- Rows 9, 18, 27, ... -> sensor9.csv

Defaults target the repository layout on CRC cluster.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split sensor_set.csv into sensor1..sensor9 per 9-row interleave rule."
        )
    )
    default_input = (
        "/users/zchen27/SensorReconstruction/backup/Sensor/robot_bending/"
        "sensor_set.csv"
    )
    default_output_dir = "/users/zchen27/SensorReconstruction/backup/Sensor/robot_bending"

    parser.add_argument(
        "-i",
        "--input",
        dest="input_csv",
        default=default_input,
        help="Path to input sensor_set.csv",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=default_output_dir,
        help="Directory to write sensor1..sensor9 CSV files",
    )
    parser.add_argument(
        "-n",
        "--num-groups",
        dest="num_groups",
        type=int,
        default=9,
        help="Number of output groups (default: 9)",
    )

    return parser.parse_args()


def split_sensor_set(input_csv: str, output_dir: str, num_groups: int = 9) -> None:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    header: List[str]
    rows: List[List[str]]
    try:
        with open(input_csv, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError("Input CSV is empty") from exc
            # Filter out empty rows (e.g., trailing blank lines)
            rows = [
                r
                for r in reader
                if r and any((c.strip() for c in r))
            ]
    except Exception as exc:  # noqa: BLE001 (broad for robust CLI)
        raise RuntimeError(f"Failed reading CSV: {input_csv}") from exc

    grouped_rows: Dict[int, List[List[str]]] = {g: [] for g in range(1, num_groups + 1)}

    # 1-indexed within the data rows (excluding header)
    for data_index, row in enumerate(rows, start=1):
        group_index = ((data_index - 1) % num_groups) + 1
        grouped_rows[group_index].append(row)

    # Write output files sensor1.csv .. sensor{num_groups}.csv
    for group in range(1, num_groups + 1):
        out_path = os.path.join(output_dir, f"sensor{group}.csv")
        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(grouped_rows[group])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed writing output CSV: {out_path}") from exc

    total_written = sum(len(v) for v in grouped_rows.values())
    print(
        f"Split complete. Input rows: {len(rows)}, "
        f"outputs: {num_groups}, written rows (sum): {total_written}"
    )
    for group in range(1, num_groups + 1):
        print(
            f"  sensor{group}.csv -> {len(grouped_rows[group])} rows | "
            f"{os.path.join(output_dir, f'sensor{group}.csv')}"
        )


def main() -> int:
    args = parse_args()
    try:
        split_sensor_set(args.input_csv, args.output_dir, args.num_groups)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

