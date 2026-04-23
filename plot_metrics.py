# CS5180 Final Project
# Max Huber

# This file plots training metrics from CSV files produced by train.py.

# IMPORTS
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PLOT FUNCTIONS


# PLOT CSV
# Reads a single metrics CSV and saves one PNG per metric group to a plots/ subdirectory.
def plot_csv(csv_path: Path):
    csv_path = Path(csv_path)
    plots_dir = csv_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Read CSV
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"SKIP (empty): {csv_path}")
        return

    fieldnames = list(rows[0].keys())
    if len(fieldnames) < 2:
        print(f"SKIP (only one column): {csv_path}")
        return

    x_col = fieldnames[0]
    y_cols = fieldnames[1:]

    # Parse values, skip rows where any needed value is missing/non-numeric
    x_vals = []
    y_data = {col: [] for col in y_cols}

    for row in rows:
        try:
            x = float(row[x_col])
        except (ValueError, TypeError):
            continue
        # Try parsing all y columns; skip row if x fails
        parsed = {}
        for col in y_cols:
            try:
                parsed[col] = float(row[col])
            except (ValueError, TypeError):
                parsed[col] = float("nan")
        x_vals.append(x)
        for col in y_cols:
            y_data[col].append(parsed[col])

    if not x_vals:
        print(f"  SKIP (no numeric data): {csv_path}")
        return

    x_arr = np.array(x_vals)

    phase_name = csv_path.parent.parent.name  # e.g. "baseline_a2c_vs_heuristic"
    model_name = csv_path.parent.name  # e.g. "a2c"
    title_prefix = f"{phase_name}/{model_name}"

    # Partition columns into groups
    win_rate_cols = [c for c in y_cols if c.startswith("win_rate")]
    loss_cols = [c for c in y_cols if c.endswith("loss")]
    other_cols = [c for c in y_cols if c not in win_rate_cols and c not in loss_cols]

    COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "sienna"]

    saved = []

    def save_grouped(cols, filename, group_title, ylabel):
        if not cols:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, col in enumerate(cols):
            ax.plot(
                x_arr,
                np.array(y_data[col]),
                linewidth=1.2,
                color=COLORS[i % len(COLORS)],
                label=col,
            )
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix} - {group_title}")
        ax.grid(True, alpha=0.3)
        if len(cols) > 1:
            ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = plots_dir / filename
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(filename)

    def save_single(col):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x_arr, np.array(y_data[col]), linewidth=1.2, color="steelblue")
        ax.set_xlabel(x_col)
        ax.set_ylabel(col)
        ax.set_title(f"{title_prefix} - {col}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe_col = col.replace("/", "_").replace(" ", "_")
        out_path = plots_dir / f"{safe_col}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved.append(out_path.name)

    save_grouped(win_rate_cols, "win_rates.png", "win rates", "win rate")
    save_grouped(loss_cols, "losses.png", "losses", "loss")
    for col in other_cols:
        save_single(col)

    print(f"{csv_path}  ->  {plots_dir}/  [{', '.join(saved)}]")


# FIND CSVS
# Recursively finds all CSV files under a root directory.
def find_csvs(root: Path):
    return sorted(root.rglob("*.csv"))


# MAIN FUNCTION


# MAIN
# Parses command-line paths and plots all discovered CSV metrics files.
def main():
    parser = argparse.ArgumentParser(description="Plot metrics from CSV files")
    parser.add_argument(
        "paths",
        nargs="+",
        help="CSV file(s) or directory/directories to search recursively",
    )
    args = parser.parse_args()

    targets = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            found = find_csvs(path)
            if not found:
                print(f"No CSV files found under {path}")
            targets.extend(found)
        elif path.is_file() and path.suffix == ".csv":
            targets.append(path)
        else:
            print(f"WARNING: {p} is not a CSV file or directory")

    if not targets:
        print("Nothing to plot.")
        sys.exit(1)

    print(f"Plotting {len(targets)} CSV file(s)...\n")
    for csv_path in targets:
        plot_csv(csv_path)
    print("\nDone.")


# MAIN
if __name__ == "__main__":
    main()
