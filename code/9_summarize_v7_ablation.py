#!/usr/bin/env python3
"""
Summarize V7 ablation runs as a paper-ready CSV and PNG.

Example
-------
python code/9_summarize_v7_ablation.py \
  --run "Full V7=work_dirs/ablation_v7_full" \
  --run "w/o Salient selection=work_dirs/ablation_v7_no_salient" \
  --run "w/o UMP tokens=work_dirs/ablation_v7_no_ump" \
  --run "w/o Global context=work_dirs/ablation_v7_no_global" \
  --run "w/o Edge type=work_dirs/ablation_v7_no_edge_type" \
  --out_dir work_dirs/analysis/v7_ablation
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_run(value):
    if "=" not in value:
        raise ValueError("--run must use NAME=/path/to/experiment_or_summary.json")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if path.is_dir():
        path = path / "results" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return name, path


def kappa_from_cm(path):
    if not path.exists():
        return np.nan
    cm = np.loadtxt(path, delimiter=",").astype(np.float64)
    total = cm.sum()
    if total <= 0:
        return 0.0
    observed = np.diag(cm).sum() / total
    expected = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / (total * total)
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 0.0


def load_rows(run_values):
    rows = []
    for order, value in enumerate(run_values):
        name, summary_path = parse_run(value)
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        cm_path = summary_path.parent / "confusion_matrix.csv"
        rows.append({
            "Order": order,
            "Configuration": name,
            "Ablation": summary.get("ablation", "full" if order == 0 else "unknown"),
            "OA": float(summary["test_oa"]),
            "Macro-F1": float(summary["test_macro_f1"]),
            "Weighted-F1": float(summary["test_weighted_f1"]),
            "Kappa": float(kappa_from_cm(cm_path)),
            "Parameters": int(summary.get("params", 0)),
        })
    frame = pd.DataFrame(rows).sort_values("Order").reset_index(drop=True)
    reference = frame.iloc[0]
    for metric in ["OA", "Macro-F1", "Weighted-F1", "Kappa"]:
        frame[f"Delta {metric}"] = frame[metric] - float(reference[metric])
    return frame


def plot_table(frame, out_png, dpi):
    columns = [
        "Configuration", "OA", "Delta OA", "Macro-F1", "Delta Macro-F1",
        "Weighted-F1", "Kappa",
    ]
    best = {metric: frame[metric].max() for metric in ["OA", "Macro-F1", "Weighted-F1", "Kappa"]}
    cell_text = []
    for _, row in frame.iterrows():
        cell_text.append([
            row["Configuration"],
            f"{row['OA']:.4f}",
            f"{row['Delta OA']:+.4f}",
            f"{row['Macro-F1']:.4f}",
            f"{row['Delta Macro-F1']:+.4f}",
            f"{row['Weighted-F1']:.4f}",
            f"{row['Kappa']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(14.5, 4.4))
    ax.axis("off")
    ax.text(
        0.02, 0.94, "Ablation study of BGF-LCZNet (V7)",
        transform=ax.transAxes, fontsize=15, fontweight="bold", va="top",
    )
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.20, 0.96, 0.58],
        colWidths=[0.28, 0.10, 0.11, 0.12, 0.14, 0.14, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    n_rows = len(frame)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        elif row_index == n_rows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)
    metric_columns = {"OA": 1, "Macro-F1": 3, "Weighted-F1": 5, "Kappa": 6}
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        for metric, col_index in metric_columns.items():
            if np.isclose(float(row[metric]), best[metric]):
                table[(row_index, col_index)].set_text_props(fontweight="bold")
    ax.plot([0.02, 0.98], [0.80, 0.80], color="black", linewidth=1.2, transform=ax.transAxes)
    ax.text(
        0.5, 0.10,
        "Delta values are measured relative to the complete V7 configuration.",
        transform=ax.transAxes, ha="center", fontsize=9, color="#444444",
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/experiment_or_summary.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_rows(args.run)
    csv_path = out_dir / "v7_ablation_summary.csv"
    png_path = out_dir / "v7_ablation_summary.png"
    frame.drop(columns=["Order"]).to_csv(csv_path, index=False, encoding="utf-8-sig", float_format="%.6f")
    plot_table(frame, png_path, args.dpi)
    print("saved:", csv_path)
    print("saved:", png_path)


if __name__ == "__main__":
    main()
