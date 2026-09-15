#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a paper-style accuracy table for V7 ablation results.

Inputs are summary.json and confusion_matrix.csv from each experiment.
The table includes:
  Method | OA | OA_urb | OA_u | Macro-F1 | Weighted-F1 | Kappa
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_named_paths(values, argument_name):
    parsed = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{argument_name} must use NAME=/path/to/file_or_dir")
        name, path = value.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def compute_metrics(cm_path, class_codes):
    cm = np.loadtxt(cm_path, delimiter=",", dtype=np.float64)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square: {cm_path}")
    if cm.shape[0] != len(class_codes):
        raise ValueError(f"CM size mismatch for {cm_path}: {cm.shape[0]} != {len(class_codes)}")

    total = cm.sum()
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    oa = float(diag.sum() / total) if total > 0 else 0.0
    pa = safe_div(diag, row_sum)
    ua = safe_div(diag, col_sum)
    f1 = safe_div(2 * pa * ua, pa + ua)
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float((f1 * row_sum).sum() / row_sum.sum()) if row_sum.sum() > 0 else 0.0
    expected = (row_sum * col_sum).sum() / (total * total) if total > 0 else 0.0
    kappa = float((oa - expected) / (1.0 - expected)) if (1.0 - expected) != 0 else 0.0

    class_codes = np.asarray(class_codes)
    urban_mask = class_codes <= 10
    urban_ref_total = row_sum[urban_mask].sum()
    oaurb = float(diag[urban_mask].sum() / urban_ref_total) if urban_ref_total > 0 else 0.0
    oau = float(cm[np.ix_(urban_mask, urban_mask)].sum() / urban_ref_total) if urban_ref_total > 0 else 0.0

    return {
        "OA": oa,
        "OAurb": oaurb,
        "OAu": oau,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Kappa": kappa,
    }


def load_summary(summary_json):
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)
    return json.loads(summary_json.read_text())


def make_metric_percent(v):
    return f"{v * 100:.2f}"


def plot_table(df, out_png, caption, dpi):
    metric_columns = ["OA", "OAurb", "OAu", "Macro-F1", "Weighted-F1", "Kappa"]
    best_values = {column: df[column].max() for column in metric_columns}

    display_rows = []
    for _, row in df.iterrows():
        display_rows.append([
            row["Method"],
            make_metric_percent(row["OA"]),
            make_metric_percent(row["OAurb"]),
            make_metric_percent(row["OAu"]),
            make_metric_percent(row["Macro-F1"]),
            make_metric_percent(row["Weighted-F1"]),
            f"{row['Kappa']:.4f}",
        ])

    column_labels = [
        "Method",
        "OA(%)",
        "OA$_{urb}$(%)",
        "OA$_u$(%)",
        "Macro-F1(%)",
        "Weighted-F1(%)",
        "Kappa",
    ]

    fig_height = 3.2 + 0.42 * max(len(display_rows) - 5, 0)
    fig, ax = plt.subplots(figsize=(16.0, fig_height))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.axis("off")
    ax.text(0.01, 0.985, caption, ha="left", va="top", fontsize=13.5, fontweight="bold")

    table = ax.table(
        cellText=display_rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.15, 0.98, 0.74],
        colWidths=[0.20, 0.12, 0.14, 0.13, 0.14, 0.15, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 1.48)

    n_rows = len(display_rows)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.0)
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)
        elif row_index == n_rows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)

    for row_index, (_, row) in enumerate(df.iterrows(), start=1):
        for metric_index, column in enumerate(metric_columns, start=1):
            table_col = metric_index
            if np.isclose(float(row[column]), best_values[column]):
                table[(row_index, table_col)].set_text_props(fontweight="bold")

    ax.plot([0.015, 0.985], [0.89, 0.89], color="black", linewidth=1.2, transform=ax.transAxes)
    ax.text(
        0.5,
        0.065,
        "The highest value for each accuracy metric is shown in bold.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/experiment_dir_or_summary.json")
    parser.add_argument("--class_codes", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--table_caption", default="Accuracy assessment results for the LCZ ablation models.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    run_items = parse_named_paths(args.run, "--run")
    rows = []
    for name, path in run_items:
        if path.is_dir():
            summary_path = path / "results" / "summary.json"
            cm_path = path / "results" / "confusion_matrix.csv"
        else:
            summary_path = path
            cm_path = path.parent / "confusion_matrix.csv"

        summary = load_summary(summary_path)
        metrics = compute_metrics(cm_path, args.class_codes)
        metrics["Method"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)[[
        "Method",
        "OA",
        "OAurb",
        "OAu",
        "Macro-F1",
        "Weighted-F1",
        "Kappa",
    ]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "v7_ablation_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    plot_table(df, out_dir / "v7_ablation_summary.png", args.table_caption, args.dpi)
    print("saved:", out_dir / "v7_ablation_summary.csv")
    print("saved:", out_dir / "v7_ablation_summary.png")


if __name__ == "__main__":
    main()
