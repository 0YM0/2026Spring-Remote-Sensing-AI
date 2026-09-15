#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute OA, Macro-F1, Weighted-F1, Kappa, PA/UA/F1 from a confusion matrix CSV.

Assumption
----------
Rows = reference LCZ
Columns = predicted / classified LCZ
This matches sklearn.metrics.confusion_matrix(y_true, y_pred), which is used
by the LCZ training scripts:
  PA / recall = diagonal / row sum
  UA / precision = diagonal / column sum

Example
-------
# A. RS-only 8채널 모델
python code/3_accuracy_from_confusion_matrix.py \
  --cm_csv /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/results/confusion_matrix.csv \
  --class_codes 1 2 3 4 5 6 8 101 102 104 107 \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_baseline_accuracy \
  --title "RS-CNN (Baseline)"

# B. RS + Rasterized Building 10채널 모델
python code/3_accuracy_from_confusion_matrix.py \
  --cm_csv /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_raster_building/results/confusion_matrix.csv \
  --class_codes 1 2 3 4 5 6 8 101 102 104 107 \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_raster_building_accuracy \
  --title "RasterConcat-CNN"

# C. RS + Building dual visual encoder
python code/3_accuracy_from_confusion_matrix.py \
  --cm_csv /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_building_visual_encoder/results/confusion_matrix.csv \
  --class_codes 1 2 3 4 5 6 8 101 102 104 107 \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_building_visual_encoder_accuracy \
  --title "BuildingVisual-CNN"

# D. Building graph fusion model
python code/3_accuracy_from_confusion_matrix.py \
  --cm_csv /mnt/disk1/workspace_jym/LCZ/work_dirs/graph_v7_node_edge_scale_interaction_oa/results/confusion_matrix.csv \
  --class_codes 1 2 3 4 5 6 8 101 102 104 107 \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/v7_accuracy \
  --title "BGF-LCZNet (Ours)"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


LCZ_NAME = {
    1: "LCZ1 Compact high-rise",
    2: "LCZ2 Compact mid-rise",
    3: "LCZ3 Compact low-rise",
    4: "LCZ4 Open high-rise",
    5: "LCZ5 Open mid-rise",
    6: "LCZ6 Open low-rise",
    7: "LCZ7 Lightweight low-rise",
    8: "LCZ8 Large low-rise",
    9: "LCZ9 Sparsely built",
    10: "LCZ10 Heavy industry",
    101: "LCZA Dense trees",
    102: "LCZB Scattered trees",
    103: "LCZC Bush/scrub",
    104: "LCZD Low plants",
    105: "LCZE Bare rock/paved",
    106: "LCZF Bare soil/sand",
    107: "LCZG Water",
}

LCZ_SHORT_CODE = {
    101: "A",
    102: "B",
    103: "C",
    104: "D",
    105: "E",
    106: "F",
    107: "G",
}


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def class_tick_label(code):
    return LCZ_SHORT_CODE.get(int(code), str(int(code)))


def add_cell_grid(ax, rows, cols, color="#777777", linewidth=0.55):
    for row in range(rows + 1):
        ax.axhline(row - 0.5, color=color, linewidth=linewidth, zorder=5)
    for col in range(cols + 1):
        ax.axvline(col - 0.5, color=color, linewidth=linewidth, zorder=5)


def plot_accuracy_matrix(cm,
                         class_codes,
                         oa,
                         macro_f1,
                         weighted_f1,
                         kappa,
                         pa_recall,
                         ua_precision,
                         out_png,
                         title,
                         dpi):
    """
    Plot in paper-style orientation: rows=classified, columns=reference.

    Input cm follows sklearn orientation (rows=reference, columns=classified),
    so it is transposed only for visualization.
    """
    display_cm = cm.T
    n_classes = len(class_codes)
    labels = [class_tick_label(code) for code in class_codes]

    fig = plt.figure(figsize=(max(10.5, n_classes * 0.78), max(8.0, n_classes * 0.68)))
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 0.095, 0.03],
        height_ratios=[1.0, 0.12, 0.14],
        wspace=0.06,
        hspace=0.06,
    )
    ax_cm = fig.add_subplot(grid[0, 0])
    ax_ua = fig.add_subplot(grid[0, 1])
    ax_pa = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[2, 0:2])

    offdiag_cmap = LinearSegmentedColormap.from_list(
        "offdiag",
        ["#ffffff", "#f7e5df", "#e9bfb2"],
    )
    diagonal_cmap = LinearSegmentedColormap.from_list(
        "diagonal",
        ["#eeeeee", "#b9b9b9", "#626262"],
    )

    offdiag = display_cm.copy()
    np.fill_diagonal(offdiag, np.nan)
    diagonal = np.full_like(display_cm, np.nan, dtype=np.float64)
    np.fill_diagonal(diagonal, np.diag(display_cm))

    offdiag_max = float(np.nanmax(offdiag)) if np.any(np.isfinite(offdiag)) else 1.0
    diagonal_max = float(np.nanmax(diagonal)) if np.any(np.isfinite(diagonal)) else 1.0
    ax_cm.imshow(
        offdiag,
        cmap=offdiag_cmap,
        vmin=0,
        vmax=max(offdiag_max, 1.0),
        interpolation="nearest",
        aspect="auto",
    )
    ax_cm.imshow(
        diagonal,
        cmap=diagonal_cmap,
        vmin=0,
        vmax=max(diagonal_max, 1.0),
        interpolation="nearest",
        aspect="auto",
    )
    add_cell_grid(ax_cm, n_classes, n_classes)

    for row in range(n_classes):
        for col in range(n_classes):
            value = int(display_cm[row, col])
            if value == 0:
                continue
            is_dark_diagonal = row == col and diagonal_max > 0 and value / diagonal_max >= 0.55
            ax_cm.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if is_dark_diagonal else "#222222",
                fontweight="bold" if row == col else "normal",
            )

    ax_cm.set_xticks(np.arange(n_classes), labels=[""] * n_classes)
    ax_cm.set_yticks(np.arange(n_classes), labels=labels)
    ax_cm.tick_params(axis="both", length=0, labelsize=10)
    ax_cm.set_ylabel("Classified LCZ", fontsize=11, labelpad=10)
    ax_cm.set_title(f"{title}  (OA: {oa * 100:.2f}%)", fontsize=15, fontweight="bold", pad=14)

    ua_values = ua_precision.reshape(-1, 1) * 100
    ax_ua.imshow(ua_values, cmap="Greys", vmin=0, vmax=100, interpolation="nearest", aspect="auto")
    add_cell_grid(ax_ua, n_classes, 1)
    for row, value in enumerate(ua_values[:, 0]):
        ax_ua.text(
            0,
            row,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=7.5,
            color="white" if value >= 58 else "#222222",
            fontweight="bold",
        )
    ax_ua.set_xticks([0], labels=["UA%"])
    ax_ua.xaxis.tick_top()
    ax_ua.set_yticks([])
    ax_ua.tick_params(length=0, labelsize=9)

    pa_values = pa_recall.reshape(1, -1) * 100
    ax_pa.imshow(pa_values, cmap="Greys", vmin=0, vmax=100, interpolation="nearest", aspect="auto")
    add_cell_grid(ax_pa, 1, n_classes)
    for col, value in enumerate(pa_values[0]):
        ax_pa.text(
            col,
            0,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value >= 58 else "#222222",
            fontweight="bold",
        )
    ax_pa.set_xticks(np.arange(n_classes), labels=labels)
    ax_pa.set_yticks([0], labels=["PA%"])
    ax_pa.tick_params(length=0, labelsize=9)
    ax_pa.set_xlabel("Reference LCZ", fontsize=11, labelpad=9)

    ax_summary.axis("off")
    summary_text = (
        f"Macro-F1  {macro_f1 * 100:.2f}%     "
        f"Weighted-F1  {weighted_f1 * 100:.2f}%     "
        f"Kappa  {kappa:.4f}"
    )
    ax_summary.add_patch(
        Rectangle(
            (0.08, 0.18),
            0.84,
            0.62,
            transform=ax_summary.transAxes,
            facecolor="#f4f4f4",
            edgecolor="#777777",
            linewidth=0.8,
        )
    )
    ax_summary.text(
        0.5,
        0.49,
        summary_text,
        transform=ax_summary.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#333333",
    )

    fig.text(
        0.985,
        0.015,
        "PA = producer's accuracy (recall)   |   UA = user's accuracy (precision)",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cm_csv", required=True)
    p.add_argument("--class_codes", type=int, nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--title", default="LCZ classification accuracy")
    p.add_argument("--png_name", default="confusion_matrix_accuracy.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no_png", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = np.loadtxt(args.cm_csv, delimiter=",")
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square. Got {cm.shape}")
    if cm.shape[0] != len(args.class_codes):
        raise ValueError(f"CM size {cm.shape[0]} != number of class codes {len(args.class_codes)}")

    total = cm.sum()
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)   # reference total
    col_sum = cm.sum(axis=0)   # predicted/classified total

    oa = diag.sum() / total if total > 0 else 0.0

    # sklearn confusion_matrix: rows=reference, columns=predicted.
    pa_recall = safe_div(diag, row_sum)
    ua_precision = safe_div(diag, col_sum)
    f1 = safe_div(2 * ua_precision * pa_recall, ua_precision + pa_recall)

    support = row_sum
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum() if support.sum() > 0 else 0.0

    expected = (row_sum * col_sum).sum() / (total * total) if total > 0 else 0.0
    kappa = (oa - expected) / (1.0 - expected) if (1.0 - expected) != 0 else 0.0

    rows = []
    for i, code in enumerate(args.class_codes):
        rows.append({
            "class_code": code,
            "class_name": LCZ_NAME.get(code, str(code)),
            "support_reference": int(row_sum[i]),
            "predicted_count": int(col_sum[i]),
            "PA_recall": pa_recall[i],
            "UA_precision": ua_precision[i],
            "F1": f1[i],
        })

    class_df = pd.DataFrame(rows)
    class_df.to_csv(out_dir / "classwise_accuracy.csv", index=False, encoding="utf-8-sig")

    summary = {
        "OA": float(oa),
        "Macro_F1": float(macro_f1),
        "Weighted_F1": float(weighted_f1),
        "Kappa": float(kappa),
        "total_samples": int(total),
        "note": "Rows=reference, columns=predicted/classified. PA=recall, UA=precision.",
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary_accuracy.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "summary_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    png_path = out_dir / args.png_name
    if not args.no_png:
        plot_accuracy_matrix(
            cm=cm,
            class_codes=args.class_codes,
            oa=oa,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            kappa=kappa,
            pa_recall=pa_recall,
            ua_precision=ua_precision,
            out_png=png_path,
            title=args.title,
            dpi=args.dpi,
        )

    print("========== Accuracy summary ==========")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("saved:", out_dir / "summary_accuracy.csv")
    print("saved:", out_dir / "classwise_accuracy.csv")
    if not args.no_png:
        print("saved:", png_path)


if __name__ == "__main__":
    main()
