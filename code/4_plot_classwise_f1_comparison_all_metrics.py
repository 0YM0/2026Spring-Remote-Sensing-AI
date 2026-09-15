#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot paper-ready classwise F1-score comparison and Table-4-style LCZ accuracy table.

This version creates an overall table with:
  Method | OA | OAurb | OAu | Macro-F1 | Weighted-F1 | Kappa

Inputs
------
1) classwise_accuracy.csv from 3_accuracy_from_confusion_matrix.py
2) summary_accuracy.csv from 3_accuracy_from_confusion_matrix.py
3) confusion_matrix.csv from each model result folder

Important assumption
--------------------
Confusion matrix orientation follows sklearn:
  rows    = reference / true LCZ
  columns = predicted / classified LCZ

Definitions
-----------
OA:
  Correctly classified pixels among all LCZ reference pixels.

OAurb:
  Correctly classified pixels among urban-type LCZ reference pixels only.
  Urban classes are LCZ1-10.

OAu:
  Urban/natural separation accuracy for urban reference pixels.
  A reference urban pixel is counted correct if it is predicted as any urban LCZ class.

Macro-F1:
  Mean of class-wise F1-scores.

Weighted-F1:
  Class-wise F1 weighted by reference support.

Kappa:
  Cohen's kappa from the full multiclass confusion matrix.

Example
-------
python code/4_plot_classwise_f1_comparison_all_metrics.py \
  --model "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_baseline_accuracy/classwise_accuracy.csv" \
  --model "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_raster_building_accuracy/classwise_accuracy.csv" \
  --model "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_building_visual_encoder_accuracy/classwise_accuracy.csv" \
  --model "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/v7_accuracy/classwise_accuracy.csv" \
  --summary "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_baseline_accuracy/summary_accuracy.csv" \
  --summary "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_raster_building_accuracy/summary_accuracy.csv" \
  --summary "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/cnn_rs_building_visual_encoder_accuracy/summary_accuracy.csv" \
  --summary "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/v7_accuracy/summary_accuracy.csv" \
  --cm "RS-CNN (Baseline)=/mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/results/confusion_matrix.csv" \
  --cm "RasterConcat-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_raster_building/results/confusion_matrix.csv" \
  --cm "BuildingVisual-CNN=/mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_building_visual_encoder/results/confusion_matrix.csv" \
  --cm "BGF-LCZNet (Ours)=/mnt/disk1/workspace_jym/LCZ/work_dirs/graph_v7_node_edge_scale_interaction_oa/results/confusion_matrix.csv" \
  --class_codes 1 2 3 4 5 6 8 101 102 104 107 \
  --out_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/analysis/model_f1_comparison_all_metrics \
  --title "Classwise LCZ classification performance" \
  --table_caption "Accuracy assessment results for the LCZ classification models."
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LCZ_SHORT_CODE = {
    101: "A",
    102: "B",
    103: "C",
    104: "D",
    105: "E",
    106: "F",
    107: "G",
}

DEFAULT_COLORS = ["#BDBDBD", "#A9B8C6", "#7F9DB9", "#D62728", "#54A24B", "#ECA82C"]

MODEL_METADATA = {
    "RS-CNN (Baseline)": ("RS", "Single", "-"),
    "RasterConcat-CNN": ("RS + Building", "Single", "Raster"),
    "BuildingVisual-CNN": ("RS + Building", "Dual", "Raster"),
    "BGF-LCZNet (Ours)": ("RS + Building", "Dual", "Graph"),
}


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def parse_named_paths(values, argument_name):
    parsed = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{argument_name} must use NAME=/path/to/file")
        name, path = value.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def class_label(code):
    code = int(code)
    return f"LCZ{LCZ_SHORT_CODE.get(code, code)}"


def model_color(model_name, model_index):
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")
    if "bgf" in normalized or "ours" in normalized or "proposed" in normalized:
        return "#D62728"
    if "baseline" in normalized or "rs_cnn" in normalized:
        return "#C7C7C7"
    if "rasterconcat" in normalized or "raster" in normalized:
        return "#A9B8C6"
    if "buildingvisual" in normalized or "visual" in normalized:
        return "#7F9DB9"
    return DEFAULT_COLORS[model_index % len(DEFAULT_COLORS)]


def load_classwise(model_items):
    frames = []
    expected_codes = None

    for model_name, path in model_items:
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path)
        required = {"class_code", "F1"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain columns {sorted(required)}")

        frame = frame.copy()
        frame["class_code"] = frame["class_code"].astype(int)
        codes = frame["class_code"].tolist()

        if expected_codes is None:
            expected_codes = codes
        elif codes != expected_codes:
            raise ValueError(f"Class order mismatch in {path}: {codes} != {expected_codes}")

        frame["model"] = model_name
        frames.append(frame)

    return expected_codes, pd.concat(frames, ignore_index=True)


def load_summaries(summary_items):
    summaries = {}
    for model_name, path in summary_items:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"Expected one summary row in {path}")
        summaries[model_name] = frame.iloc[0].to_dict()
    return summaries


def compute_metrics_from_cm(cm_path, class_codes):
    cm = np.loadtxt(cm_path, delimiter=",")
    cm = np.asarray(cm, dtype=np.float64)

    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square. Got {cm.shape}: {cm_path}")
    if cm.shape[0] != len(class_codes):
        raise ValueError(
            f"CM size {cm.shape[0]} != number of class codes {len(class_codes)} for {cm_path}"
        )

    total = cm.sum()
    diag = np.diag(cm)

    # sklearn confusion_matrix convention:
    # rows = reference/true, columns = predicted/classified.
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    oa = float(diag.sum() / total) if total > 0 else 0.0

    pa_recall = safe_div(diag, row_sum)
    ua_precision = safe_div(diag, col_sum)
    f1 = safe_div(2 * pa_recall * ua_precision, pa_recall + ua_precision)

    support = row_sum
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() > 0 else 0.0

    expected = (row_sum * col_sum).sum() / (total * total) if total > 0 else 0.0
    kappa = float((oa - expected) / (1.0 - expected)) if (1.0 - expected) != 0 else 0.0

    class_codes = np.asarray(class_codes)
    urban_mask = class_codes <= 10
    natural_mask = class_codes >= 101

    # OAurb: exact class accuracy within reference urban LCZ classes.
    urban_ref_total = row_sum[urban_mask].sum()
    oaurb = float(diag[urban_mask].sum() / urban_ref_total) if urban_ref_total > 0 else 0.0

    # OAnat: exact class accuracy within reference natural LCZ classes. Saved for CSV only.
    natural_ref_total = row_sum[natural_mask].sum()
    oanat = float(diag[natural_mask].sum() / natural_ref_total) if natural_ref_total > 0 else 0.0

    # OAu: urban-vs-natural separation accuracy for reference urban pixels.
    # If reference is urban, prediction is counted correct when predicted as any urban class.
    urban_to_urban = cm[np.ix_(urban_mask, urban_mask)].sum()
    oau = float(urban_to_urban / urban_ref_total) if urban_ref_total > 0 else 0.0

    return {
        "OA": oa,
        "OAurb": oaurb,
        "OAu": oau,
        "OAnat": oanat,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Kappa": kappa,
        "total_samples": int(total),
    }


def merge_metrics(model_names, summaries, cm_items, class_codes):
    cm_paths = dict(cm_items)
    rows = []

    for model_name in model_names:
        if model_name not in cm_paths:
            raise ValueError(f"Missing --cm for model: {model_name}")

        metrics = compute_metrics_from_cm(cm_paths[model_name], class_codes)

        # Use summary values for Macro/Weighted/Kappa if provided, to match previous outputs exactly.
        if model_name in summaries:
            summary = summaries[model_name]
            if "Macro_F1" in summary:
                metrics["Macro-F1"] = float(summary["Macro_F1"])
            if "Weighted_F1" in summary:
                metrics["Weighted-F1"] = float(summary["Weighted_F1"])
            if "Kappa" in summary:
                metrics["Kappa"] = float(summary["Kappa"])
            if "OA" in summary:
                metrics["OA"] = float(summary["OA"])

        metrics["Method"] = model_name
        input_type, encoder_type, building_representation = MODEL_METADATA.get(
            model_name, ("-", "-", "-")
        )
        metrics["Input"] = input_type
        metrics["Encoder type"] = encoder_type
        metrics["Building representation"] = building_representation
        rows.append(metrics)

    table = pd.DataFrame(rows)
    return table[[
        "Method",
        "Encoder type",
        "Input",
        "Building representation",
        "OA",
        "OAurb",
        "OAu",
        "Macro-F1",
        "Weighted-F1",
        "Kappa",
        "OAnat",
        "total_samples",
    ]]


def plot_classwise_f1(class_codes, comparison, model_names, out_png, title, dpi):
    n_classes = len(class_codes)
    n_models = len(model_names)
    x = np.arange(n_classes)

    group_width = 0.78
    bar_width = group_width / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 1.05), 6.3))

    model_values = []
    bars_by_model = []
    for model_index, model_name in enumerate(model_names):
        values = comparison.loc[comparison["model"] == model_name, "F1"].to_numpy(dtype=float)
        model_values.append(values)

        bars = ax.bar(
            x + offsets[model_index],
            values,
            width=bar_width * 0.92,
            color=model_color(model_name, model_index),
            edgecolor="white",
            linewidth=0.6,
            label=model_name,
            zorder=3,
        )
        bars_by_model.append(bars)

    value_matrix = np.stack(model_values, axis=0)
    best_indices = np.argmax(value_matrix, axis=0)
    for class_index, model_index in enumerate(best_indices):
        bar = bars_by_model[model_index][class_index]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.025, 1.06),
            "*",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xticks(x, [class_label(code) for code in class_codes], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=min(n_models, 4),
        frameon=False,
        fontsize=10,
    )

    fig.text(
        0.99,
        0.012,
        "* Highest F1-score for each LCZ class",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.subplots_adjust(bottom=0.20, left=0.08, right=0.98, top=0.90)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_metric_percent(value):
    return f"{value * 100:.2f}"


def plot_accuracy_table(table_df, out_png, caption, dpi):
    metric_columns = ["OA", "OAurb", "OAu", "Macro-F1", "Weighted-F1", "Kappa"]
    best_values = {column: table_df[column].max() for column in metric_columns}

    display_rows = []
    for _, row in table_df.iterrows():
        display_rows.append([
            row["Method"],
            row["Encoder type"],
            row["Input"],
            row["Building representation"],
            make_metric_percent(row["OA"]),
            make_metric_percent(row["OAurb"]),
            make_metric_percent(row["OAu"]),
            make_metric_percent(row["Macro-F1"]),
            make_metric_percent(row["Weighted-F1"]),
            f"{row['Kappa']:.4f}",
        ])

    column_labels = [
        "Method",
        "Encoder type",
        "Input",
        "Building representation",
        "OA(%)",
        "OA$_{urb}$(%)",
        "OA$_u$(%)",
        "Macro-F1(%)",
        "Weighted-F1(%)",
        "Kappa",
    ]

    fig_width = 19.5
    fig_height = 3.3 + 0.35 * max(len(display_rows) - 4, 0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.94,
        caption,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        wrap=True,
    )

    table = ax.table(
        cellText=display_rows,
        colLabels=column_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.03, 0.20, 0.94, 0.58],
        colWidths=[0.18, 0.12, 0.12, 0.19, 0.08, 0.10, 0.09, 0.11, 0.11, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1, 1.30)

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

    # Bold best metric values.
    for row_index, (_, row) in enumerate(table_df.iterrows(), start=1):
        for metric_index, column in enumerate(metric_columns, start=1):
            table_col = metric_index + 3
            if np.isclose(float(row[column]), best_values[column]):
                table[(row_index, table_col)].set_text_props(fontweight="bold")

    # Top rule, like paper table.
    ax.plot([0.04, 0.96], [0.81, 0.81], color="black", linewidth=1.2, transform=ax.transAxes)

    ax.text(
        0.5,
        0.09,
        "The highest value for each accuracy metric is shown in bold.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#444444",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="NAME=/path/to/classwise_accuracy.csv")
    parser.add_argument("--summary", action="append", default=None, help="NAME=/path/to/summary_accuracy.csv")
    parser.add_argument("--cm", action="append", required=True, help="NAME=/path/to/confusion_matrix.csv")
    parser.add_argument("--class_codes", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--title", default="Classwise LCZ classification performance")
    parser.add_argument(
        "--table_caption",
        default="Accuracy assessment results for the LCZ classification models.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--png_name", default="classwise_f1_comparison.png")
    args = parser.parse_args()

    model_items = parse_named_paths(args.model, "--model")
    summary_items = parse_named_paths(args.summary, "--summary")
    cm_items = parse_named_paths(args.cm, "--cm")

    model_names = [name for name, _ in model_items]
    if len(set(model_names)) != len(model_names):
        raise ValueError("--model names must be unique")

    cm_names = [name for name, _ in cm_items]
    if set(cm_names) != set(model_names):
        missing = set(model_names) - set(cm_names)
        extra = set(cm_names) - set(model_names)
        raise ValueError(f"--cm model names must match --model names. Missing={missing}, extra={extra}")

    class_codes, comparison = load_classwise(model_items)
    if list(map(int, args.class_codes)) != list(map(int, class_codes)):
        raise ValueError(
            f"--class_codes {args.class_codes} must match classwise files {class_codes}"
        )

    summaries = load_summaries(summary_items)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Class-wise F1 comparison CSV and figure.
    pivot = comparison.pivot(index="class_code", columns="model", values="F1").reset_index()
    pivot = pivot[["class_code"] + model_names]
    pivot.to_csv(out_dir / "classwise_f1_comparison.csv", index=False, encoding="utf-8-sig")

    plot_classwise_f1(
        class_codes=class_codes,
        comparison=comparison,
        model_names=model_names,
        out_png=out_dir / args.png_name,
        title=args.title,
        dpi=args.dpi,
    )

    # 2) Overall LCZ accuracy table.
    full_metrics = merge_metrics(
        model_names=model_names,
        summaries=summaries,
        cm_items=cm_items,
        class_codes=args.class_codes,
    )
    full_metrics.to_csv(out_dir / "lcz_metrics_full.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    table_df = full_metrics[[
        "Method",
        "Input",
        "Encoder type",
        "Building representation",
        "OA",
        "OAurb",
        "OAu",
        "Macro-F1",
        "Weighted-F1",
        "Kappa",
    ]].copy()
    table_df.to_csv(out_dir / "lcz_accuracy_table.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    plot_accuracy_table(
        table_df=table_df,
        out_png=out_dir / "lcz_accuracy_table.png",
        caption=args.table_caption,
        dpi=args.dpi,
    )

    print("saved:", out_dir / args.png_name)
    print("saved:", out_dir / "classwise_f1_comparison.csv")
    print("saved:", out_dir / "lcz_accuracy_table.csv")
    print("saved:", out_dir / "lcz_accuracy_table.png")
    print("saved:", out_dir / "lcz_metrics_full.csv")


if __name__ == "__main__":
    main()
