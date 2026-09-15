import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
cd /mnt/disk1/workspace_jym/LCZ


# A. RS-only baseline
CUDA_VISIBLE_DEVICES=2 python train/train_lcz_rs_cnn.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --exp_name cnn_rs_baseline \
  --patch_size 33 \
  --batch_size 256 \
  --epochs 150 \
  --patience 25 \
  --class_weight \
  --rebuild_split

# B. RS + rasterized building baseline
CUDA_VISIBLE_DEVICES=2 python train/train_lcz_rs_cnn.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm_rs_building \
  --exp_name cnn_rs_raster_building \
  --patch_size 33 \
  --batch_size 256 \
  --epochs 150 \
  --patience 25 \
  --class_weight

"""
# ============================================================
# Default settings
# ============================================================

DEFAULT_BASE_DIR = "/mnt/disk1/workspace_jym/LCZ/data"
DEFAULT_WORK_DIR = "/mnt/disk1/workspace_jym/LCZ/work_dirs"

# 현재 seoul_LCZ.tif에 존재하는 실제 class
LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]

LCZ_CLASS_NAMES = {
    1: "LCZ1",
    2: "LCZ2",
    3: "LCZ3",
    4: "LCZ4",
    5: "LCZ5",
    6: "LCZ6",
    8: "LCZ8",
    101: "LCZA",
    102: "LCZB",
    104: "LCZD",
    107: "LCZG",
}


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_stack(norm_dir: Path):
    """
    Load b1_norm.mat, b2_norm.mat, ... as C x H x W feature stack.
    Each .mat file should contain variable 'norm'.
    """
    mat_paths = sorted(norm_dir.glob("b*_norm.mat"))

    if len(mat_paths) == 0:
        raise FileNotFoundError(f"No b*_norm.mat files found in {norm_dir}")

    bands = []
    print("========== Load normalized feature bands ==========")

    for p in mat_paths:
        data = loadmat(p)
        if "norm" not in data:
            raise KeyError(f"'norm' variable not found in {p}")

        arr = data["norm"].astype(np.float32)
        arr[~np.isfinite(arr)] = 0.0

        bands.append(arr)

        print(
            f"{p.name}: shape={arr.shape}, "
            f"min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"
        )

    stack = np.stack(bands, axis=0).astype(np.float32)
    print("Feature stack:", stack.shape)

    return stack, mat_paths


def make_polygon_split(component_csv: Path,
                       out_dir: Path,
                       seed: int = 42,
                       train_ratio: float = 0.6,
                       val_ratio: float = 0.2):
    """
    Polygon-level split.
    Same polygon_id never appears in more than one split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(component_csv)
    df["polygon_id"] = df["polygon_id"].astype(int)
    df["lcz_class"] = df["lcz_class"].astype(int)

    rng = np.random.default_rng(seed)
    records = []

    for cls in LCZ_CLASSES:
        cls_df = df[df["lcz_class"] == cls].copy()
        polygon_ids = cls_df["polygon_id"].values.astype(int)

        if len(polygon_ids) == 0:
            print(f"[WARN] class {cls}: no polygons")
            continue

        rng.shuffle(polygon_ids)

        n = len(polygon_ids)

        if n >= 3:
            n_train = max(1, int(np.floor(n * train_ratio)))
            n_val = max(1, int(np.floor(n * val_ratio)))

            # 최소 test 1개 보장
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1

            train_ids = polygon_ids[:n_train]
            val_ids = polygon_ids[n_train:n_train + n_val]
            test_ids = polygon_ids[n_train + n_val:]
        elif n == 2:
            train_ids = polygon_ids[:1]
            val_ids = polygon_ids[1:]
            test_ids = np.array([], dtype=int)
        else:
            train_ids = polygon_ids
            val_ids = np.array([], dtype=int)
            test_ids = np.array([], dtype=int)

        for pid in train_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "train"})
        for pid in val_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "val"})
        for pid in test_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "test"})

        print(
            f"Class {cls}: total={n}, "
            f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

    split_df = pd.DataFrame(records)
    split_path = out_dir / "polygon_split.csv"
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    print("Saved polygon split:", split_path)
    return split_df


def build_sample_index(gt_path: Path,
                       polygon_tif: Path,
                       split_df: pd.DataFrame,
                       out_dir: Path):
    """
    Build train/val/test sample index from 50m LCZ raster and polygon number raster.
    Each sample is one 50m LCZ pixel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(gt_path) as src:
        lcz = src.read(1)
        nodata = src.nodata

    with rasterio.open(polygon_tif) as src:
        poly = src.read(1)

    if lcz.shape != poly.shape:
        raise ValueError(f"Shape mismatch: lcz={lcz.shape}, poly={poly.shape}")

    class_to_idx = {cls: i for i, cls in enumerate(LCZ_CLASSES)}

    valid_lcz = np.ones_like(lcz, dtype=bool)
    if nodata is not None:
        valid_lcz &= (lcz != nodata)

    valid_lcz &= np.isin(lcz, LCZ_CLASSES)
    valid_poly = poly > 0

    print("\n========== Build sample index ==========")
    print("LCZ shape:", lcz.shape)
    print("Valid LCZ pixels:", int(valid_lcz.sum()))
    print("Valid polygon pixels:", int(valid_poly.sum()))

    for split in ["train", "val", "test"]:
        split_pids = split_df.loc[split_df["split"] == split, "polygon_id"].values.astype(int)

        split_mask = np.isin(poly, split_pids) & valid_lcz & valid_poly

        rows, cols = np.where(split_mask)
        labels_raw = lcz[rows, cols].astype(int)
        labels = np.array([class_to_idx[int(v)] for v in labels_raw], dtype=np.int64)

        out_path = out_dir / f"{split}_samples.npz"
        np.savez_compressed(
            out_path,
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            labels=labels,
            labels_raw=labels_raw.astype(np.int32),
        )

        print(f"{split}: samples={len(labels)} -> {out_path}")

        if len(labels) > 0:
            unique, counts = np.unique(labels_raw, return_counts=True)
            print("  class distribution:")
            for u, c in zip(unique, counts):
                print(f"    {u}: {c}")

    meta = {
        "lcz_classes": LCZ_CLASSES,
        "class_to_idx": {str(k): v for k, v in class_to_idx.items()},
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
    }

    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ============================================================
# Dataset
# ============================================================

class LCZPatchDataset(Dataset):
    """
    50m LCZ cell classification dataset.

    Input:
        33 x 33 x C 10m feature patch

    Label:
        center 50m LCZ class
    """

    def __init__(self,
                 feature_stack: np.ndarray,
                 sample_npz: Path,
                 patch_size: int = 33):
        self.feature_stack = feature_stack
        self.patch_size = patch_size
        self.radius = patch_size // 2

        data = np.load(sample_npz)
        self.rows50 = data["rows"].astype(np.int64)
        self.cols50 = data["cols"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)

        # edge patch 처리를 위해 padding
        self.padded = np.pad(
            self.feature_stack,
            pad_width=((0, 0), (self.radius, self.radius), (self.radius, self.radius)),
            mode="constant",
            constant_values=0.0,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])

        # 50m pixel 하나는 10m pixel 5x5에 대응
        # 해당 50m cell의 중심 10m pixel
        r10 = r50 * 5 + 2
        c10 = c50 * 5 + 2

        # padding shift
        r10 += self.radius
        c10 += self.radius

        patch = self.padded[
            :,
            r10 - self.radius:r10 + self.radius + 1,
            c10 - self.radius:c10 + self.radius + 1,
        ]

        if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
            raise RuntimeError(f"Invalid patch shape: {patch.shape}")

        x = torch.from_numpy(patch.copy()).float()
        y = torch.tensor(self.labels[idx]).long()

        return x, y


# ============================================================
# Model
# ============================================================

class LCZCNN(nn.Module):
    """
    Similar spirit to the paper baseline:
    Conv2D x 4 + MaxPool x 2 + FC + Softmax classifier.
    """

    def __init__(self, in_channels: int, num_classes: int, patch_size: int = 33, dropout: float = 0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            feat_dim = self.encoder(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


# ============================================================
# Train / Eval
# ============================================================

def evaluate(model, loader, device, num_classes):
    model.eval()

    all_preds = []
    all_labels = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss = total_loss / len(all_labels)
    oa = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    return {
        "loss": avg_loss,
        "oa": oa,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cm": cm,
        "preds": all_preds,
        "labels": all_labels,
    }


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0

    # inverse sqrt frequency
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def train(args):
    set_seed(args.seed)

    base_dir = Path(args.base_dir)

    gt_path = base_dir / "GT" / "seoul_LCZ.tif"
    polygon_tif = base_dir / "GT" / "lcz_polygonnumber_50m.tif"
    component_csv = base_dir / "GT" / "LCZ_class_from_components.csv"

    if args.norm_dir is None:
        norm_dir = base_dir / "Satellite" / "processed" / "norm"
    else:
        norm_dir = Path(args.norm_dir)

    # Save all experiment outputs under work_dir, not under data/experiments.
    # Example:
    #   /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/
    work_dir = Path(args.work_dir)
    exp_dir = work_dir / args.exp_name
    split_dir = exp_dir / "splits"
    ckpt_dir = exp_dir / "checkpoints"
    result_dir = exp_dir / "results"

    split_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Base data dir:", base_dir)
    print("Work dir:", work_dir)
    print("Experiment dir:", exp_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    # ------------------------------------------------------------
    # 1. Build split and sample index
    # ------------------------------------------------------------
    polygon_split_path = split_dir / "polygon_split.csv"

    if polygon_split_path.exists() and not args.rebuild_split:
        print("Load existing polygon split:", polygon_split_path)
        split_df = pd.read_csv(polygon_split_path)
    else:
        split_df = make_polygon_split(
            component_csv=component_csv,
            out_dir=split_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    sample_files_exist = all((split_dir / f"{s}_samples.npz").exists() for s in ["train", "val", "test"])

    if not sample_files_exist or args.rebuild_split:
        build_sample_index(
            gt_path=gt_path,
            polygon_tif=polygon_tif,
            split_df=split_df,
            out_dir=split_dir,
        )

    # ------------------------------------------------------------
    # 2. Load features
    # ------------------------------------------------------------
    features, band_paths = load_feature_stack(norm_dir)
    in_channels = features.shape[0]
    num_classes = len(LCZ_CLASSES)

    # ------------------------------------------------------------
    # 3. Dataset / Loader
    # ------------------------------------------------------------
    train_ds = LCZPatchDataset(features, split_dir / "train_samples.npz", patch_size=args.patch_size)
    val_ds = LCZPatchDataset(features, split_dir / "val_samples.npz", patch_size=args.patch_size)
    test_ds = LCZPatchDataset(features, split_dir / "test_samples.npz", patch_size=args.patch_size)

    print("\n========== Dataset ==========")
    print("train:", len(train_ds))
    print("val:", len(val_ds))
    print("test:", len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------
    # 4. Model
    # ------------------------------------------------------------
    model = LCZCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=args.patch_size,
        dropout=args.dropout,
    ).to(device)

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]

    if args.class_weight:
        weights = compute_class_weights(train_labels, num_classes).to(device)
        print("Class weights:", weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_oa = -1.0
    best_epoch = -1
    patience_count = 0

    log_records = []

    # ------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------
    print("\n========== Training ==========")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(n_seen, 1)

        val_result = evaluate(model, val_loader, device, num_classes)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_result['loss']:.4f} | "
            f"val_OA={val_result['oa']:.4f} | "
            f"val_macroF1={val_result['macro_f1']:.4f}"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_oa": val_result["oa"],
            "val_macro_f1": val_result["macro_f1"],
            "val_weighted_f1": val_result["weighted_f1"],
        })

        if val_result["oa"] > best_val_oa:
            best_val_oa = val_result["oa"]
            best_epoch = epoch
            patience_count = 0

            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "patch_size": args.patch_size,
                    "lcz_classes": LCZ_CLASSES,
                    "band_files": [str(p) for p in band_paths],
                    "val_oa": best_val_oa,
                },
                ckpt_path,
            )

            print(f"  Saved best model: {ckpt_path}")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(log_records).to_csv(result_dir / "train_log.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------
    # 6. Test best model
    # ------------------------------------------------------------
    print("\n========== Test ==========")

    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_result = evaluate(model, test_loader, device, num_classes)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val OA: {best_val_oa:.4f}")
    print(f"Test OA: {test_result['oa']:.4f}")
    print(f"Test Macro-F1: {test_result['macro_f1']:.4f}")
    print(f"Test Weighted-F1: {test_result['weighted_f1']:.4f}")

    cm = test_result["cm"]
    np.savetxt(result_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    target_names = [LCZ_CLASS_NAMES[c] for c in LCZ_CLASSES]

    report = classification_report(
        test_result["labels"],
        test_result["preds"],
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    with open(result_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_oa": float(best_val_oa),
        "test_oa": float(test_result["oa"]),
        "test_macro_f1": float(test_result["macro_f1"]),
        "test_weighted_f1": float(test_result["weighted_f1"]),
        "classes": LCZ_CLASSES,
        "class_names": target_names,
    }

    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved results to:", result_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--work_dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--exp_name", type=str, default="cnn_rs_baseline")

    parser.add_argument("--patch_size", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--rebuild_split", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--norm_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)