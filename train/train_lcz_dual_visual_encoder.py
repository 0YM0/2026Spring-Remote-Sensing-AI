#!/usr/bin/env python3
"""
CNN_RS_BuildingVisualEncoder
- RS branch: 10m RS patch, e.g. b1~b8_norm.mat
- Building visual branch: 10m rasterized building patch, e.g. b9~b10_norm.mat
- Fusion: concat(RS feature, building visual feature) -> LCZ class

Recommended run:
cd /mnt/disk1/workspace_jym/LCZ
CUDA_VISIBLE_DEVICES=2 python train/train_lcz_dual_visual_encoder.py \
  --base_dir /mnt/disk1/workspace_jym/LCZ/data \
  --work_dir /mnt/disk1/workspace_jym/LCZ/work_dirs \
  --rs_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm \
  --building_norm_dir /mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm_rs_building \
  --split_dir /mnt/disk1/workspace_jym/LCZ/work_dirs/cnn_rs_baseline/splits \
  --exp_name cnn_rs_building_visual_encoder \
  --patch_size 33 \
  --batch_size 256 \
  --epochs 150 \
  --patience 25 \
  --class_weight
"""

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

DEFAULT_BASE_DIR = "/mnt/disk1/workspace_jym/LCZ/data"
DEFAULT_WORK_DIR = "/mnt/disk1/workspace_jym/LCZ/work_dirs"

LCZ_CLASSES = [1, 2, 3, 4, 5, 6, 8, 101, 102, 104, 107]
LCZ_CLASS_NAMES = {
    1: "LCZ1", 2: "LCZ2", 3: "LCZ3", 4: "LCZ4", 5: "LCZ5", 6: "LCZ6", 8: "LCZ8",
    101: "LCZA", 102: "LCZB", 104: "LCZD", 107: "LCZG",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_mat_stack(paths):
    bands = []
    for p in paths:
        data = loadmat(p)
        if "norm" not in data:
            raise KeyError(f"'norm' variable not found in {p}")
        arr = data["norm"].astype(np.float32)
        arr[~np.isfinite(arr)] = 0.0
        bands.append(arr)
        print(f"{p.name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    stack = np.stack(bands, axis=0).astype(np.float32)
    print("Stack:", stack.shape)
    return stack


def load_rs_stack(rs_norm_dir: Path):
    print("========== Load RS bands ==========")
    paths = [rs_norm_dir / f"b{i}_norm.mat" for i in range(1, 9)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RS norm files: {missing}")
    return load_mat_stack(paths), paths


def load_building_stack(building_norm_dir: Path):
    print("========== Load building raster bands ==========")
    # norm_rs_building 폴더 안의 b9, b10을 building branch 입력으로 사용
    paths = [building_norm_dir / "b9_norm.mat", building_norm_dir / "b10_norm.mat"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing building norm files: {missing}")
    return load_mat_stack(paths), paths


def make_polygon_split(component_csv: Path, out_dir: Path, seed: int = 42, train_ratio: float = 0.6, val_ratio: float = 0.2):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(component_csv)
    df["polygon_id"] = df["polygon_id"].astype(int)
    df["lcz_class"] = df["lcz_class"].astype(int)
    rng = np.random.default_rng(seed)
    records = []

    for cls in LCZ_CLASSES:
        ids = df.loc[df["lcz_class"] == cls, "polygon_id"].values.astype(int)
        if len(ids) == 0:
            print(f"[WARN] class {cls}: no polygons")
            continue
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            n_train = max(1, int(np.floor(n * train_ratio)))
            n_val = max(1, int(np.floor(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train_ids = ids[:n_train]
            val_ids = ids[n_train:n_train + n_val]
            test_ids = ids[n_train + n_val:]
        elif n == 2:
            train_ids, val_ids, test_ids = ids[:1], ids[1:], np.array([], dtype=int)
        else:
            train_ids, val_ids, test_ids = ids, np.array([], dtype=int), np.array([], dtype=int)

        for pid in train_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "train"})
        for pid in val_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "val"})
        for pid in test_ids:
            records.append({"polygon_id": int(pid), "lcz_class": int(cls), "split": "test"})
        print(f"Class {cls}: total={n}, train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    split_df = pd.DataFrame(records)
    split_df.to_csv(out_dir / "polygon_split.csv", index=False, encoding="utf-8-sig")
    return split_df


def build_sample_index(gt_path: Path, polygon_tif: Path, split_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(gt_path) as src:
        lcz = src.read(1)
        nodata = src.nodata
    with rasterio.open(polygon_tif) as src:
        poly = src.read(1)
    if lcz.shape != poly.shape:
        raise ValueError(f"Shape mismatch: lcz={lcz.shape}, poly={poly.shape}")

    class_to_idx = {cls: i for i, cls in enumerate(LCZ_CLASSES)}
    valid = np.isin(lcz, LCZ_CLASSES) & (poly > 0)
    if nodata is not None:
        valid &= (lcz != nodata)

    for split in ["train", "val", "test"]:
        pids = split_df.loc[split_df["split"] == split, "polygon_id"].values.astype(int)
        mask = np.isin(poly, pids) & valid
        rows, cols = np.where(mask)
        labels_raw = lcz[rows, cols].astype(int)
        labels = np.array([class_to_idx[int(v)] for v in labels_raw], dtype=np.int64)
        np.savez_compressed(
            out_dir / f"{split}_samples.npz",
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            labels=labels,
            labels_raw=labels_raw.astype(np.int32),
        )
        print(f"{split}: samples={len(labels)} -> {out_dir / f'{split}_samples.npz'}")

    meta = {
        "lcz_classes": LCZ_CLASSES,
        "class_to_idx": {str(k): v for k, v in class_to_idx.items()},
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
    }
    with open(out_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


class LCZDualPatchDataset(Dataset):
    def __init__(self, rs_stack: np.ndarray, building_stack: np.ndarray, sample_npz: Path, patch_size: int = 33):
        if rs_stack.shape[1:] != building_stack.shape[1:]:
            raise ValueError(f"RS/building shape mismatch: {rs_stack.shape} vs {building_stack.shape}")
        self.rs_stack = rs_stack
        self.building_stack = building_stack
        self.patch_size = patch_size
        self.radius = patch_size // 2
        data = np.load(sample_npz)
        self.rows50 = data["rows"].astype(np.int64)
        self.cols50 = data["cols"].astype(np.int64)
        self.labels = data["labels"].astype(np.int64)
        self.rs_pad = np.pad(rs_stack, ((0, 0), (self.radius, self.radius), (self.radius, self.radius)), mode="constant")
        self.bld_pad = np.pad(building_stack, ((0, 0), (self.radius, self.radius), (self.radius, self.radius)), mode="constant")

    def __len__(self):
        return len(self.labels)

    def _crop(self, stack_pad, r50, c50):
        r10 = r50 * 5 + 2 + self.radius
        c10 = c50 * 5 + 2 + self.radius
        patch = stack_pad[:, r10 - self.radius:r10 + self.radius + 1, c10 - self.radius:c10 + self.radius + 1]
        if patch.shape[1] != self.patch_size or patch.shape[2] != self.patch_size:
            raise RuntimeError(f"Invalid patch shape: {patch.shape}")
        return patch

    def __getitem__(self, idx):
        r50 = int(self.rows50[idx])
        c50 = int(self.cols50[idx])
        rs = self._crop(self.rs_pad, r50, c50)
        bld = self._crop(self.bld_pad, r50, c50)
        y = self.labels[idx]
        return torch.from_numpy(rs.copy()).float(), torch.from_numpy(bld.copy()).float(), torch.tensor(y).long()


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, out_dim: int, base_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, patch_size, patch_size)
            flat_dim = self.encoder(dummy).view(1, -1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, out_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.proj(self.encoder(x))


class DualVisualLCZCNN(nn.Module):
    def __init__(self, rs_channels: int, building_channels: int, num_classes: int, patch_size: int = 33, dropout: float = 0.5):
        super().__init__()
        self.rs_encoder = ConvEncoder(rs_channels, patch_size, out_dim=256, base_channels=32)
        self.building_encoder = ConvEncoder(building_channels, patch_size, out_dim=128, base_channels=16)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, rs, building):
        rs_feat = self.rs_encoder(rs)
        bld_feat = self.building_encoder(building)
        return self.classifier(torch.cat([rs_feat, bld_feat], dim=1))


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for rs, bld, y in loader:
            rs, bld, y = rs.to(device), bld.to(device), y.to(device)
            logits = model(rs, bld)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return {
        "loss": total_loss / max(n, 1),
        "oa": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "cm": confusion_matrix(labels, preds, labels=list(range(num_classes))),
        "labels": labels,
        "preds": preds,
    }


def train(args):
    set_seed(args.seed)
    base_dir = Path(args.base_dir)
    work_dir = Path(args.work_dir)
    exp_dir = work_dir / args.exp_name
    split_dir = Path(args.split_dir) if args.split_dir else (work_dir / "cnn_rs_baseline" / "splits")
    local_split_dir = exp_dir / "splits"
    ckpt_dir = exp_dir / "checkpoints"
    result_dir = exp_dir / "results"
    for d in [local_split_dir, ckpt_dir, result_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # split_dir에 샘플이 없으면 새로 생성. 있으면 그대로 재사용.
    if not all((split_dir / f"{s}_samples.npz").exists() for s in ["train", "val", "test"]):
        print(f"Split files not found in {split_dir}. Build new split there.")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = make_polygon_split(base_dir / "GT" / "LCZ_class_from_components.csv", split_dir, args.seed, args.train_ratio, args.val_ratio)
        build_sample_index(base_dir / "GT" / "seoul_LCZ.tif", base_dir / "GT" / "lcz_polygonnumber_50m.tif", split_df, split_dir)
    print("Using split_dir:", split_dir)

    rs_stack, rs_paths = load_rs_stack(Path(args.rs_norm_dir))
    bld_stack, bld_paths = load_building_stack(Path(args.building_norm_dir))

    train_ds = LCZDualPatchDataset(rs_stack, bld_stack, split_dir / "train_samples.npz", args.patch_size)
    val_ds = LCZDualPatchDataset(rs_stack, bld_stack, split_dir / "val_samples.npz", args.patch_size)
    test_ds = LCZDualPatchDataset(rs_stack, bld_stack, split_dir / "test_samples.npz", args.patch_size)
    print(f"Dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = DualVisualLCZCNN(rs_stack.shape[0], bld_stack.shape[0], len(LCZ_CLASSES), args.patch_size, args.dropout).to(device)
    num_params = count_parameters(model)
    print("Device:", device)
    print("Trainable parameters:", f"{num_params:,}")

    train_labels = np.load(split_dir / "train_samples.npz")["labels"]
    if args.class_weight:
        weights = compute_class_weights(train_labels, len(LCZ_CLASSES)).to(device)
        print("Class weights:", weights.detach().cpu().numpy())
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_oa, best_epoch, patience = -1.0, -1, 0
    logs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for rs, bld, y in train_loader:
            rs, bld, y = rs.to(device), bld.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(rs, bld), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = total_loss / max(n, 1)
        val = evaluate(model, val_loader, device, len(LCZ_CLASSES))
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | val_OA={val['oa']:.4f} | val_macroF1={val['macro_f1']:.4f}")
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"], "val_oa": val["oa"], "val_macro_f1": val["macro_f1"], "val_weighted_f1": val["weighted_f1"]})
        if val["oa"] > best_val_oa:
            best_val_oa, best_epoch, patience = val["oa"], epoch, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "rs_channels": int(rs_stack.shape[0]),
                "building_channels": int(bld_stack.shape[0]),
                "num_classes": len(LCZ_CLASSES),
                "patch_size": args.patch_size,
                "lcz_classes": LCZ_CLASSES,
                "params": num_params,
                "val_oa": best_val_oa,
                "rs_files": [str(p) for p in rs_paths],
                "building_files": [str(p) for p in bld_paths],
            }, ckpt_dir / "best_model.pt")
            print("  Saved best model")
        else:
            patience += 1
        if patience >= args.patience:
            print("Early stopping")
            break

    pd.DataFrame(logs).to_csv(result_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(model, test_loader, device, len(LCZ_CLASSES))
    np.savetxt(result_dir / "confusion_matrix.csv", test["cm"], delimiter=",", fmt="%d")
    target_names = [LCZ_CLASS_NAMES[c] for c in LCZ_CLASSES]
    report = classification_report(test["labels"], test["preds"], labels=list(range(len(LCZ_CLASSES))), target_names=target_names, zero_division=0, output_dict=True)
    with open(result_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    summary = {
        "model": "CNN_RS_BuildingVisualEncoder",
        "best_epoch": int(best_epoch),
        "best_val_oa": float(best_val_oa),
        "test_oa": float(test["oa"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "params": int(num_params),
        "classes": LCZ_CLASSES,
        "class_names": target_names,
    }
    with open(result_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("========== Test ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved results to:", result_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    p.add_argument("--work_dir", type=str, default=DEFAULT_WORK_DIR)
    p.add_argument("--rs_norm_dir", type=str, default="/mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm")
    p.add_argument("--building_norm_dir", type=str, default="/mnt/disk1/workspace_jym/LCZ/data/Satellite/processed/norm_rs_building")
    p.add_argument("--split_dir", type=str, default=None, help="Use the same split as baseline, e.g. work_dirs/cnn_rs_baseline/splits")
    p.add_argument("--exp_name", type=str, default="cnn_rs_building_visual_encoder")
    p.add_argument("--patch_size", type=int, default=33)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--train_ratio", type=float, default=0.6)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
