#!/usr/bin/env python3
"""Build Label Studio tasks with brush-RLE predictions from a generalizability yolo_dataset.

Supports the WaRP, PackWISE and SpectralWaste exports. The pseudo-GT polygons are loaded
into Label Studio for refinement; 0804_Generalizability_ls_export_truegt.py converts the
refined export back.

Inputs (after renaming `labels/` to `labels_pseudo_gt/`):

    <yolo_dataset>/
        images/<stem>.png
        labels_pseudo_gt/<stem>.txt      YOLO-seg polygons, CLASS_MAP ids
        export_index.csv
        data.yaml
        DATASET_CARD.md

Polygons are rasterized to per-class binary masks at the image's own resolution and
encoded as Label Studio brush RLE predictions; no coordinate transform is applied.

Outputs (build mode):
    <out-json>                    tasks JSON for POST /api/projects/<id>/import
    <out-json>.provenance.json    run record

Modes:
    inspect  print counts, region presence, resolution check, first-task preview
    build    write the tasks JSON and provenance record

Usage:
    python 0805_Generalizability_ls_import_pseudogt.py --mode inspect --run-dir <yolo_dataset>
    python 0805_Generalizability_ls_import_pseudogt.py --mode build --run-dir <yolo_dataset> \
        --out-json <tasks.json> --dataset <name> [--sample-n N]
"""

# --- Imports ---
import os

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from label_studio_sdk.converter import brush
except ImportError:
    import label_studio_converter.brush as brush

# --- Config ---

CLASS_MAP = {"body": 0, "cap": 1, "label": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
REGIONS = ["body", "label", "cap"]

IMAGES_DIRNAME = "images"
PSEUDO_LABELS_DIRNAME = "labels_pseudo_gt"
EXPORT_INDEX_NAME = "export_index.csv"

FROM_NAME = "tag"
TO_NAME = "image"

DOCUMENT_ROOT = os.environ.get("KI_SORT_VOL_ROOT", "/VOL_D/")
LOCAL_FILES_PREFIX = "/data/local-files/?d="

# Label Studio runs behind a reverse proxy under a per-user subpath, so image URLs must
# include host and subpath. Set to the LABEL_STUDIO_HOST value of the running server;
# an empty string yields root-relative URLs.
LABEL_STUDIO_HOST = "https://jupyter.ki-werkstatt.htw-aalen.de/labelstudio/fio.h"

IMAGE_EXTS = [".png", ".jpg", ".jpeg"]

SAMPLE_N = 0            # 0 = all paired objects
RANDOM_SEED = 42
STRATIFY_COLS = []

MIN_MASK_PX = 1

# --- Helpers ---


def local_files_url(abs_path: str) -> str:
    """Label Studio local-files URL for an absolute path under DOCUMENT_ROOT."""
    rel = str(Path(abs_path)).replace(DOCUMENT_ROOT, "", 1).lstrip("/")
    return LABEL_STUDIO_HOST + LOCAL_FILES_PREFIX + urllib.parse.quote(rel, safe="/")


def find_image(images_dir: Path, stem: str):
    """First existing image for stem with one of IMAGE_EXTS, or None."""
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_yolo_seg(txt_path: Path, width: int, height: int) -> dict:
    """Rasterize YOLO-seg polygon lines into one binary mask per class name."""
    masks = {r: np.zeros((height, width), dtype=np.uint8) for r in REGIONS}
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls_id = int(float(parts[0]))
        name = ID_TO_CLASS.get(cls_id)
        if name is None:
            continue
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float64)
        if coords.size % 2 != 0:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        pts = np.round(pts).astype(np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(masks[name], [pts], 1)
    return {k: v.astype(bool) for k, v in masks.items()}


def brush_region(mask_bool: np.ndarray, label: str, width: int, height: int) -> dict:
    """Label Studio brushlabels result item (RLE) for one region mask."""
    m = (mask_bool.astype(np.uint8)) * 255
    return {
        "from_name": FROM_NAME, "to_name": TO_NAME, "type": "brushlabels",
        "original_width": int(width), "original_height": int(height),
        "image_rotation": 0,
        "value": {"format": "rle", "rle": brush.mask2rle(m), "brushlabels": [label]},
    }


def stratified_sample(df: pd.DataFrame, n: int, cols: list, seed: int) -> pd.DataFrame:
    """Sample about n rows proportionally per group of cols; n <= 0 returns all rows."""
    if n <= 0 or n >= len(df):
        return df
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.sample(n=n, random_state=seed)
    frac = n / len(df)
    out = []
    for _, grp in df.groupby(cols, dropna=False):
        k = max(1, int(round(len(grp) * frac)))
        out.append(grp.sample(n=min(k, len(grp)), random_state=seed))
    sampled = pd.concat(out)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    return sampled


def collect_pairs(root: Path):
    """Return (pairs, orphan_labels, orphan_images) for stems present in both dirs."""
    images_dir = root / IMAGES_DIRNAME
    labels_dir = root / PSEUDO_LABELS_DIRNAME
    if not images_dir.is_dir():
        sys.exit(f"missing {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"missing {labels_dir} — rename `labels/` to `{PSEUDO_LABELS_DIRNAME}/` first")

    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    image_stems = set()
    for ext in IMAGE_EXTS:
        image_stems |= {p.stem for p in images_dir.glob(f"*{ext}")}

    pairs = [(s, find_image(images_dir, s), labels_dir / f"{s}.txt")
             for s in sorted(label_stems & image_stems)]
    return pairs, sorted(label_stems - image_stems), sorted(image_stems - label_stems)


def config_hash(payload: dict) -> str:
    """Short SHA-1 hash of a JSON-serialised payload."""
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


# --- Modes ---


def run_inspect(run_dir: str) -> None:
    """Print pairing, export_index key column, resolutions and a first-object preview."""
    root = Path(run_dir)
    pairs, orphan_labels, orphan_images = collect_pairs(root)

    print(f"yolo_dataset root : {root}")
    print(f"paired objects    : {len(pairs)}")
    print(f"labels w/o image  : {len(orphan_labels)}")
    print(f"images w/o label  : {len(orphan_images)}")

    idx_path = root / EXPORT_INDEX_NAME
    if idx_path.exists():
        idx = pd.read_csv(idx_path)
        stems = {s for s, _, _ in pairs}
        best, best_hits = None, -1
        for c in idx.columns:
            hits = sum(1 for v in idx[c].astype(str)
                       if v in stems or Path(v).stem in stems)
            if hits > best_hits:
                best, best_hits = c, hits
        print(f"\n{EXPORT_INDEX_NAME}: {len(idx)} rows")
        print(f"  columns: {list(idx.columns)}")
        print(f"  best stem-matching column: {best!r} ({best_hits}/{len(idx)} rows match)")
        print("  pass to the export script as --index-key-col if auto-detect is wrong")
    else:
        print(f"\n{EXPORT_INDEX_NAME} NOT FOUND at {idx_path}")

    if not pairs:
        return

    counts = {r: 0 for r in REGIONS}
    res_seen = {}
    for stem, img_p, lbl_p in pairs[:200]:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        res_seen[(w, h)] = res_seen.get((w, h), 0) + 1
        masks = parse_yolo_seg(lbl_p, w, h)
        for r in REGIONS:
            if masks[r].any():
                counts[r] += 1

    print(f"\nresolutions seen (first 200): {res_seen}")
    print("region presence (first 200 objects):")
    for r in REGIONS:
        print(f"  {r:6s} {counts[r]}")

    stem, img_p, lbl_p = pairs[0]
    img = cv2.imread(str(img_p))
    h, w = img.shape[:2]
    masks = parse_yolo_seg(lbl_p, w, h)
    print(f"\nfirst object: {stem}  ({w}x{h})")
    for r in REGIONS:
        print(f"  {r:6s} px={int(masks[r].sum()):8d}")
    print(f"  image url: {local_files_url(str(img_p))}")


def run_build(run_dir: str, out_json: str, dataset: str, sample_n: int) -> None:
    """Write the Label Studio tasks JSON and its provenance record."""
    root = Path(run_dir)
    pairs, orphan_labels, orphan_images = collect_pairs(root)
    n_paired = len(pairs)

    df = pd.DataFrame([{"stem": s, "image_path": str(i), "label_path": str(l)}
                       for s, i, l in pairs])

    idx_path = root / EXPORT_INDEX_NAME
    if idx_path.exists() and STRATIFY_COLS:
        idx = pd.read_csv(idx_path)
        if "stem" in idx.columns:
            keep = ["stem"] + [c for c in STRATIFY_COLS if c in idx.columns]
            df = df.merge(idx[keep], on="stem", how="left")

    df = stratified_sample(df, sample_n, STRATIFY_COLS, RANDOM_SEED).sort_values("stem")

    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_version = f"sam3_pseudogt_{dataset}_{run_id}"

    tasks, n_skipped, n_empty = [], 0, 0
    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"])
        if img is None:
            n_skipped += 1
            continue
        h, w = img.shape[:2]

        masks = parse_yolo_seg(Path(row["label_path"]), w, h)

        result = []
        for region in REGIONS:
            if int(masks[region].sum()) < MIN_MASK_PX:
                continue
            result.append(brush_region(masks[region], region, w, h))

        if not result:
            n_empty += 1
            continue

        tasks.append({
            "data": {
                "image": local_files_url(row["image_path"]),
                "dataset": dataset,
                "stem": row["stem"],
                "image_path": row["image_path"],
                "pseudo_label_path": row["label_path"],
                "yolo_dataset_root": str(root),
                "img_w": int(w), "img_h": int(h),
            },
            "predictions": [{"model_version": model_version, "result": result}],
        })

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=1))

    sidecar = {
        "run_id": run_id,
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": dataset,
        "yolo_dataset_root": str(root),
        "model_version": model_version,
        "pseudo_labels_dir": PSEUDO_LABELS_DIRNAME,
        "n_paired": n_paired,
        "n_orphan_labels": len(orphan_labels),
        "n_orphan_images": len(orphan_images),
        "n_sampled": len(df),
        "n_tasks_written": len(tasks),
        "n_skipped_unreadable_image": n_skipped,
        "n_skipped_empty_label": n_empty,
        "sample_n": sample_n,
        "random_seed": RANDOM_SEED,
        "stratify_cols": STRATIFY_COLS,
        "class_map": CLASS_MAP,
        "document_root": DOCUMENT_ROOT,
    }
    sidecar["config_hash"] = config_hash(sidecar)
    Path(str(out_path) + ".provenance.json").write_text(json.dumps(sidecar, indent=2))

    print(f"paired objects        : {n_paired}")
    print(f"orphan labels/images  : {len(orphan_labels)} / {len(orphan_images)}")
    print(f"sampled               : {len(df)}")
    print(f"tasks written         : {len(tasks)}")
    print(f"skipped (bad image)   : {n_skipped}")
    print(f"skipped (empty label) : {n_empty}")
    print(f"model_version         : {model_version}")
    print(f"tasks json            : {out_path}")
    print(f"provenance            : {out_path}.provenance.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inspect", "build"], required=True)
    ap.add_argument("--run-dir", required=True, help="path to the yolo_dataset/ folder")
    ap.add_argument("--out-json")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--sample-n", type=int, default=SAMPLE_N, help="0 = all paired objects")
    args = ap.parse_args()

    if args.mode == "inspect":
        run_inspect(args.run_dir)
    else:
        if not args.out_json or not args.dataset:
            sys.exit("--out-json and --dataset are required in build mode")
        run_build(args.run_dir, args.out_json, args.dataset, args.sample_n)


if __name__ == "__main__":
    main()