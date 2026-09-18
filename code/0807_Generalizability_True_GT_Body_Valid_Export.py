#!/usr/bin/env python3
"""Bake body_valid = body AND NOT label_union AND NOT cap_union into YOLO-seg polygons.

Reads the true-GT labels written by 0804_Generalizability_ls_export_truegt.py and writes a
derived label set in which every body polygon has the union of all label and cap regions
of the same image removed. Label and cap lines are copied unchanged; labels_true_gt/ is
never modified. Only objects with true-GT status "accepted" are processed.

Inputs (per dataset in DATASET_SPECS):
    <yolo_dataset>/images/, labels_true_gt/, export_index.csv or true_gt_meta.csv

Outputs:
    <yolo_dataset>/labels_true_gt_valid/<stem>.txt
    RESULTS_DIR: per-dataset and pooled summary CSVs, JSON manifest

Usage:
    python 0807_Generalizability_True_GT_Body_Valid_Export.py
        [--datasets spectralwaste warp packwise] [--mode inspect|export] [--dry-run]
"""

# --- Imports ---
import os

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- Config ---
PROJECT_NAME = "20260826_True_GT_Body_Valid_Export"

PROJECT_ROOT  = Path(os.environ.get("KI_SORT_ROOT", "/VOL_D/fio.h")) / "KI_Sort_Project"
DATASETS_ROOT = Path(os.environ.get("KI_SORT_ROOT", "/VOL_D/fio.h")) / "KI_Sort_Datasets"
OUTPUT_DIR    = PROJECT_ROOT / "08_Generalizability"
RESULTS_DIR   = OUTPUT_DIR / "results" / PROJECT_NAME

CLASS_MAP   = {"body": 0, "cap": 1, "label": 2}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
REGIONS     = ["body", "cap", "label"]

# Regions subtracted from body; body_valid = body AND NOT their union.
SUBTRACT_REGIONS = ["label", "cap"]

TRUE_LABELS_DIRNAME  = "labels_true_gt"
VALID_LABELS_DIRNAME = "labels_true_gt_valid"
EXPORT_INDEX_NAME    = "export_index.csv"
TRUE_GT_META_NAME    = "true_gt_meta.csv"

# export_index.csv column holding the object stem. None triggers stem-overlap detection.
INDEX_KEY_COL = None

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Contour re-vectorisation. RETR_EXTERNAL discards interior holes and keeps the outer
# ring; disconnected fragments above MIN_POLY_AREA_PX are each written as their own line.
POLY_EPSILON_FRAC = 0.002      # approxPolyDP epsilon as a fraction of contour arc length
MIN_POLY_AREA_PX  = 24         # fragments below this pixel area are dropped
MIN_POLY_POINTS   = 3
COORD_DECIMALS    = 6

# A re-vectorised body counts as geometrically unchanged above this IoU with the input.
UNCHANGED_IOU = 0.999

ALLOW_OVERWRITE = True
LOG_EVERY       = 100

# Per-dataset paths. run_id None resolves to the newest run under <root>/data/.
DATASET_SPECS = {
    "spectralwaste": {
        "display": "SpectralWaste",
        "root":    DATASETS_ROOT / "Spectral_Waste"      / "KI_Sort_Preprocessing",
        "run_id":  None,
    },
    "warp": {
        "display": "WaRP",
        "root":    DATASETS_ROOT / "WaRP_Dataset"        / "KI_Sort_Preprocessing",
        "run_id":  None,
    },
    "packwise": {
        "display": "PackWISE",
        "root":    DATASETS_ROOT / "PackWISE_Dataset_v2" / "KI_Sort_Preprocessing",
        "run_id":  None,
    },
}

SUMMARY_COLUMNS = [
    "dataset", "display", "run_id", "cutout_id", "width", "height",
    "n_body_in", "n_body_out", "n_label", "n_cap",
    "body_px_raw", "body_px_valid", "body_px_removed", "removed_frac",
    "n_fragments", "n_dropped_slivers", "outcome",
]

OUTCOMES = ["unchanged", "hole_refilled", "notch_corrected",
            "fragmented", "dropped_full_overlap", "no_body"]


# --- Helpers ---
def utcnow():
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_has_images(run_dir):
    images_dir = run_dir / "yolo_dataset" / "images"
    return images_dir.is_dir() and any(
        p.suffix.lower() in IMAGE_EXTENSIONS for p in images_dir.iterdir())


def resolve_run_dir(spec):
    """Return the run directory holding the yolo_dataset folder for a dataset spec.

    When run_id is None, picks the newest run whose yolo_dataset/images is populated;
    newer runs without images are skipped.
    """
    data_root = Path(spec["root"]) / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing data root: {data_root}")
    if spec["run_id"] is None:
        candidates = sorted([p for p in data_root.iterdir()
                             if p.is_dir() and (p / "yolo_dataset").is_dir()])
        if not candidates:
            raise FileNotFoundError(f"no run with a yolo_dataset under {data_root}")
        runs = [p for p in candidates if _run_has_images(p)]
        if not runs:
            raise FileNotFoundError(
                f"no run under {data_root} has a populated yolo_dataset/images "
                f"(newest candidate: {candidates[-1].name})")
        empty_newer = [p.name for p in candidates if p not in runs
                       and p.name > runs[-1].name]
        if empty_newer:
            print(f"note: skipping empty/unpopulated run(s) newer than "
                  f"{runs[-1].name}: {empty_newer}")
        return runs[-1]
    return data_root / spec["run_id"]


def resolve_dataset(key, spec):
    """Fill in the image, true-GT, output and index paths for one dataset."""
    resolved = dict(spec)
    run_dir = resolve_run_dir(spec)
    yolo_root = run_dir / "yolo_dataset"
    resolved["key"]        = key
    resolved["run_id"]     = run_dir.name
    resolved["yolo_root"]  = yolo_root
    resolved["images_dir"] = yolo_root / "images"
    resolved["labels_dir"] = yolo_root / TRUE_LABELS_DIRNAME
    resolved["output_dir"] = yolo_root / VALID_LABELS_DIRNAME
    candidates = [yolo_root / EXPORT_INDEX_NAME,
                  Path(spec["root"]) / "results" / run_dir.name / "tables"
                  / EXPORT_INDEX_NAME]
    resolved["index_csv"] = next((p for p in candidates if p.is_file()), candidates[0])
    resolved["meta_csv"]  = yolo_root / TRUE_GT_META_NAME
    return resolved


def list_images(images_dir):
    d = Path(images_dir)
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])


def image_path_for(images_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        p = Path(images_dir) / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def as_bool(value):
    """Interpret a CSV cell as a boolean."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in ("true", "1", "yes", "t")


def as_str_series(series):
    """Coerce a column to plain strings, mapping every missing value to an empty string."""
    return series.map(lambda v: "" if pd.isna(v) else str(v))


def detect_index_key(idx_df, stems, override=None):
    """Return the export_index.csv column whose values match the most object stems."""
    if override:
        if override not in idx_df.columns:
            raise KeyError(f"INDEX_KEY_COL {override!r} not in {EXPORT_INDEX_NAME}")
        return override
    best, best_hits = None, -1
    for col in idx_df.columns:
        hits = sum(1 for v in as_str_series(idx_df[col])
                   if v and (v in stems or Path(v).stem in stems))
        if hits > best_hits:
            best, best_hits = col, hits
    return best if best_hits > 0 else None


def load_accepted_stems(resolved):
    """Resolve the accepted true-GT object stems for one dataset."""
    state = {"available": False, "source": None, "key_col": None,
             "accepted": [], "n_images": 0, "status_counts": {}, "notes": []}
    image_stems = {p.stem for p in list_images(resolved["images_dir"])}
    state["n_images"] = len(image_stems)
    if not image_stems:
        state["notes"].append(f"no images under {resolved['images_dir']}")
        return state

    status_by_stem = {}

    idx_path = resolved["index_csv"]
    if Path(idx_path).is_file():
        idx = pd.read_csv(idx_path)
        if "true_gt_status" in idx.columns or "true_gt_keep" in idx.columns:
            key_col = detect_index_key(idx, image_stems, INDEX_KEY_COL)
            if key_col is None:
                state["notes"].append(f"no {EXPORT_INDEX_NAME} column matches any stem")
            else:
                state["key_col"] = key_col
                state["source"] = f"{EXPORT_INDEX_NAME} [{key_col}]"
                stems_col = as_str_series(idx[key_col]).apply(
                    lambda v: v if v in image_stems else Path(v).stem)
                statuses = (as_str_series(idx["true_gt_status"])
                            if "true_gt_status" in idx.columns
                            else idx["true_gt_keep"].apply(
                                lambda v: "accepted" if as_bool(v) else "rejected"))
                for stem, status in zip(stems_col, statuses):
                    status_by_stem[stem] = status.strip() or "unannotated"
    else:
        state["notes"].append(f"{EXPORT_INDEX_NAME} not found at {idx_path}")

    meta_path = resolved["meta_csv"]
    if not status_by_stem and Path(meta_path).is_file():
        meta = pd.read_csv(meta_path)
        state["source"] = TRUE_GT_META_NAME
        stems_col  = as_str_series(meta["stem"]) if "stem" in meta.columns else []
        status_col = as_str_series(meta["status"]) if "status" in meta.columns else []
        for i, stem in enumerate(stems_col):
            stem = stem.strip()
            if not stem:
                continue
            status_by_stem[stem] = (status_col.iloc[i].strip()
                                    if len(status_col) else "") or "unannotated"

    if not status_by_stem:
        state["notes"].append("no true-GT status found; Route B has not been run")
        return state

    state["status_counts"] = dict(Counter(status_by_stem.values()))
    label_dir = Path(resolved["labels_dir"])
    accepted, missing = [], 0
    for stem, status in sorted(status_by_stem.items()):
        if status != "accepted":
            continue
        if stem in image_stems and (label_dir / f"{stem}.txt").is_file():
            accepted.append(stem)
        else:
            missing += 1
    if missing:
        state["notes"].append(f"{missing} accepted objects lack an image or a "
                              f"{TRUE_LABELS_DIRNAME} file and are skipped")
    state["accepted"] = accepted
    state["available"] = bool(accepted)
    return state


def parse_yolo_seg(txt_path, width, height):
    """Parse a YOLO polygon segmentation file into (class_id, pixel polygon) pairs."""
    items = []
    path = Path(txt_path)
    if not path.is_file():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = np.asarray(parts[1:], dtype=np.float64)
        if coords.size % 2:
            coords = coords[:-1]
        poly = coords.reshape(-1, 2) * np.array([width, height], dtype=np.float64)
        if poly.shape[0] < MIN_POLY_POINTS:
            continue
        items.append((cls, poly))
    return items


def polygon_to_mask(poly, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.asarray(poly, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def mask_to_polygons(mask):
    """Re-vectorise a boolean mask into simplified outer-contour polygons.

    Returns (polygons, n_dropped_slivers). RETR_EXTERNAL keeps only outer boundaries, so
    interior holes are discarded and disconnected fragments are returned separately.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys, dropped = [], 0
    for contour in contours:
        if cv2.contourArea(contour) < MIN_POLY_AREA_PX:
            dropped += 1
            continue
        eps = POLY_EPSILON_FRAC * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if approx.shape[0] < MIN_POLY_POINTS:
            approx = contour.reshape(-1, 2)
        if approx.shape[0] < MIN_POLY_POINTS:
            dropped += 1
            continue
        polys.append(approx.astype(np.float64))
    polys.sort(key=lambda p: cv2.contourArea(
        np.round(p).astype(np.int32).reshape(-1, 1, 2)), reverse=True)
    return polys, dropped


def format_line(cls_id, poly, width, height):
    """Format one YOLO-seg line from a pixel polygon."""
    norm = np.asarray(poly, dtype=np.float64) / np.array([width, height], dtype=np.float64)
    norm = np.clip(norm, 0.0, 1.0)
    coords = " ".join(f"{v:.{COORD_DECIMALS}f}" for v in norm.reshape(-1))
    return f"{cls_id} {coords}"


def classify_outcome(n_body_in, n_body_out, removed_px, iou_in_out, n_fragments):
    """Assign one of OUTCOMES to a corrected object."""
    if n_body_in == 0:
        return "no_body"
    if n_body_out == 0:
        return "dropped_full_overlap"
    if n_fragments > n_body_in:
        return "fragmented"
    if removed_px == 0:
        return "unchanged"
    if iou_in_out >= UNCHANGED_IOU:
        return "hole_refilled"
    return "notch_corrected"


def process_object(stem, resolved):
    """Correct one object's body polygons. Returns (lines, summary row) or (None, row)."""
    img_path = image_path_for(resolved["images_dir"], stem)
    img = cv2.imread(str(img_path)) if img_path is not None else None
    if img is None:
        return None, {"cutout_id": stem, "outcome": "unreadable_image"}
    h, w = img.shape[:2]

    items = parse_yolo_seg(Path(resolved["labels_dir"]) / f"{stem}.txt", w, h)
    by_region = {r: [] for r in REGIONS}
    for cls, poly in items:
        region = CLASS_NAMES.get(int(cls))
        if region is not None:
            by_region[region].append(poly)

    body_masks = [polygon_to_mask(p, w, h) for p in by_region["body"]]
    subtract_union = np.zeros((h, w), dtype=bool)
    for region in SUBTRACT_REGIONS:
        for poly in by_region[region]:
            subtract_union |= polygon_to_mask(poly, w, h)

    body_lines = []
    px_raw = px_valid = 0
    n_fragments = n_dropped = 0
    iou_values = []
    for mask in body_masks:
        px_raw += int(mask.sum())
        valid = mask & ~subtract_union
        px_valid += int(valid.sum())
        if not valid.any():
            continue
        polys, dropped = mask_to_polygons(valid)
        n_dropped += dropped
        if not polys:
            continue
        n_fragments += len(polys)
        rebuilt = np.zeros((h, w), dtype=bool)
        for poly in polys:
            rebuilt |= polygon_to_mask(poly, w, h)
            body_lines.append(format_line(CLASS_MAP["body"], poly, w, h))
        iou_values.append(mask_iou(mask, rebuilt))

    other_lines = []
    for region in ("cap", "label"):
        for poly in by_region[region]:
            other_lines.append(format_line(CLASS_MAP[region], poly, w, h))

    removed_px = px_raw - px_valid
    iou_in_out = float(min(iou_values)) if iou_values else 0.0
    row = {
        "dataset": resolved["key"], "display": resolved["display"],
        "run_id": resolved["run_id"], "cutout_id": stem,
        "width": w, "height": h,
        "n_body_in": len(body_masks),
        "n_body_out": len(body_lines),
        "n_label": len(by_region["label"]),
        "n_cap": len(by_region["cap"]),
        "body_px_raw": px_raw,
        "body_px_valid": px_valid,
        "body_px_removed": removed_px,
        "removed_frac": float(removed_px) / px_raw if px_raw else 0.0,
        "n_fragments": n_fragments,
        "n_dropped_slivers": n_dropped,
        "outcome": classify_outcome(len(body_masks), len(body_lines), removed_px,
                                    iou_in_out, n_fragments),
    }
    return body_lines + other_lines, row


def export_dataset(resolved, state, dry_run=False):
    """Write labels_true_gt_valid/ for one dataset and return the summary frame."""
    out_dir = Path(resolved["output_dir"])
    if out_dir.exists() and not ALLOW_OVERWRITE and any(out_dir.iterdir()):
        raise RuntimeError(f"{out_dir} is not empty and ALLOW_OVERWRITE is False")
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stems = state["accepted"]
    rows, t0 = [], time.time()
    for i, stem in enumerate(stems, start=1):
        lines, row = process_object(stem, resolved)
        rows.append(row)
        if lines is not None and not dry_run:
            (out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        if i % LOG_EVERY == 0 or i == len(stems):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0.0
            print(f"[{resolved['display']}] {i}/{len(stems)} "
                  f"({100.0 * i / max(len(stems), 1):5.1f}%) | "
                  f"elapsed {elapsed:6.1f}s | {rate:5.1f} obj/s", flush=True)
    return pd.DataFrame(rows), time.time() - t0


def print_dataset_report(resolved, df, elapsed, indent="  "):
    counts = Counter(df["outcome"])
    print(f"{indent}objects written      : {int((df['n_body_out'] > 0).sum())}")
    print(f"{indent}elapsed              : {elapsed:.1f}s")
    for outcome in OUTCOMES:
        n = counts.get(outcome, 0)
        if n:
            print(f"{indent}  {outcome:<22}: {n:>5}  "
                  f"({100.0 * n / max(len(df), 1):5.1f}%)")
    for outcome, n in sorted(counts.items()):
        if outcome not in OUTCOMES:
            print(f"{indent}  {outcome:<22}: {n:>5}")
    corrected = df[df["body_px_removed"] > 0]
    if len(corrected):
        print(f"{indent}removed body area    : mean {corrected['removed_frac'].mean():.4f}  "
              f"median {corrected['removed_frac'].median():.4f}  "
              f"max {corrected['removed_frac'].max():.4f}")
    n_frag = int((df["n_body_out"] > df["n_body_in"]).sum())
    if n_frag:
        print(f"{indent}objects split into extra body instances: {n_frag}")
    n_sliver = int(df["n_dropped_slivers"].sum())
    if n_sliver:
        print(f"{indent}fragments dropped below {MIN_POLY_AREA_PX} px2: {n_sliver}")


# --- Main ---
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_SPECS),
                        choices=list(DATASET_SPECS),
                        help="subset of datasets to process")
    parser.add_argument("--mode", choices=["inspect", "export"], default="export",
                        help="inspect resolves paths and counts only, writing nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="run the full correction but write no label files")
    args = parser.parse_args(argv)

    dry_run = args.dry_run or args.mode == "inspect"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print(f"{PROJECT_NAME}   mode={args.mode}  dry_run={dry_run}")
    print(f"body_valid = body AND NOT {' AND NOT '.join(SUBTRACT_REGIONS)}")
    print("=" * 78)

    resolved_all, states = {}, {}
    for key in args.datasets:
        try:
            resolved = resolve_dataset(key, DATASET_SPECS[key])
        except FileNotFoundError as exc:
            print(f"{DATASET_SPECS[key]['display']:<14} SKIPPED: {exc}")
            continue
        state = load_accepted_stems(resolved)
        resolved_all[key], states[key] = resolved, state
        print(f"{resolved['display']:<14} run_id={resolved['run_id']}  "
              f"images={state['n_images']:>4}  accepted={len(state['accepted']):>4}  "
              f"source={state['source']}")
        print(f"{'':<14} labels in  : {resolved['labels_dir']}")
        print(f"{'':<14} labels out : {resolved['output_dir']}")
        for note in state["notes"]:
            print(f"{'':<14} note: {note}")

    active = [k for k in resolved_all if states[k]["available"]]
    if not active:
        print("\nno dataset has accepted true-GT objects; run Route B first")
        return 1
    if args.mode == "inspect":
        print(f"\ninspect only: {[resolved_all[k]['display'] for k in active]}")
        return 0

    frames, manifest = [], {}
    for key in active:
        resolved, state = resolved_all[key], states[key]
        print(f"\n=== {resolved['display']} === {len(state['accepted'])} accepted objects")
        df, elapsed = export_dataset(resolved, state, dry_run=dry_run)
        print_dataset_report(resolved, df, elapsed)
        frames.append(df)
        summary_path = RESULTS_DIR / f"{PROJECT_NAME}_{key}_summary_{stamp}.csv"
        df.to_csv(summary_path, index=False)
        print(f"  summary written      : {summary_path.name}")
        manifest[key] = {
            "display": resolved["display"],
            "run_id": resolved["run_id"],
            "images_dir": str(resolved["images_dir"]),
            "labels_dir": str(resolved["labels_dir"]),
            "output_dir": str(resolved["output_dir"]),
            "index_csv": str(resolved["index_csv"]),
            "status_source": state["source"],
            "n_accepted": len(state["accepted"]),
            "n_written": int((df["n_body_out"] > 0).sum()),
            "outcomes": {k: int(v) for k, v in Counter(df["outcome"]).items()},
            "elapsed_seconds": float(elapsed),
        }

    pooled = pd.concat(frames, ignore_index=True)
    pooled_path = RESULTS_DIR / f"{PROJECT_NAME}_summary_all_{stamp}.csv"
    pooled.to_csv(pooled_path, index=False)

    manifest_path = RESULTS_DIR / f"{PROJECT_NAME}_manifest_{stamp}.json"
    manifest_path.write_text(json.dumps({
        "project": PROJECT_NAME,
        "timestamp_utc": utcnow(),
        "dry_run": dry_run,
        "params": {
            "subtract_regions": SUBTRACT_REGIONS,
            "class_map": CLASS_MAP,
            "poly_epsilon_frac": POLY_EPSILON_FRAC,
            "min_poly_area_px": MIN_POLY_AREA_PX,
            "coord_decimals": COORD_DECIMALS,
            "unchanged_iou": UNCHANGED_IOU,
            "fragment_policy": "keep_all_above_min_area",
            "output_dirname": VALID_LABELS_DIRNAME,
        },
        "datasets": manifest,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("POOLED")
    print("=" * 78)
    print_dataset_report(None, pooled, 0.0)
    print(f"\nwritten to {RESULTS_DIR}")
    if dry_run:
        print("dry run: no label files were written")
    return 0


if __name__ == "__main__":
    sys.exit(main())