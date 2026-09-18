#!/usr/bin/env python3
"""Convert a Label Studio JSON export of refined pseudo-GT into true-GT YOLO-seg labels.

Reads the Label Studio export of tasks created by
0805_Generalizability_ls_import_pseudogt.py and writes the results back into the
`yolo_dataset/` folder the tasks came from. Nothing is deleted from images/ or
labels_pseudo_gt/.

Outputs:
    <yolo_dataset>/labels_true_gt/<stem>.txt   YOLO-seg polygons, CLASS_MAP ids
    <yolo_dataset>/true_gt_meta.csv            per-object region stats + LS metadata
    <yolo_dataset>/true_gt_provenance.json     run record
    <yolo_dataset>/export_index.csv            updated in place, .bak written first

Columns appended to export_index.csv:
    true_gt_status        accepted | rejected | unannotated
    true_gt_keep          bool; True only for accepted
    true_gt_reason        exclusion reason, empty for accepted
    true_gt_label_path    relative path to the labels_true_gt file, empty if rejected
    true_gt_annotated_utc LS annotation timestamp

Downstream code filters on `true_gt_keep` to load only usable objects.

Body/label overlap: ENFORCE_LABEL_PRIORITY subtracts the label union from the body mask,
but YOLO-seg polygons cannot encode holes (cv2.RETR_EXTERNAL refills them), so the written
body polygon still covers the label area. `true_gt_meta.csv` records `body_px_raw` and
`body_px_valid`; consumers apply `body_valid = body AND NOT label_union` at load time
(0807_Generalizability_True_GT_Body_Valid_Export.py bakes this into a label set).
Objects without a body (label-only or cap-only crops) are accepted; only objects with no
regions at all are rejected.

Usage:
    python 0804_Generalizability_ls_export_truegt.py --export-json <ls_export.json> \
        --run-dir <yolo_dataset> --dataset <name> [--index-key-col COL] [--no-index-update]
"""

# --- Imports ---
import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from label_studio_sdk.converter.brush import decode_rle
except ImportError:
    from label_studio_converter.brush import decode_rle

# --- Config ---

CLASS_MAP = {"body": 0, "cap": 1, "label": 2}
REGIONS = ["body", "label", "cap"]

TRUE_LABELS_DIRNAME = "labels_true_gt"
EXPORT_INDEX_NAME = "export_index.csv"

FROM_NAME_BRUSH = "tag"
QC_EXCLUDE_TAG = "qc_exclude"
QC_REASON_TAG = "qc_reason"
QC_REASON_OTHER_TAG = "qc_reason_other"
QC_EXCLUDE_VALUE = "Exclude object"
QC_REASON_OTHER_VALUE = "other (see note)"

MIN_CONTOUR_AREA_PX = 200
ENFORCE_LABEL_PRIORITY = True

STATUS_COLS = ["true_gt_status", "true_gt_keep", "true_gt_reason",
               "true_gt_label_path", "true_gt_annotated_utc"]

# --- Helpers ---


def rle_to_mask(rle: list, height: int, width: int) -> np.ndarray:
    """Decode a Label Studio brush RLE into a boolean (height, width) mask."""
    flat = decode_rle(rle)
    return np.reshape(flat, [height, width, 4])[:, :, 3] > 0


def pick_annotation(task: dict):
    """Most recently updated annotation that is neither cancelled nor skipped, or None."""
    anns = [a for a in task.get("annotations", [])
            if not a.get("was_cancelled") and not a.get("skipped")]
    if not anns:
        return None
    return sorted(anns, key=lambda a: a.get("updated_at") or "")[-1]


def parse_result(result: list):
    """Extract per-region brush masks, QC exclude flag and QC reason from one annotation."""
    masks = {r: [] for r in REGIONS}
    qc_exclude, qc_reason, qc_reason_other = False, "", ""

    for item in result:
        from_name = item.get("from_name")
        value = item.get("value", {})

        if item.get("type") == "brushlabels" and from_name == FROM_NAME_BRUSH:
            labels = value.get("brushlabels") or []
            if not labels or value.get("format") != "rle":
                continue
            label = labels[0]
            if label not in masks:
                continue
            h = int(item["original_height"])
            w = int(item["original_width"])
            masks[label].append((rle_to_mask(value["rle"], h, w), h, w))

        elif from_name == QC_EXCLUDE_TAG:
            qc_exclude = QC_EXCLUDE_VALUE in (value.get("choices") or [])

        elif from_name == QC_REASON_TAG:
            choices = value.get("choices") or []
            qc_reason = choices[0] if choices else ""

        elif from_name == QC_REASON_OTHER_TAG:
            texts = value.get("text") or []
            qc_reason_other = texts[0].strip() if texts else ""

    if qc_reason == QC_REASON_OTHER_VALUE and qc_reason_other:
        qc_reason = qc_reason_other

    return masks, qc_exclude, qc_reason


def merge_masks(entries: list):
    """Union of (mask, h, w) entries of one region; returns (mask, h, w) or Nones."""
    if not entries:
        return None, None, None
    h, w = entries[0][1], entries[0][2]
    out = np.zeros((h, w), dtype=bool)
    for m, mh, mw in entries:
        if (mh, mw) != (h, w):
            raise ValueError(f"inconsistent region geometry: {(mh, mw)} vs {(h, w)}")
        out |= m
    return out, h, w


def mask_to_yolo_lines(mask: np.ndarray, class_id: int, w: int, h: int) -> list:
    """Outer contours above MIN_CONTOUR_AREA_PX as normalised YOLO-seg lines."""
    m = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA_PX:
            continue
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.extend([x / w, y / h])
        lines.append(str(class_id) + " " + " ".join(f"{c:.6f}" for c in coords))
    return lines


def detect_index_key(idx: pd.DataFrame, stems: set, override: str) -> str:
    """Return the export_index.csv column whose values match the most object stems."""
    if override:
        if override not in idx.columns:
            sys.exit(f"--index-key-col {override!r} not in {EXPORT_INDEX_NAME}")
        return override
    best, best_hits = None, -1
    for c in idx.columns:
        hits = sum(1 for v in idx[c].astype(str) if v in stems or Path(v).stem in stems)
        if hits > best_hits:
            best, best_hits = c, hits
    if best_hits <= 0:
        sys.exit(f"no column in {EXPORT_INDEX_NAME} matches any stem; pass --index-key-col")
    print(f"export_index key column: {best!r} ({best_hits}/{len(idx)} rows match)")
    return best


# --- Main ---


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-json", required=True)
    ap.add_argument("--run-dir", required=True, help="path to the yolo_dataset/ folder")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index-key-col", default="")
    ap.add_argument("--no-index-update", action="store_true",
                    help="write labels and meta only, leave export_index.csv untouched")
    args = ap.parse_args()

    root = Path(args.run_dir)
    out_labels = root / TRUE_LABELS_DIRNAME
    out_labels.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(args.export_json).read_text())

    rows = []
    n_unannotated = n_cancelled = 0

    for task in tasks:
        data = task.get("data", {})
        stem = data.get("stem")
        if not stem:
            continue

        if not task.get("annotations"):
            n_unannotated += 1
            continue

        ann = pick_annotation(task)
        if ann is None:
            n_cancelled += 1
            continue

        masks, qc_exclude, qc_reason = parse_result(ann.get("result", []))
        base = {
            "stem": stem,
            "dataset": args.dataset,
            "annotator_id": ann.get("completed_by"),
            "annotated_utc": ann.get("updated_at"),
            "image_path": data.get("image_path", ""),
            "pseudo_label_path": data.get("pseudo_label_path", ""),
        }

        if qc_exclude:
            rows.append({**base, "status": "rejected",
                         "reason": qc_reason or "unspecified",
                         "true_gt_label_path": ""})
            continue

        merged, h, w = {}, None, None
        for region in REGIONS:
            m, mh, mw = merge_masks(masks[region])
            if m is None:
                continue
            h, w = mh, mw
            merged[region] = m

        if h is None:
            rows.append({**base, "status": "rejected",
                         "reason": "no regions after refinement",
                         "true_gt_label_path": ""})
            continue

        for region in REGIONS:
            merged.setdefault(region, np.zeros((h, w), dtype=bool))

        body_px_raw = int(merged["body"].sum())
        if ENFORCE_LABEL_PRIORITY:
            merged["body"] = merged["body"] & ~merged["label"]

        lines = []
        for region in REGIONS:
            lines.extend(mask_to_yolo_lines(merged[region], CLASS_MAP[region], w, h))

        if not lines:
            rows.append({**base, "status": "rejected",
                         "reason": "all contours below MIN_CONTOUR_AREA_PX",
                         "true_gt_label_path": ""})
            continue

        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")

        rows.append({
            **base, "status": "accepted", "reason": "",
            "true_gt_label_path": f"{TRUE_LABELS_DIRNAME}/{stem}.txt",
            "img_w": w, "img_h": h,
            "body_px_raw": body_px_raw,
            "body_px_valid": int(merged["body"].sum()),
            "label_px": int(merged["label"].sum()),
            "cap_px": int(merged["cap"].sum()),
            "cap_present": bool(merged["cap"].any()),
            "n_yolo_polygons": len(lines),
        })

    meta = pd.DataFrame(rows)
    meta.to_csv(root / "true_gt_meta.csv", index=False)

    n_accepted = int((meta["status"] == "accepted").sum()) if not meta.empty else 0
    n_rejected = int((meta["status"] == "rejected").sum()) if not meta.empty else 0

    n_index_matched = 0
    idx_path = root / EXPORT_INDEX_NAME
    if not args.no_index_update and idx_path.exists() and not meta.empty:
        idx = pd.read_csv(idx_path)
        shutil.copy2(idx_path, str(idx_path) + ".bak")

        key_col = detect_index_key(idx, set(meta["stem"]), args.index_key_col)
        idx["_stem"] = idx[key_col].astype(str).apply(
            lambda v: v if v in set(meta["stem"]) else Path(v).stem)

        lut = meta.set_index("stem")
        idx = idx.drop(columns=[c for c in STATUS_COLS if c in idx.columns])
        idx["true_gt_status"] = idx["_stem"].map(lut["status"]).fillna("unannotated")
        idx["true_gt_keep"] = idx["true_gt_status"] == "accepted"
        idx["true_gt_reason"] = idx["_stem"].map(lut["reason"]).fillna("")
        idx["true_gt_label_path"] = idx["_stem"].map(lut["true_gt_label_path"]).fillna("")
        idx["true_gt_annotated_utc"] = idx["_stem"].map(lut["annotated_utc"]).fillna("")

        n_index_matched = int((idx["true_gt_status"] != "unannotated").sum())
        idx = idx.drop(columns=["_stem"])
        idx.to_csv(idx_path, index=False)

    prov = {
        "created_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": args.dataset,
        "yolo_dataset_root": str(root),
        "export_json": str(args.export_json),
        "true_labels_dir": TRUE_LABELS_DIRNAME,
        "n_tasks": len(tasks),
        "n_annotated": n_accepted + n_rejected,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_unannotated": n_unannotated,
        "n_cancelled": n_cancelled,
        "n_export_index_rows_updated": n_index_matched,
        "class_map": CLASS_MAP,
        "min_contour_area_px": MIN_CONTOUR_AREA_PX,
        "enforce_label_priority": ENFORCE_LABEL_PRIORITY,
        "index_updated": not args.no_index_update,
    }
    prov["config_hash"] = hashlib.sha1(
        json.dumps(prov, sort_keys=True).encode()).hexdigest()[:12]
    (root / "true_gt_provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"tasks               : {len(tasks)}")
    print(f"accepted            : {n_accepted}")
    print(f"rejected            : {n_rejected}")
    print(f"unannotated         : {n_unannotated}")
    print(f"cancelled (skipped) : {n_cancelled}")
    print(f"export_index updated: {n_index_matched} rows")
    if n_rejected:
        print("\nrejection reasons:")
        print(meta[meta["status"] == "rejected"]["reason"].value_counts().to_string())
    print(f"\nlabels : {out_labels}")
    print(f"meta   : {root / 'true_gt_meta.csv'}")


if __name__ == "__main__":
    main()