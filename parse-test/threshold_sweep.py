"""Sweep DE2 threshold knobs on the same image and dump results.

Tests each knob in isolation so the user can see what changes when only
that one knob moves. Saves an annotated PNG for each variant under
./after/sweep_<image>_<knobname>_<value>.png and a JSON of the parsed
content list, plus a summary CSV.

Usage:
    python threshold_sweep.py --image tick_009.png
    python threshold_sweep.py --image tick_011.png --runs 1
"""
import argparse
import base64
import csv
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
BEFORE = HERE / "before"
AFTER = HERE / "after"
SERVER = "http://localhost:8000"


def b64_of(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def parse_one(image_b64: str, overrides: dict) -> dict:
    payload = {"base64_image": image_b64}
    payload.update(overrides)
    r = requests.post(f"{SERVER}/parse/", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def save_outputs(stem: str, response: dict) -> dict:
    AFTER.mkdir(parents=True, exist_ok=True)
    if response.get("som_image_base64"):
        (AFTER / f"sweep_{stem}.png").write_bytes(
            base64.b64decode(response["som_image_base64"])
        )
    parsed = response.get("parsed_content_list", [])
    (AFTER / f"sweep_{stem}.json").write_text(
        json.dumps(parsed, indent=2, ensure_ascii=False)
    )
    n_text = sum(1 for e in parsed if e.get("type") == "text")
    n_icon = sum(1 for e in parsed if e.get("type") == "icon")
    return {"total": len(parsed), "text": n_text, "icon": n_icon}


# Knob sweeps — single-axis variations from default
SWEEPS = [
    # Baseline (no overrides) so we have a reference point
    {"label": "baseline", "overrides": {}},
    # YOLO box_threshold
    {"label": "box_thr_0.02", "overrides": {"box_threshold": 0.02}},
    {"label": "box_thr_0.05", "overrides": {"box_threshold": 0.05}},  # default
    {"label": "box_thr_0.10", "overrides": {"box_threshold": 0.10}},
    {"label": "box_thr_0.20", "overrides": {"box_threshold": 0.20}},
    {"label": "box_thr_0.30", "overrides": {"box_threshold": 0.30}},
    # Merge iou_threshold
    {"label": "iou_thr_0.05", "overrides": {"iou_threshold": 0.05}},
    {"label": "iou_thr_0.10", "overrides": {"iou_threshold": 0.10}},  # default
    {"label": "iou_thr_0.20", "overrides": {"iou_threshold": 0.20}},
    {"label": "iou_thr_0.50", "overrides": {"iou_threshold": 0.50}},
    # PaddleOCR detection: unclip_ratio (the one that controls clubbing)
    {"label": "unclip_1.0", "overrides": {"det_db_unclip_ratio": 1.0}},
    {"label": "unclip_1.3", "overrides": {"det_db_unclip_ratio": 1.3}},
    {"label": "unclip_1.5", "overrides": {"det_db_unclip_ratio": 1.5}},  # default
    {"label": "unclip_1.8", "overrides": {"det_db_unclip_ratio": 1.8}},
    {"label": "unclip_2.0", "overrides": {"det_db_unclip_ratio": 2.0}},
    # PaddleOCR detection: box_thresh
    {"label": "det_box_0.3", "overrides": {"det_db_box_thresh": 0.3}},
    {"label": "det_box_0.5", "overrides": {"det_db_box_thresh": 0.5}},
    {"label": "det_box_0.6", "overrides": {"det_db_box_thresh": 0.6}},  # default
    {"label": "det_box_0.7", "overrides": {"det_db_box_thresh": 0.7}},
    # PaddleOCR detection: thresh
    {"label": "det_thr_0.2", "overrides": {"det_db_thresh": 0.2}},
    {"label": "det_thr_0.3", "overrides": {"det_db_thresh": 0.3}},  # default
    {"label": "det_thr_0.5", "overrides": {"det_db_thresh": 0.5}},
    # PaddleOCR detection: score_mode
    {"label": "score_fast", "overrides": {"det_db_score_mode": "fast"}},
    {"label": "score_slow", "overrides": {"det_db_score_mode": "slow"}},
    # Promising combos for fixing tab-bar clubbing
    {"label": "split_combo_a", "overrides": {
        "det_db_unclip_ratio": 1.0, "det_db_box_thresh": 0.5}},
    {"label": "split_combo_b", "overrides": {
        "det_db_unclip_ratio": 1.0, "det_db_box_thresh": 0.4,
        "det_db_thresh": 0.2}},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="tick_009.png")
    ap.add_argument("--runs", type=int, default=1, help="repeat each setting N times")
    args = ap.parse_args()

    img_path = BEFORE / args.image
    if not img_path.exists():
        print(f"image not found: {img_path}")
        return 2

    AFTER.mkdir(parents=True, exist_ok=True)
    img_b64 = b64_of(img_path)

    rows = []
    print(f"\nSweeping {len(SWEEPS)} configurations on {args.image}\n")
    print(f"{'label':<20} {'total':>6} {'text':>5} {'icon':>5} {'latency':>8}  overrides")
    print("-" * 100)
    for cfg in SWEEPS:
        label = cfg["label"]
        overrides = cfg["overrides"]
        # Take the FIRST run for the saved PNG/JSON; if --runs>1, capture
        # element counts for each repetition to spot determinism issues.
        counts_per_run = []
        latency_per_run = []
        for i in range(args.runs):
            t0 = time.time()
            resp = parse_one(img_b64, overrides)
            wall = time.time() - t0
            stem = f"{img_path.stem}_{label}" + (f"_run{i+1}" if args.runs > 1 else "")
            stats = save_outputs(stem, resp)
            counts_per_run.append(stats["total"])
            latency_per_run.append(resp.get("latency", wall))
        # First-run stats for the row
        n_total = counts_per_run[0]
        n_text = sum(1 for e in resp.get("parsed_content_list", [])
                     if e.get("type") == "text")
        n_icon = sum(1 for e in resp.get("parsed_content_list", [])
                     if e.get("type") == "icon")
        latency = latency_per_run[0]
        det_status = "OK" if all(c == counts_per_run[0] for c in counts_per_run) else f"DRIFT {counts_per_run}"
        print(f"{label:<20} {n_total:>6} {n_text:>5} {n_icon:>5} {latency:>7.2f}s  {overrides}  [{det_status}]")
        rows.append({
            "image": args.image,
            "label": label,
            "overrides": json.dumps(overrides),
            "total": n_total,
            "text": n_text,
            "icon": n_icon,
            "latency_s": round(latency, 2),
            "runs_counts": json.dumps(counts_per_run),
            "deterministic": det_status,
        })

    # CSV summary
    csv_path = AFTER / f"sweep_{img_path.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {csv_path}")
    print(f"PNG/JSON for each variant: {AFTER}/sweep_{img_path.stem}_*.png/.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
