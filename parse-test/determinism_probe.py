"""DE2 determinism probe.

Sends the same image N times to the running DE2 server and reports drift.
Also runs a cache-contamination test (image A -> B -> A) to detect
whether the perceptual hash cache leaks state across requests.

Outputs:
  - after/<image>_run<i>.png      annotated image returned by server
  - after/<image>_run<i>.json     parsed_content_list returned by server
  - after/<image>_summary.json    drift summary across N runs
  - after/contamination.json      cross-image cache leak report
  - after/_REPORT.md              human-readable rollup
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_SERVER = "http://localhost:8000"
HERE = Path(__file__).parent
BEFORE = HERE / "before"
AFTER = HERE / "after"


def b64_of(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_once(server: str, b64: str, overrides: dict[str, Any] | None = None) -> dict:
    payload: dict[str, Any] = {"base64_image": b64}
    if overrides:
        payload.update(overrides)
    r = requests.post(f"{server}/parse/", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def cache_stats(server: str) -> dict:
    try:
        r = requests.get(f"{server}/cache_stats/", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def quantize_bbox(bbox: list[float], q: int = 200) -> tuple[int, int, int, int]:
    """Quantize ratio-coords (0-1) to integer ticks at 1/q precision.

    q=200 means each tick is image_dim/200 pixels — for a 1920-wide image that's
    ~9.6 px. Sub-pixel jitter (typical of fp16 attention reduction-order
    nondeterminism) doesn't move things across a tick, so we don't false-flag
    it as drift. The agent never cares about <10 px bbox accuracy.
    """
    return tuple(int(round(v * q)) for v in bbox)


def fingerprint_element(elem: dict) -> tuple:
    """Order-independent fingerprint of a single parsed element.
    type + content + quantized bbox.
    """
    return (
        elem.get("type", "?"),
        (elem.get("content") or "").strip(),
        quantize_bbox(elem["bbox"]),
        elem.get("source", "?"),
    )


def diff_runs(runs: list[list[dict]]) -> dict:
    """Compare N runs of parsed_content_list.

    Reports:
      - per-run element count
      - is element-set identical across all runs (set of fingerprints equal)?
      - element-content drift: same bboxes, different text?
      - bbox-coord drift: same content, slightly different bboxes?
    """
    counts = [len(r) for r in runs]
    fps_per_run = [set(fingerprint_element(e) for e in r) for r in runs]
    all_identical = all(fps == fps_per_run[0] for fps in fps_per_run[1:])

    # union & intersection
    union = set().union(*fps_per_run)
    inter = set.intersection(*fps_per_run) if fps_per_run else set()
    only_in_run: list[list[tuple]] = []
    for i, s in enumerate(fps_per_run):
        only_in_run.append(sorted(s - inter))

    # bbox-only drift: same (type, content) but different quantized bbox
    by_text: list[dict[tuple, set]] = []
    for r in runs:
        d: dict[tuple, set] = {}
        for e in r:
            k = (e.get("type", "?"), (e.get("content") or "").strip())
            d.setdefault(k, set()).add(quantize_bbox(e["bbox"]))
        by_text.append(d)
    text_keys_union: set = set()
    for d in by_text:
        text_keys_union |= d.keys()
    text_drift = []
    for tk in sorted(text_keys_union):
        boxes_per_run = [d.get(tk, set()) for d in by_text]
        # if any run has it and any other doesn't, OR boxes differ
        seen_in = sum(1 for b in boxes_per_run if b)
        all_box_sets_equal = all(b == boxes_per_run[0] for b in boxes_per_run[1:])
        if seen_in < len(runs) or not all_box_sets_equal:
            text_drift.append(
                {
                    "type": tk[0],
                    "content": tk[1],
                    "boxes_per_run": [sorted(b) for b in boxes_per_run],
                    "seen_in_runs": seen_in,
                }
            )

    # content drift at SAME bbox: same (type, quantized bbox) but different content
    by_bbox: list[dict[tuple, set]] = []
    for r in runs:
        d: dict[tuple, set] = {}
        for e in r:
            k = (e.get("type", "?"), quantize_bbox(e["bbox"]))
            d.setdefault(k, set()).add((e.get("content") or "").strip())
        by_bbox.append(d)
    bbox_keys_union: set = set()
    for d in by_bbox:
        bbox_keys_union |= d.keys()
    content_drift = []
    for bk in sorted(bbox_keys_union):
        contents_per_run = [d.get(bk, set()) for d in by_bbox]
        all_equal = all(c == contents_per_run[0] for c in contents_per_run[1:])
        if not all_equal:
            content_drift.append(
                {
                    "type": bk[0],
                    "bbox_q": list(bk[1]),
                    "content_per_run": [sorted(c) for c in contents_per_run],
                }
            )

    return {
        "n_runs": len(runs),
        "element_counts_per_run": counts,
        "element_count_stable": len(set(counts)) == 1,
        "all_runs_identical": all_identical,
        "union_size": len(union),
        "intersection_size": len(inter),
        "elements_only_in_some_runs": [
            {"run_index": i, "count": len(only), "items": [list(t) for t in only]}
            for i, only in enumerate(only_in_run)
            if only
        ],
        "text_drift_count": len(text_drift),
        "text_drift_examples": text_drift[:25],
        "content_drift_count": len(content_drift),
        "content_drift_examples": content_drift[:25],
    }


def save_run_outputs(image_stem: str, run_idx: int, response: dict) -> None:
    AFTER.mkdir(parents=True, exist_ok=True)
    # annotated image
    som_b64 = response.get("som_image_base64", "")
    if som_b64:
        out_png = AFTER / f"{image_stem}_run{run_idx:02d}.png"
        out_png.write_bytes(base64.b64decode(som_b64))
    # parsed content
    out_json = AFTER / f"{image_stem}_run{run_idx:02d}.json"
    out_json.write_text(
        json.dumps(
            {
                "parsed_content_list": response.get("parsed_content_list", []),
                "latency": response.get("latency"),
                "config_used": response.get("config_used"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def count_kinds(elems: list[dict]) -> dict:
    """Count elements by type+source so we can spot Paddle/Florence regressions."""
    out: dict[str, int] = {}
    for e in elems:
        # source is set in get_som_labeled_img; if missing the element came from
        # an unusual code path. Treat that as ambiguous.
        src = e.get("source") or "?"
        key = f"{e.get('type','?')}:{src}"
        out[key] = out.get(key, 0) + 1
    return out


def repeat_test(
    server: str,
    image_path: Path,
    n_runs: int,
    overrides: dict[str, Any] | None,
    save_images: bool = True,
) -> dict:
    print(f"\n=== {image_path.name} : {n_runs} sequential runs ===")
    b64 = b64_of(image_path)
    runs: list[list[dict]] = []
    latencies: list[float] = []
    cache_progression: list[dict] = []
    kind_progression: list[dict] = []
    for i in range(n_runs):
        cache_progression.append(cache_stats(server))
        t0 = time.time()
        resp = parse_once(server, b64, overrides)
        dt = time.time() - t0
        latencies.append(dt)
        elems = resp.get("parsed_content_list", [])
        kinds = count_kinds(elems)
        kind_progression.append(kinds)
        print(
            f"  run {i+1}/{n_runs}: {len(elems)} elements, kinds={kinds}, "
            f"server-latency={resp.get('latency'):.2f}s, wall={dt:.2f}s"
        )
        runs.append(elems)
        if save_images:
            save_run_outputs(image_path.stem, i + 1, resp)
    cache_progression.append(cache_stats(server))

    summary = diff_runs(runs)
    summary["image"] = image_path.name
    summary["latencies_s"] = latencies
    summary["cache_progression"] = cache_progression
    summary["kind_progression"] = kind_progression
    # Detect Paddle dropouts: any run with zero `box_ocr_content_ocr` text/icon items
    summary["paddle_dropouts"] = [
        i
        for i, k in enumerate(kind_progression)
        if not any("box_ocr_content_ocr" in key for key in k.keys())
    ]
    summary["overrides"] = overrides or {}
    out = AFTER / f"{image_path.stem}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Console rollup
    print(
        f"  [SUMMARY] count_stable={summary['element_count_stable']}, "
        f"all_identical={summary['all_runs_identical']}, "
        f"text_drift={summary['text_drift_count']}, "
        f"content_drift={summary['content_drift_count']}, "
        f"union={summary['union_size']} / inter={summary['intersection_size']}"
    )
    return summary


def contamination_test(
    server: str,
    images: list[Path],
    overrides: dict[str, Any] | None,
) -> dict:
    """A->B->A->B->A interleave. Compare run1 vs run3 of image A:
    if cache leak from B affects A, the second A parse will differ.
    """
    print("\n=== CONTAMINATION TEST: A -> B -> A -> B -> A ===")
    if len(images) < 2:
        return {"skipped": "need >= 2 images"}
    a, b = images[0], images[1]
    seq = [a, b, a, b, a]
    parses: dict[str, list[list[dict]]] = {a.name: [], b.name: []}
    for i, img in enumerate(seq):
        b64 = b64_of(img)
        resp = parse_once(server, b64, overrides)
        elems = resp.get("parsed_content_list", [])
        parses[img.name].append(elems)
        save_run_outputs(f"contam_{i+1:02d}_{img.stem}", i + 1, resp)
        print(f"  step {i+1}: {img.name} -> {len(elems)} elements")

    report: dict = {"sequence": [p.name for p in seq]}
    for name, runs in parses.items():
        report[name] = diff_runs(runs)
    out = AFTER / "contamination.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  [CONTAM] A drift across 3 visits: text={report[a.name]['text_drift_count']}, "
          f"content={report[a.name]['content_drift_count']}")
    print(f"  [CONTAM] B drift across 2 visits: text={report[b.name]['text_drift_count']}, "
          f"content={report[b.name]['content_drift_count']}")
    return report


def write_markdown_report(summaries: list[dict], contam: dict | None) -> None:
    lines = []
    lines.append("# DE2 Determinism Probe — Results\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Per-image repeat test\n")
    for s in summaries:
        lines.append(f"### {s['image']}")
        lines.append(f"- runs: {s['n_runs']}")
        lines.append(f"- element counts: {s['element_counts_per_run']} "
                     f"(stable: {s['element_count_stable']})")
        lines.append(f"- ALL RUNS IDENTICAL: **{s['all_runs_identical']}**")
        lines.append(f"- union/intersection: {s['union_size']} / {s['intersection_size']}")
        lines.append(f"- text drift (same text, diff bboxes): {s['text_drift_count']}")
        lines.append(f"- content drift (same bbox, diff text): {s['content_drift_count']}")
        lines.append(f"- latencies: {[f'{x:.2f}s' for x in s['latencies_s']]}")
        if s["content_drift_examples"]:
            lines.append("\n  **Content drift examples (same bbox, server returned different text):**")
            for ex in s["content_drift_examples"][:10]:
                lines.append(
                    f"  - bbox≈{ex['bbox_q']} type={ex['type']} "
                    f"contents per run: {ex['content_per_run']}"
                )
        lines.append("")
    if contam:
        lines.append("## Cache contamination test (A B A B A)\n")
        for k, v in contam.items():
            if k == "sequence":
                lines.append(f"- sequence: {v}")
                continue
            if isinstance(v, dict):
                lines.append(f"### {k}")
                lines.append(f"- visits: {v.get('n_runs')}")
                lines.append(f"- ALL VISITS IDENTICAL: **{v.get('all_runs_identical')}**")
                lines.append(f"- text drift: {v.get('text_drift_count')}, "
                             f"content drift: {v.get('content_drift_count')}")
                lines.append("")
    (AFTER / "_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {AFTER / '_REPORT.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--images", nargs="*", default=None,
                    help="image filenames inside ./before. default = all tick_*.png")
    ap.add_argument("--no-contamination", action="store_true")
    ap.add_argument("--no-images", action="store_true",
                    help="don't save annotated PNGs (faster)")
    ap.add_argument("--box-threshold", type=float, default=None)
    ap.add_argument("--iou-threshold", type=float, default=None)
    ap.add_argument("--text-threshold", type=float, default=None)
    ap.add_argument("--qwen-full-ocr", action="store_true",
                    help="Send qwen_full_ocr=true on every request (Qwen detect+recognize)")
    args = ap.parse_args()

    AFTER.mkdir(parents=True, exist_ok=True)
    # cleanup prior run output (preserve hand-written findings)
    PRESERVE = {"_FINDINGS.md"}
    for p in AFTER.glob("*"):
        if p.is_file() and p.name not in PRESERVE:
            p.unlink()

    # health check
    r = requests.get(f"{args.server}/probe/", timeout=5)
    print(f"server probe: {r.status_code} {r.json()}")
    print(f"cache stats at start: {cache_stats(args.server)}")

    if args.images:
        images = [BEFORE / n for n in args.images]
    else:
        images = sorted(p for p in BEFORE.glob("tick_*.png"))
    images = [p for p in images if p.exists()]
    if not images:
        print("no images found in ./before", file=sys.stderr)
        return 2
    print(f"images: {[p.name for p in images]}")

    overrides: dict[str, Any] = {}
    if args.box_threshold is not None:
        overrides["box_threshold"] = args.box_threshold
    if args.iou_threshold is not None:
        overrides["iou_threshold"] = args.iou_threshold
    if args.text_threshold is not None:
        overrides["text_threshold"] = args.text_threshold
    if args.qwen_full_ocr:
        overrides["qwen_full_ocr"] = True
    print(f"overrides: {overrides}")

    summaries = []
    for img in images:
        s = repeat_test(
            args.server, img, args.n_runs, overrides, save_images=not args.no_images
        )
        summaries.append(s)

    contam = None
    if not args.no_contamination and len(images) >= 2:
        contam = contamination_test(args.server, images[:2], overrides)

    write_markdown_report(summaries, contam)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
