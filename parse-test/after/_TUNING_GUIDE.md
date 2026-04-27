# DE2 hybrid-mode tuning guide

The hybrid path (PaddleOCR detect + Qwen2.5-VL recognise) is the default and
the right choice for every screen except the rare cases where Paddle's text
detector clubs labels into one polygon. Hybrid is sub-second and deterministic
after the Phase-1 fixes.

This doc lists every per-request knob, what it actually does on the wire, and
the recipes that fix the failure modes we've seen on Cyberpunk / Civ VI / Wukong.

## All knobs exposed in the request body

```jsonc
POST /parse/
{
  "base64_image": "...",

  // ── YOLO icon detector ─────────────────────────────────────────
  "box_threshold":   0.05,   // confidence floor; lower → more icons
  "iou_threshold":   0.10,   // merge IoU; lower → more dedupe

  // ── PaddleOCR text-recognition confidence ──────────────────────
  // ONLY active in pure-Paddle mode. NO effect when use_qwen_ocr=True
  // (which is the default for our deployment), because Qwen does
  // recognition without a confidence threshold.
  "text_threshold":  0.5,

  // ── PaddleOCR text-DETECTION (DBNet) ───────────────────────────
  // Active in BOTH hybrid and pure-Paddle paths. These are what
  // actually control whether tab labels club together vs split.
  "det_db_thresh":         0.3,   // pixel binarisation
  "det_db_box_thresh":     0.6,   // min confidence to keep a box
  "det_db_unclip_ratio":   1.5,   // box expansion factor
  "det_db_score_mode":   "slow",  // "fast" or "slow"

  // ── Hybrid vs full Qwen ────────────────────────────────────────
  "use_paddleocr":   false,
  "use_qwen_ocr":    true,        // default
  "qwen_full_ocr":   false        // true → 10-60s, last resort

  // ── Plumbing (rarely tuned) ────────────────────────────────────
  "use_local_semantics": true,   // Florence-2 captions for icons
  "scale_img":           false,
  "imgsz":               null
}
```

## What changes when you move each knob — measured on tick_009.png (Cyberpunk Sound menu, the worst case)

| Knob | Default | Range tested | Element count | Top-tab bar split? | Slider clubbing? |
|---|---|---|---|---|---|
| `box_threshold` | 0.05 | 0.02 – 0.30 | 47 → 39 (fewer icons at high) | No | **Yes** — high values (0.20+) cut slider noise |
| `iou_threshold` | 0.10 | 0.05 – 0.50 | 47 → 48 | No | No effect |
| `det_db_unclip_ratio` | 1.5 | 1.0 – 2.0 | 46 – 47 | No | No effect |
| `det_db_box_thresh` | 0.6 | 0.3 – 0.7 | 46 – 47 | No | No effect |
| `det_db_thresh` | 0.3 | 0.2 – 0.5 | 47 | No | No effect |
| **`det_db_score_mode`** | "slow" | "fast" | **52** | **YES** ✓ | No |

**Summary**: only two knobs meaningfully change behaviour on this screen.

- `box_threshold` ↑ → fewer YOLO false positives on slider gradients (good for
  Cyberpunk Sound sliders, F1 telemetry overlays, etc.)
- `det_db_score_mode='fast'` → Paddle stops returning the long row-spanning
  polygon. The tab area falls through to YOLO+Florence-2 instead, which gives
  individually-clickable boxes. Trade-off: labels are Florence-2 captions
  ("Graphs" instead of "GRAPHICS", "Key Functions" instead of "KEY BINDINGS")
  — close enough for the agent's pattern matching but not bit-accurate OCR.

## Recipes by screen type

### 1. Plain dialogs (Civ VI Exit, Wukong "exit benchmark", typical pause menus)
**No overrides.** Default thresholds work — 6/6 elements deterministic, ~0.5s.

```yaml
# phases.yaml — no de2_overrides needed
- name: exit_dialog
```

### 2. Settings menus with horizontal tab bar (Cyberpunk-style)
Use **`det_db_score_mode: fast`** to split the tab row, and **`box_threshold: 0.20`**
to suppress slider-gradient YOLO noise. Net result: all 10 tabs individually
clickable, latency ~2.3s (vs 5.5s default and 60s qwen_full_ocr).

```yaml
- name: settings_menu
  de2_overrides:
    det_db_score_mode: fast
    box_threshold: 0.20
    det_db_unclip_ratio: 1.3   # tighter Paddle boxes
```

### 3. Heavy slider screens (volume / brightness / sensitivity sliders)
Just raise `box_threshold` to suppress YOLO over-detection on gradient bars.

```yaml
- name: volume_settings
  de2_overrides:
    box_threshold: 0.20    # or 0.30 if YOLO still over-detects
```

### 4. Last-resort: Paddle is missing critical labels
If raising knobs doesn't recover the missing label, fall back to full Qwen OCR
**for that one phase only**. Costs 10-60s but reads everything.

```yaml
- name: hard_screen
  de2_overrides:
    qwen_full_ocr: true
```

## Determinism guarantee

All knob combinations above were verified deterministic across multiple runs
in `threshold_sweep.py`. The Phase-1 executor patches (single-thread for YOLO
and OCR) hold — none of the threshold mutations break that.

## Where this plugs into rpx-core

`rpx-core/modules/omniparser_client.py` builds the request payload from the
phase's `de2_overrides` dict. Currently the client filters to a known key list
— before any of the new keys reach the server, you need to either:

1. Pass the dict through opaquely (recommended), or
2. Add the new keys to `omniparser_client.py`'s allowed list:
   ```
   det_db_thresh, det_db_box_thresh, det_db_unclip_ratio,
   det_db_score_mode, qwen_full_ocr
   ```

The server-side ParseRequest already accepts them (see
`omniparserserver.py:73-110`).

## Files in `parse-test/after/` from this run

```
sweep_tick_009.csv                  Sweep summary
sweep_tick_009_*.png                Annotated PNG per variant (26 variants)
sweep_tick_009_*.json               Parsed list per variant
sweep_tick_009_BEST_combo.png       score_fast + box_thr_0.20 + unclip_1.3
sweep_tick_011.csv                  Same sweep on Civ VI Exit (no variation)
threshold_sweep.py                  The sweep tool — re-run any time
```

## Reproducing

```
cd F:/Raptor-X-V2/omniparser-server/parse-test
python threshold_sweep.py --image tick_009.png   # Cyberpunk
python threshold_sweep.py --image tick_011.png   # Civ VI Exit
python threshold_sweep.py --image tick_008.png   # Wukong dialog
```

CSV gives raw numbers; PNGs let you eyeball the bbox layout difference.

## What the request body looks like end-to-end

```bash
curl -s http://localhost:8000/parse/ \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg b64 "$(base64 -w0 cyberpunk_sound.png)" \
        '{base64_image: $b64,
          box_threshold: 0.20,
          det_db_score_mode: "fast",
          det_db_unclip_ratio: 1.3}')" \
  | jq '.parsed_content_list | length'
# → 45  (vs 47 default)
```

## What is NOT in this guide and why

- `qwen_full_ocr: true` is documented in `_FINDINGS.md`. It works, it's
  bit-deterministic, but it's slow (10-60s on dense screens, max 11s on
  small dialogs). Reach for it only when a phase repeatedly fails despite
  threshold tuning.
- AWQ-Int4 model swap (`QWEN_OCR_USE_AWQ=1`) is plumbed but did not actually
  speed up our stack — the sdpa kernel path doesn't unlock AWQ throughput
  on Windows. Skip it; it just halves VRAM with no latency benefit.
- flash_attention_2 broke determinism in our env and was no faster. Don't
  enable it.
