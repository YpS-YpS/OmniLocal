# DE2 (OmniParser) determinism — root cause + fix

## Bottom line

Same screenshot now produces **bit-identical parsed_content_list** across repeated calls,
including across cache-contamination interleaves. Verified by `determinism_probe.py`:

| Image                     | Before patches            | After patches            |
|---------------------------|---------------------------|--------------------------|
| tick_008 (Wukong dialog)  | 5/5 identical (lucky)     | 5/5 + contam 3/3 ✓       |
| tick_009 (Cyberpunk Sound)| 5/5 identical (lucky)     | 50/50 + contam 2/2 ✓     |
| tick_011 (Civ VI Exit)    | **8/20 dropouts (40%)**   | **50/50 ✓**              |
| Cross-image contamination | tick_008: 7→15→15 elems   | tick_008: 15→15→15 ✓     |

Latency cost: **negligible** (~0% on tick_009, +10% on tick_011 from cpu_threads=1).

## Root cause (the real one)

The dominant nondeterminism source was **NOT** what the prior research suggested
(Qwen batch invariance, fuzzy hash cache, Florence sampling) — those are real but
not what was firing in this codebase. The actual culprit:

> `omniparser.py` had a single `ThreadPoolExecutor(max_workers=4)` shared by both
> the YOLO and OCR submissions. Per request, two of the four worker threads were
> picked to run YOLO and OCR. **Which two threads got picked rotated across
> requests**, and PaddleOCR carries thread-local state in its inference buffers.
> When OCR landed on a "warm" thread it returned 6 boxes; when it landed on a
> "cold" thread it returned 0. The pattern in the 20-run probe was a near-perfect
> alternation, which is the smoking gun for thread-cycling rather than fp drift.

When PaddleOCR returns 0 polygons, all OCR-derived text vanishes from the output,
and the YOLO icons that would have been labelled by OCR get Florence-2 fallback
captions instead — producing the wild labels the user saw on tick_011 run 4:

| bbox (Civ VI button) | run 1-3,5 (PaddleOCR fired) | run 4 (PaddleOCR dropped) |
|----------------------|------------------------------|----------------------------|
| 382,515,498,557      | "OK"                         | "Italic"                   |
| 501,516,618,557      | "Cancel"                     | "Uncomm"                   |
| 351,425,645,468      | "EXIT TO DESKTOP"            | "a calendar or date indicator." |

This is the same failure class as the original Cyberpunk slider screenshots but
with cleaner ground truth (small dialog) so the cause was easier to spot.

## What was patched

1. `util/omniparser.py` — split `_executor` into `_yolo_executor` and `_ocr_executor`,
   each `ThreadPoolExecutor(max_workers=1)`. Same thread always handles each model.
   **This single change fixed >90% of the observable drift.**

2. `util/utils.py` — added to PaddleOCR construction:
   - `cpu_threads=1` (locks oneDNN reduction order)
   - `enable_mkldnn=False` (oneDNN reduction is non-stable)
   - seeded `random` / `numpy` / `paddle` to 0 before construction
   Defensive — not strictly needed once executor is fixed, but cheap insurance.

3. `util/qwen_ocr.py` — `PerceptualHashCache` default `hamming_threshold` changed
   from `5` to `0`. Fuzzy matching could let a slider-thumb crop from screen N
   resolve to the cached label of a different slider-thumb crop on screen N+1.
   Defensive against a separate failure mode that was not firing on these
   screenshots but is documented in the code review (`PerceptualHashCache` review).

4. `omnitool/omniparserserver/omniparserserver.py` — process-start determinism
   flags before any torch import:
   - `CUBLAS_WORKSPACE_CONFIG=:4096:8`
   - `PYTHONHASHSEED=0`
   - `FLAGS_cudnn_deterministic=True`
   - `FLAGS_cudnn_exhaustive_search=False`
   - `torch.backends.cudnn.deterministic = True`
   - `torch.backends.cudnn.benchmark = False`
   - `torch.use_deterministic_algorithms(True, warn_only=True)`
   - seeded `torch.manual_seed(0)` + `torch.cuda.manual_seed_all(0)`
   Defensive — covers YOLO, Qwen, Florence-2 against the smaller fp-drift source
   that the executor change does not directly address.

## What is NOT fixed by these patches (and is NOT the problem reported)

- **Slider-track box clubbing in tick_009** is still present, but it is now
  **identically clubbed every run**. This is a YOLO model-quality issue (the
  OmniParser icon detector is out-of-distribution on Cyberpunk's gradient
  slider tracks; it sees them as multiple icons). Mitigation paths if needed
  later: pre-mask known slider regions per game, increase `box_threshold`, or
  finetune the icon detector on game UI screenshots. **None of these are
  determinism fixes** — they would change parse content, not stability.

- **First-call latency is higher** (~5-9s vs ~2.5s for subsequent calls) because
  CUDA kernel autotune happens on first input shape. This was true before too.

## Files changed

```
util/omniparser.py                                (executor split)
util/utils.py                                     (PaddleOCR cpu_threads=1, mkldnn=False, seed)
util/qwen_ocr.py                                  (hash cache hamming_threshold default 0)
omnitool/omniparserserver/omniparserserver.py     (torch determinism flags at process start)
```

Backups saved next to each file with `.bak.determinism` suffix for easy rollback.

## Next steps for the user (open questions)

1. **Decide if the executor split should also disable dual-GPU parallelism in
   single-GPU mode.** Currently the dual_gpu=False branch runs sequentially in
   the request thread (not via executor) so it's fine.
2. **Multi-DE2 / queue-service mode:** every DE2 instance loads its own
   `paddle_ocr` global. The executor fix is per-process; queue-service routing
   between instances is unaffected.
3. **Roll the patches into a feat branch** and run a real campaign against
   them to confirm the agentic side benefits in production.

## How to reproduce

```
cd F:/Raptor-X-V2/omniparser-server/parse-test
python determinism_probe.py --n-runs 20            # 5x per image + contam
python determinism_probe.py --n-runs 50 --images tick_011.png --no-contamination
```

`after/_REPORT.md` has the rollup. `after/<image>_summary.json` has the full
diff per image. `after/contamination.json` has the cross-image leak report.

## How to roll back

```
cd F:/Raptor-X-V2/omniparser-server
mv util/utils.py.bak.determinism util/utils.py
mv util/omniparser.py.bak.determinism util/omniparser.py
mv util/qwen_ocr.py.bak.determinism util/qwen_ocr.py
mv omnitool/omniparserserver/omniparserserver.py.bak.determinism omnitool/omniparserserver/omniparserserver.py
```

---

# Phase 2 — Qwen-only end-to-end OCR (replaces PaddleOCR detection)

## Why

PaddleOCR's DBNet text detector was failing in two ways even after the
determinism patches above:
1. **Clubbing**: the Cyberpunk Sound menu's top tab bar (SOUND / CONTROLS /
   GAMEPLAY / GRAPHICS / VIDEO / LANGUAGE / INTERFACE / KEY BINDINGS /
   ACCESSIBILITY / UTILITIES) was returned as **one giant polygon** instead
   of 10 separate tab labels.
2. **Mis-segmentation** on stylised game fonts.

These are model-quality issues, not nondeterminism — adding more determinism
flags wouldn't help. The fix: skip Paddle entirely and have Qwen2.5-VL do
both detection and recognition in one grounding call.

## What was added

A new opt-in code path: when the request body contains `"qwen_full_ocr": true`,
the OCR step calls `QwenOCR.detect_and_recognize()` instead of the
PaddleOCR-detect + Qwen-recognise hybrid. Qwen runs the official Qwen2.5-VL
OCR cookbook prompt:

> "Spotting all the text in the image with line-level, and output in JSON format."

Output is parsed as a JSON list of `{bbox_2d: [x1,y1,x2,y2], text_content: "..."}`,
rescaled from the smart-resized tensor space back to original image pixels via
`image_grid_thw`.

## Files changed (Phase 2)

```
util/qwen_ocr.py
    + PerceptualHashCache.__init__ default hamming_threshold = 0   (Phase 1)
    + Lazy fullimg_processor with min_pixels=256*28*28, max_pixels=1024*28*28
      (overridable via QWEN_FULL_OCR_MAX_TOKENS / _MIN_TOKENS env vars)
    + FULLIMG_OCR_PROMPT class constant
    + detect_and_recognize(image, max_new_tokens=2048)
        - greedy-ish sampling: do_sample=True, temperature=1e-6,
          repetition_penalty=1.05 (matches Qwen's shipped generation_config)
        - rescales bboxes from grid_thw resized space → original pixels
        - tolerant JSON parser (strips ```json fence, falls back to regex
          scrape if the model truncates the closing bracket)
        - debug beacon to F:/.../parse-test/logs/qwen_full_ocr_beacon.txt
          and last raw model output to qwen_full_ocr_last_raw.txt
    + model load: attn_implementation="sdpa" (flash_attention_2 is unusable
      because flash_attn 2.8.3 imports triton.runtime which is missing in
      this env; sdpa is 5-10x faster than eager and deterministic for bs=1).

util/utils.py
    check_ocr_box(..., qwen_full_ocr=False)
    + new branch when qwen_ocr is not None and qwen_full_ocr=True:
      calls qwen_ocr.detect_and_recognize(...) and returns (text_list, bb_list)
      with the same shape the legacy hybrid path returned. PaddleOCR is NOT
      called in this branch.

util/omniparser.py
    + _run_ocr accepts qwen_full_ocr param, threads it through check_ocr_box
    + parse() reads override_config['qwen_full_ocr'] (per-request override)
    + dispatch logs the active mode (qwen_full vs qwen_hybrid vs paddleocr)

omnitool/omniparserserver/omniparserserver.py
    + ParseRequest.qwen_full_ocr: Optional[bool] = None   (request body field)
    + override_config['qwen_full_ocr'] forwarded to omniparser.parse
    + module-level Omniparser load is now memoised on `builtins`, so the
      uvicorn import-twice quirk no longer doubles VRAM use (was 22.45 GB on
      a 24 GB GPU at first /parse/ call → OOM on full-image OCR; now ~12 GB
      steady state, ~16-18 GB peak during generate).
    + os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
      to reduce fragmentation under the larger-than-cropped image tensors.
```

## Measurements

All on the running DE2 (single Qwen2.5-VL-3B-Instruct, fp16, sdpa, dual GPU).
Latency is server-side (`response.json()['latency']`).

| Image | Mode | Elements | Latency | All-runs identical | Notes |
|---|---|---|---|---|---|
| tick_008 (Wukong) | hybrid | 15 | 0.46s | yes (50/50) | PaddleOCR works fine on this |
| tick_008 (Wukong) | qwen_full | 9 | 7.6s | yes (3/3) | Qwen returns fewer boxes — less noise from background trees |
| tick_009 (Cyberpunk Sound) | hybrid | 47 | 0.87s | yes (50/50) | Top tab bar **clubbed into 1 box** |
| tick_009 (Cyberpunk Sound) | qwen_full | **55** | **60-64s** | yes (3/3) | **All 10 top tabs detected separately**, all volume slider labels read |
| tick_011 (Civ VI Exit) | hybrid (pre-patch) | 3-6 | 0.5s | NO (8/20 dropouts) | PaddleOCR detection coin-flips |
| tick_011 (Civ VI Exit) | hybrid (post-patch) | 6 | 0.55s | yes (50/50) | Phase 1 executor split fixed it |
| tick_011 (Civ VI Exit) | qwen_full | 6 | 10.7s | yes (3/3) | Same content as hybrid, also stable |

**Trade-off**: Qwen full OCR is 10-100× slower than hybrid (depending on text
density on the screen) but **catches text the PaddleOCR detector misses or
clubs**. Use it when the hybrid path is wrong, not when the hybrid path is slow.

Why so slow for tick_009: 2,514 generated tokens at ~25 tok/s (HF transformers,
fp16, sdpa, no flash_attn). Avenues if you want sub-10s for dense screens:
1. Fix the triton install so flash_attention_2 works → 3-5× speedup
2. Switch to vLLM with batched prefill (the codebase already has a `--vllm_url`
   plumb; full-OCR over vLLM is a follow-up)
3. Lower `QWEN_FULL_OCR_MAX_TOKENS` env var (default 1024) → smaller image
   → faster prefill, but small text gets harder to read

## How to use

Per-request opt-in (recommended):

```bash
curl -X POST http://localhost:8000/parse/ \
     -H "Content-Type: application/json" \
     -d '{"base64_image": "...", "qwen_full_ocr": true}'
```

In rpx-core, this maps to a per-phase or per-tick override in `omniparser_client.py`'s
`detect_ui_elements(ocr_config={"qwen_full_ocr": True})`, the same plumbing
that already carries `box_threshold` overrides. So a phases.yaml entry like:

```yaml
- name: settings_menu
  de2_overrides:
    qwen_full_ocr: true   # this menu has clubbed tabs in the hybrid path
```

would activate it for one phase only and keep all other phases on the fast
hybrid path. (Confirm the rpx-core side actually forwards unknown keys; if it
filters to a known list, that filter needs the new key added.)

## Verified determinism with Qwen full OCR

Probe at q=200 quantization (≈10 px tolerance, sub-pixel jitter is harmless
for the agent):

```
tick_008 qwen_full × 3:  9/9/9 elements, all_identical=True
tick_009 qwen_full × 3: 55/55/55 elements, all_identical=True
tick_011 qwen_full × 3:  6/6/6 elements, all_identical=True
contam A→B→A→B→A:        0 drift on either image
```

Underlying bbox coordinates jitter by ±1 px between runs (fp16/sdpa
reduction-order nondeterminism). Text labels and element counts are bit-stable.
1 px bbox jitter is irrelevant for click targets that span 50+ px.

