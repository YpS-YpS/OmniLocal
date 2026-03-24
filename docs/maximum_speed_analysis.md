# Maximum Speed Analysis: Full Pipeline on RTX 4090 + RTX 4080

## Hardware Specifications

| Spec | RTX 4090 | RTX 4080 |
|---|---|---|
| CUDA Cores | 16,384 | 9,728 |
| Tensor Cores (4th gen) | 512 | 304 |
| FP16 Tensor TFLOPS | 330 | 196 |
| Memory | 24 GB GDDR6X | 16 GB GDDR6X |
| Bandwidth | 1,008 GB/s | 717 GB/s |
| TDP | 450W | 320W |
| Relative compute (vs 4090) | 1.0× | ~0.59× |

**Key insight:** RTX 4080 is roughly 59% the compute of RTX 4090. The 4090 is 46% faster than the 4080 in matched workloads. This ratio matters for workload splitting.

---

## Current Pipeline Timing Breakdown (Baseline: ~80s total)

```
Screenshot capture                          ~5ms
  ↓
YOLO v8 detection (PyTorch, shared GPU)     ~10–15ms
  ↓
PaddleOCR text detection (CPU)              ~50ms
  ↓
Qwen2.5-VL-3B OCR × 60 regions (sequential) ~79,000ms  ← THE BOTTLENECK
  ↓
Florence2 icon captioning × N icons         ~700–2000ms
  ↓
Overlap removal + JSON formatting           ~5ms
─────────────────────────────────────────────────────
TOTAL                                       ~80–82 seconds
```

**99.3% of the time is Qwen2.5-VL-3B sequential inference.** Everything else is noise.

---

## Architecture: The Speed-Maximized Dual-GPU Pipeline

### GPU Assignment

```
┌─────────────────────────────────────────────────────────┐
│  RTX 4090 (24GB) — "The OCR Engine"                     │
│                                                         │
│  vLLM server: Qwen2.5-VL-3B-Instruct                   │
│  - Continuous batching with PagedAttention               │
│  - Prefix caching (shared system prompt)                 │
│  - FP16 / AWQ-INT4 with Marlin kernels                  │
│  - CUDA graphs for decode phase                          │
│  - ~6GB model + ~16GB KV-cache = ~22GB used             │
│                                                         │
│  Processes: ALL 60 text region crops                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RTX 4080 (16GB) — "Detection & Captioning"             │
│                                                         │
│  YOLO v8 TensorRT FP16 engine     (~3–5ms/frame)       │
│  Florence2-base TensorRT/ONNX     (~50–150ms batched)   │
│  PaddleOCR (pre-filter, optional) (~30ms CPU)           │
│                                                         │
│  Processes: Screenshot → bboxes → icon captions          │
│  Total: runs in PARALLEL with OCR, ~200ms               │
└─────────────────────────────────────────────────────────┘
```

### Pipeline Flow (Pipelined & Parallel)

```
Time 0ms:     Screenshot captured
              ├── RTX 4080: YOLO detection starts
              │
Time 5ms:     RTX 4080: YOLO detection complete (~5ms TRT FP16)
              ├── Crop 60 text regions (CPU, ~2ms with numpy slicing)
              ├── RTX 4080: Florence2 icon captioning starts (parallel)
              ├── RTX 4090: vLLM receives 60 OCR requests (async batch)
              │
Time ~200ms:  RTX 4080: Florence2 captioning complete
              RTX 4090: vLLM still processing OCR batch...
              │
Time ~1500ms: RTX 4090: vLLM completes all 60 OCR requests ← TARGET
              │
Time ~1510ms: Merge results, format JSON, return
```

**Critical path = vLLM OCR time.** Detection + captioning finish in ~200ms on the 4080, completely hidden behind the OCR latency on the 4090.

---

## Component-by-Component Speed Calculations

### 1. YOLO v8 — TensorRT FP16 on RTX 4080

**Current:** ~10–15ms in PyTorch on shared GPU
**Optimized:** Export to TensorRT FP16 engine

```bash
yolo export model=omniparser_icon_detect.pt format=engine half=True imgsz=1024
```

Benchmarked speeds for YOLOv8 TensorRT FP16:
- YOLOv8n: 5–8ms per frame on RTX 4090; ~8–12ms on RTX 4080
- YOLOv8m (OmniParser uses custom weights, likely medium-size): ~8–15ms on 4080
- With INT8 calibration: additional 20–40% speedup → ~6–10ms

**Estimated: 8–12ms per screenshot on RTX 4080 TensorRT FP16**

This is already negligible. No further optimization needed.

### 2. Florence2 Icon Captioning — Batched + ONNX/TensorRT on RTX 4080

**Current:** ~700ms sequential in OmniParser V1 (PyTorch, batch of 128 crops at 64×64)
**Architecture:** Florence2-base = 232M params (ViT encoder + text decoder)

Optimization stack:
1. **ONNX export:** Florence2 ONNX models exist on HuggingFace (onnx-community/Florence-2-base). The vision encoder, encoder model, and decoder can be exported separately.
2. **TensorRT conversion of ViT encoder:** The encoder is a standard DaViT — converts cleanly. Expected 2–3× speedup over PyTorch.
3. **Batch all icon crops in single forward pass:** Instead of sequential 64×64 crops, pad all icons to same size and batch through ViT.
4. **Reduce max_new_tokens for captions:** Caps at 20 tokens (OmniParser default) — minimal decode time.

Speed calculation:
- ViT encoder (DaViT): ~10–15ms batched for 30 crops in TensorRT FP16
- Text decoder: ~5ms per token × 10 tokens average × 1 batch = ~50ms
- Total per batch: ~65ms
- For 60 icon crops in 2 batches: ~130ms

**Estimated: 100–200ms total for all Florence2 captioning on RTX 4080**

### 3. Qwen2.5-VL-3B OCR via vLLM — THE CRITICAL PATH

This is where the money is. Let's calculate from first principles.

#### 3a. Token budget per OCR request

```
Input tokens per request:
  System prompt + chat template:     ~30 tokens
  Visual tokens per crop:            4–16 tokens (with min_pixels=28*28*1, max_pixels=28*28*16)
  Instruction text:                  ~20 tokens
  ─────────────────────────────────
  Total input per request:           ~54–66 tokens

Output tokens per request:
  Typical game UI text ("OPTIONS"):  1–5 tokens
  max_new_tokens cap:                15 tokens
  Average output:                    ~3–5 tokens
```

#### 3b. vLLM throughput for Qwen2.5-3B on RTX 4090

From benchmark data:
- Qwen2.5-3B (text-only) on RTX 4090 via vLLM achieves **highest throughput among all models tested** in the DatabaseMart benchmark
- For text-only Qwen2.5-3B: ~2,000–3,000 total tokens/s at batch=1, scaling to 4,000–6,000 tokens/s at higher concurrency
- Vision models add overhead for ViT encoding: ~20–40% slower than text-only for equivalent token counts

Conservative estimate for Qwen2.5-VL-3B on RTX 4090 via vLLM:
- **Prefill throughput:** ~3,000–5,000 tokens/s (processes all input tokens)
- **Decode throughput:** ~500–1,000 tokens/s per concurrent sequence (autoregressive)
- With continuous batching of 60 requests: vLLM processes them as overlapping batches

#### 3c. Detailed timing calculation

**Phase 1: Prefill (processing all 60 inputs)**

With continuous batching, vLLM processes prefill in chunks:
```
Total input tokens: 60 requests × ~60 tokens = 3,600 tokens
Prefill throughput: ~4,000 tokens/s (conservative for VLM on 4090)
Prefill time: 3,600 / 4,000 = 0.9 seconds
```

But vLLM overlaps prefill with decode, so as early requests finish prefill, they start decoding while later requests are still being prefilled.

**Phase 2: Decode (generating output tokens)**

```
Total output tokens: 60 requests × ~4 tokens average = 240 tokens
With batched decode at 60 concurrent sequences:
  Per-token latency: ~15–25ms (batched, 3B model, short sequences)
  Tokens needed: ~4 per request
  Decode steps: 4 (all sequences generate in parallel)
  Decode time: 4 steps × 20ms = 80ms
```

**Phase 3: ViT encoding overhead**

Each image crop passes through Qwen2.5-VL's ViT (675M params):
```
With reduced pixels (max_pixels=28*28*16):
  Each crop → ~16 visual tokens → ViT processes ~16 patches
  Batched ViT encoding for 60 crops: ~100–300ms
  (ViT uses window attention, scales linearly with patches)
```

#### 3d. Total vLLM OCR time

```
ViT encoding (batched):     200ms
Prefill (overlapped):       600ms  (faster than raw calc due to continuous batching)
Decode (4 steps batched):    80ms
Scheduling overhead:         50ms
API/network overhead:       100ms  (localhost HTTP, 60 async requests)
─────────────────────────────
TOTAL:                    ~1,030ms → round to ~1.0–1.5 seconds
```

#### 3e. With prefix caching enabled

vLLM's `--enable-prefix-caching` caches the KV-cache for the shared system prompt + chat template (~30 tokens). After the first request, all 59 subsequent requests skip prefilling those tokens.

```
Savings: 59 × 30 tokens × ~0.25ms/token = ~440ms saved
Adjusted total: ~600ms–1.1 seconds
```

#### 3f. With LMCache multimodal KV caching

LMCache integration with vLLM caches vision KV states. For your use case (same game, similar UI regions), many crops may produce near-identical visual tokens across consecutive screenshots:
- First run: ~1.0–1.5s
- Subsequent runs with similar screenshots: ~200–500ms (KV cache hits)

---

## THE MAXIMUM SPEED CONFIGURATION

### vLLM Launch Command (Linux/Docker)

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --mm-processor-kwargs '{"min_pixels": 784, "max_pixels": 12544}' \
  --limit-mm-per-prompt image=1 \
  --enforce-eager false \
  -O2 \
  --api-server-count 2
```

Key flags explained:
- `--max-model-len 2048`: We only need ~80 tokens per request; 2048 gives headroom while maximizing KV-cache slots
- `--gpu-memory-utilization 0.92`: Use almost all 24GB (model=6GB, KV-cache=~16GB)
- `--enable-prefix-caching`: Cache the shared system prompt across all 60 requests
- `--max-num-seqs 64`: Allow all 60 requests to be in-flight simultaneously
- `--max-num-batched-tokens 4096`: Total tokens processed per iteration
- `--mm-processor-kwargs`: Aggressive pixel reduction (1–16 visual tokens per crop)
- `-O2`: Full CUDA graph optimization + kernel fusion
- `--api-server-count 2`: Two API server processes for faster request handling

### Client Code (Async, Fire All 60 at Once)

```python
import asyncio
import aiohttp
import time

async def ocr_one_crop(session, crop_base64, semaphore):
    async with semaphore:
        payload = {
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_base64}"}},
                    {"type": "text", "text": "Read all text in this image exactly as it appears. Output only the text, nothing else."}
                ]
            }],
            "max_tokens": 15,
            "temperature": 0,
        }
        async with session.post("http://localhost:8000/v1/chat/completions", json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]

async def ocr_all_crops(crops_base64: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(60)  # all at once
    async with aiohttp.ClientSession() as session:
        tasks = [ocr_one_crop(session, crop, semaphore) for crop in crops_base64]
        return await asyncio.gather(*tasks)

# Usage
t0 = time.perf_counter()
results = asyncio.run(ocr_all_crops(all_60_crops_base64))
print(f"60 crops OCR: {time.perf_counter() - t0:.2f}s")
```

---

## Speed Results: Before vs After

### Conservative estimate (achievable day 1)

| Component | Before | After | Speedup |
|---|---|---|---|
| YOLO detection | 15ms (PyTorch) | 10ms (TRT FP16 on 4080) | 1.5× |
| PaddleOCR pre-filter | 50ms (CPU) | 30ms (optimized) | 1.7× |
| **Qwen OCR × 60** | **79,000ms** | **1,500ms** (vLLM batched) | **53×** |
| Florence2 captioning | 700ms | 200ms (ONNX batched on 4080) | 3.5× |
| Merge + format | 5ms | 5ms | 1× |
| **TOTAL** | **~80,000ms** | **~1,750ms** | **~46×** |

### Aggressive estimate (with all tricks)

| Component | Time |
|---|---|
| YOLO TRT INT8 on 4080 | 6ms |
| Qwen OCR × 60 via vLLM (prefix cache + reduced pixels + CUDA graphs) | 800ms |
| Florence2 TRT batched on 4080 | 100ms |
| Overlap removal + JSON | 3ms |
| **TOTAL** | **~910ms** |

### Theoretical floor (absolute minimum with this hardware)

```
YOLO:        Can't go below ~3ms (kernel launch overhead)
ViT encode:  60 crops × 16 patches × 14×14 pixels = 60 × ~3ms batched = ~50ms
LLM prefill: 3,600 tokens at 10,000 tok/s (4090 peak) = 360ms  
LLM decode:  4 steps × 15ms batched = 60ms
Overhead:    50ms (scheduling + memory + API)
─────────────────────────────────────────
Theoretical floor: ~523ms
```

**You won't hit 523ms in practice,** but it shows the hardware limit. **800ms–1,500ms is the realistic achievable range.**

---

## Every Trick Under the Sleeve

### Trick 1: CUDA Graphs (vLLM -O2)
vLLM captures the decode phase as a CUDA graph after warmup. This eliminates kernel launch overhead (~100μs per kernel) across ~50 kernels per decode step. For 4 decode steps: saves ~20ms total. Enabled by default with `-O2`.

### Trick 2: PagedAttention
vLLM's PagedAttention manages KV-cache in non-contiguous 16-token pages. For 60 concurrent sequences with ~80 tokens each, this packs efficiently into ~16GB of KV-cache space. Without PagedAttention, you'd need pre-allocated contiguous memory and couldn't fit 60 concurrent sequences.

### Trick 3: Prefix Caching
The system prompt + chat template (~30 tokens) is identical across all 60 requests. vLLM computes the KV-cache for this prefix ONCE and reuses it. Saves ~30 × 60 × (prefill cost per token) = significant prefill time reduction.

### Trick 4: Continuous Batching
Rather than waiting for all 60 to prefill before decoding, vLLM interleaves: as request 1 finishes prefill, it starts decoding immediately while requests 2–60 are still being prefilled. This overlaps phases and keeps the GPU saturated.

### Trick 5: Aggressive Pixel Reduction
`min_pixels=784, max_pixels=12544` (1–16 visual tokens per crop). Your crops are 50–300px text regions. At 14px patch stride, a 112×28 crop produces exactly 4×1 = 4 patches (after 2×2 merge = 1 visual token). Most OCR crops need ≤16 visual tokens. This is the single biggest speed lever after batching.

### Trick 6: max_tokens=15
Game UI text is short. "OPTIONS" = 1 token, "Benchmark Results" = 2 tokens, "Press any key to continue" = 6 tokens. Setting max_tokens=15 (not 128) means:
- KV-cache allocation per request is 8.5× smaller
- Maximum decode steps reduced from 128 to 15
- More concurrent sequences fit in memory

### Trick 7: Greedy Decoding with Early Stopping
`temperature=0` means greedy decode (no sampling overhead). The model emits EOS after the text, stopping immediately. Average decode steps: 3–5 (not 15).

### Trick 8: Image Pre-processing on CPU (Pipeline Overlap)
While the previous screenshot's OCR is running on the 4090, prepare the next screenshot's crops on CPU:
```python
# Thread 1: YOLO + crop extraction (4080 + CPU)
# Thread 2: vLLM OCR inference (4090)
# These run in parallel via asyncio/threading
```

### Trick 9: JPEG Compression for API Transfer
Encode crops as JPEG quality=85 before base64-encoding for vLLM API. Reduces payload size 5–10× vs PNG, faster network transfer, faster base64 decode. vLLM handles JPEG natively.

### Trick 10: Florence2 Batched ViT with ONNX Runtime
Export Florence2's DaViT encoder to ONNX, run with ONNX Runtime CUDA EP:
```python
import onnxruntime as ort
session = ort.InferenceSession("florence2_encoder.onnx", 
    providers=["CUDAExecutionProvider"], 
    provider_options=[{"device_id": 1}])  # RTX 4080
```
Batch all 60 icon crops → single forward pass → ~50ms for the encoder, then decode captions sequentially (~100ms total).

### Trick 11: Template Matching Bypass (Skip VLM for Known Elements)
For recurring game UI elements (same button, same position), use perceptual hash (dHash) + template matching on CPU (~1ms per check). If hash matches a known element within Hamming distance ≤5, return cached text instantly. For a typical game menu: **40–50 of 60 elements are static/known** → only 10–20 actually need VLM inference.

With this: 15 crops × vLLM = ~400ms instead of 60 crops × vLLM = ~1,200ms.

### Trick 12: Dual vLLM Instances (Both GPUs)
If Florence2 can be made fast enough on CPU (or skipped):
```
RTX 4090: vLLM instance 1 (Qwen2.5-VL-3B) — handles 38 crops
RTX 4080: vLLM instance 2 (Qwen2.5-VL-3B) — handles 22 crops
```
The 4080 has 16GB: model (6GB) + KV-cache (8GB) = fits. It's ~59% the speed of 4090.

```
RTX 4090: 38 crops → ~750ms
RTX 4080: 22 crops → ~750ms (slower GPU, fewer crops, balanced)
Both in parallel → ~750ms total
```

**This requires moving Florence2/YOLO to CPU or a scheduling trick where YOLO runs on 4080 first (12ms), then vLLM takes over the GPU.**

### Trick 13: Multi-Image Single Prompt (Experimental)
Instead of 60 separate requests, send one request with 10 images:
```
"Read the text in each of these 10 images, one per line:"
[img1] [img2] [img3] ... [img10]
```
6 such requests × 10 images each = 60 crops. Benefits:
- Amortizes system prompt overhead across 10 crops
- Single prefill + single decode stream per request
- Risk: model may confuse images or miss some

### Trick 14: vLLM API Server Scale-Out
vLLM supports `--api-server-count 4` to run 4 API processes handling HTTP requests, tokenization, and preprocessing, all feeding one GPU engine. This eliminates API server as a bottleneck for 60 concurrent requests.

### Trick 15: Pre-warm the Engine
Before the benchmark run, send 5 dummy OCR requests to vLLM. This:
- Triggers CUDA graph capture (first-run compilation)
- Warms the KV-cache allocator
- Pre-compiles any torch.compile kernels
- Fills the prefix cache with the system prompt

---

## Comparison: vLLM vs HuggingFace Transformers Batched

| Metric | HF Transformers (batched) | vLLM |
|---|---|---|
| 60 crops OCR time | 5–10s (manual batching) | 0.8–1.5s |
| Prefill strategy | One batch at a time | Continuous batching |
| Decode strategy | All sequences in sync | Asynchronous per-sequence |
| KV-cache management | Pre-allocated, wasteful | PagedAttention, efficient |
| Prefix caching | Not available | Built-in |
| CUDA graphs | Manual (torch.compile) | Automatic (-O2) |
| Multi-image batching | Must handle padding manually | Automatic |
| Memory efficiency | ~60–70% utilization | ~90%+ utilization |
| **Platform** | **Windows native** | **Linux only (Docker/WSL2)** |

**The vLLM advantage is 5–10× over HuggingFace transformers for this workload.** The continuous batching + PagedAttention + CUDA graphs combination is why.

---

## Decision Matrix: Which Path?

| Scenario | Total OCR Time | Total Pipeline | Notes |
|---|---|---|---|
| Current (sequential HF) | 79s | ~80s | Baseline |
| HF batched + reduced pixels | 5–10s | ~6–11s | Windows native, no vLLM |
| vLLM single 4090 | 0.8–1.5s | ~1.7–2.5s | Requires Linux/Docker |
| vLLM + template matching | 0.3–0.6s | ~1.0–1.5s | 40+ crops skipped via cache |
| Dual vLLM (4090+4080) | 0.5–0.8s | ~0.8–1.3s | Max throughput, complex setup |
| Dual vLLM + template bypass | 0.2–0.4s | ~0.5–0.8s | **Theoretical best** |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Day 1)
- Reduce `max_pixels` to `28*28*16` and `max_new_tokens` to 15
- Move YOLO + Florence2 to RTX 4080 (`device_map={"": "cuda:1"}`)
- Keep Qwen on RTX 4090 with HF batched inference (batch_size=8–12)
- **Expected: 79s → 5–10s**

### Phase 2: Docker + vLLM (Days 2–3)
- Install Docker Desktop (uses WSL2 backend, but your app stays on Windows)
- Launch vLLM container on RTX 4090 with the flags above
- Client code stays native Windows Python, hits localhost:8000 API
- **Expected: 5–10s → 1.0–1.5s**

### Phase 3: Full Optimization (Days 4–7)
- Export YOLO to TensorRT engine on 4080
- Export Florence2 encoder to ONNX on 4080
- Implement template matching cache for known UI elements
- Add perceptual hash caching (skip identical frames)
- **Expected: 1.0–1.5s → 0.5–1.0s**

### Phase 4: Endgame (Week 2)
- Dual vLLM instances (both GPUs) with load balancing
- LMCache integration for multimodal KV caching
- Fine-tuned TrOCR (334M) as fast path for high-confidence detections
- Template library per game (50–80% of elements cached)
- **Expected: 0.3–0.8s per screenshot**

---

## Final Answer

**Maximum achievable speed for 60 text regions on RTX 4090 + RTX 4080:**

| Configuration | Time |
|---|---|
| Theoretical hardware floor | ~520ms |
| Best realistic (dual vLLM + all tricks) | **500–800ms** |
| Practical target (single vLLM + template cache) | **800–1,500ms** |
| Conservative (HF batched + pixel reduction) | **5,000–10,000ms** |

**The vLLM path turns a 79-second bottleneck into a sub-2-second operation.** The Docker requirement is the only obstacle, and it's a thin one — your Windows application just makes HTTP calls to localhost:8000. Everything except vLLM stays native Windows.
