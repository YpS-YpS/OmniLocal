import torch
import io
import base64
import asyncio
import time
import numpy as np
from PIL import Image
from typing import List, Optional, Dict
from collections import OrderedDict


class PerceptualHashCache:
    """dHash-based perceptual hashing to skip VLM inference for known UI elements.

    For typical game menus, 40-50 of 60 elements are static across frames.
    Cache hit = 0ms instead of ~20-100ms per crop.
    """

    def __init__(self, max_size: int = 2048, hamming_threshold: int = 0):
        # Default 0 = exact dHash match only. Fuzzy matching with threshold>0 caused
        # cross-image text bleed: a slider-thumb crop from frame N could match a
        # different slider-thumb crop on frame N+1 within Hamming distance 5 and
        # return the wrong cached label. Use explicit threshold>0 only when you
        # know the cached *value* is invariant to small visual perturbations
        # (NOT true for OCR text content).
        self.max_size = max_size
        self.hamming_threshold = hamming_threshold
        self._cache: OrderedDict[int, str] = OrderedDict()  # hash -> text
        self.hits = 0
        self.misses = 0

    def _dhash(self, image: Image.Image, hash_size: int = 16) -> int:
        """Compute difference hash (dHash) for an image. Fast and robust to small changes."""
        gray = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(gray)
        # Compare adjacent pixels horizontally
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Pack into integer
        return int(np.packbits(diff.flatten()).tobytes().hex(), 16)

    def _hamming_distance(self, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')

    def lookup(self, crop: Image.Image) -> Optional[str]:
        """Check if a similar crop has been seen before. Returns cached text or None."""
        crop_hash = self._dhash(crop)

        # Exact match first (fast path)
        if crop_hash in self._cache:
            self.hits += 1
            self._cache.move_to_end(crop_hash)
            return self._cache[crop_hash]

        # Fuzzy match (hamming distance)
        for cached_hash, text in self._cache.items():
            if self._hamming_distance(crop_hash, cached_hash) <= self.hamming_threshold:
                self.hits += 1
                self._cache.move_to_end(cached_hash)
                return text

        self.misses += 1
        return None

    def store(self, crop: Image.Image, text: str):
        """Store a crop->text mapping in the cache."""
        crop_hash = self._dhash(crop)
        self._cache[crop_hash] = text
        self._cache.move_to_end(crop_hash)
        # Evict oldest if over capacity
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            'size': len(self._cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hits/total*100:.1f}%" if total > 0 else "N/A"
        }


class QwenOCR:
    """Maximum-speed Qwen2.5-VL-3B OCR engine.

    Speed tricks applied:
    - Aggressive pixel reduction (max_pixels=28*28*16 = 12544, down from 200704)
    - max_new_tokens=15 (game UI text is short)
    - Batch inference (process multiple crops at once)
    - Perceptual hash cache (skip VLM for repeated UI elements)
    - JPEG compression for crops (faster processing)
    - Greedy decoding (temperature=0, no sampling overhead)
    - Optional vLLM async client mode (fires all crops simultaneously)
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda",
                 use_hash_cache: bool = True, vllm_url: str = None,
                 batch_size: int = 8, quantize: str = None):
        self.vllm_url = vllm_url
        self.batch_size = batch_size
        self.device_str = device
        self.model_path = model_path
        self.quantize = quantize  # None, 'int4', or 'int8'

        # Perceptual hash cache
        self.hash_cache = PerceptualHashCache() if use_hash_cache else None

        # Lazy-loaded full-image processor for end-to-end OCR (Qwen detect+recognise).
        # Distinct from self.processor (which is configured for tiny ≤12544-pixel
        # crops); full-image OCR needs much higher max_pixels to keep small UI
        # text legible. Loaded on first use to avoid extra startup cost when
        # not needed.
        self._fullimg_processor = None

        if vllm_url:
            # vLLM mode: no local model, just HTTP client
            print(f"[QwenOCR] vLLM mode: {vllm_url}")
            self.model = None
            self.processor = None
            self.device = None
        else:
            # Local HF mode with aggressive speed settings
            self._load_local_model(model_path, device)

    def _load_local_model(self, model_path: str, device: str):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # Optional: swap to the official AWQ-Int4 build for ~2x decode speedup.
        # Set env QWEN_OCR_USE_AWQ=1 to enable. Visual encoder is NOT quantised
        # (only the LM modules), so vision quality is unchanged. Halves VRAM.
        # Requires autoawq 0.2.7.post1 (installed with --no-deps to bypass
        # triton/triton-windows resolver conflict + transformers.qwen3 hard
        # dep in 0.2.8+).
        import os as _os
        if _os.environ.get('QWEN_OCR_USE_AWQ', '').lower() in ('1', 'true', 'yes'):
            awq_id = _os.environ.get(
                'QWEN_OCR_AWQ_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct-AWQ',
            )
            print(f"[QwenOCR] AWQ mode: swapping {model_path} -> {awq_id}", flush=True)
            model_path = awq_id
            self.model_path = awq_id

        quant_label = f" [{self.quantize}]" if self.quantize else " [fp16]"
        print(f"[QwenOCR] Loading model from {model_path} on {device}{quant_label}...")
        load_start = time.time()

        if device == "cpu":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float32, device_map="cpu"
            )
        elif self.quantize == 'int4':
            # INT4 quantization: ~2GB VRAM instead of ~6GB, ~10-20% slower
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, quantization_config=bnb_config, device_map=device
            )
        elif self.quantize == 'int8':
            # INT8 quantization: ~4GB VRAM, minimal quality loss
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, quantization_config=bnb_config, device_map=device
            )
        else:
            # bf16 + backbone-specific attention (transformers 4.49+ supports
            # passing a dict to attn_implementation). The Qwen2.5-VL ViT visual
            # encoder uses a custom rotary kernel that imports triton from
            # flash_attn — this works on Windows once triton-windows is
            # installed. The LM decoder dominates runtime (2500-token
            # generation on a dense screen), so flash_attention_2 there is
            # the big win — ~2x decode throughput on RTX 4090.
            #
            # bf16 (vs fp16) costs nothing on Ada Lovelace tensor cores but
            # avoids occasional NaN from long-context VLM dynamic range.
            # fp16 + sdpa is the known-good config for this stack. Empirically:
            #   - flash_attention_2 (with triton-windows installed) broke
            #     determinism (text_drift=62 across 3 runs) AND made latency
            #     WORSE for our use case (98s vs 60s with sdpa). The HF FA2
            #     integration is documented as nondeterministic at the forward
            #     pass; for OCR's 2500-token decode that compounds.
            #   - bf16 helps numerical stability on long contexts but on our
            #     RTX 4090 has no measured speed advantage and triggered a
            #     CUDA device-side assert in torch.multinomial under FA2.
            # Optimisations we DO take below:
            #   - StoppingCriteria on the JSON close bracket (skip dead tokens)
            #   - smaller max_new_tokens budget (2048 instead of 4096)
            #   - Qwen2.5-VL-3B-Instruct-AWQ if/when configured (env var)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device,
                attn_implementation="sdpa",
            )
            print(
                "[QwenOCR] Loaded with fp16 + attn_implementation=sdpa "
                "(FA2 disabled — see comment in qwen_ocr.py)",
                flush=True,
            )
        self.model.eval()

        # AGGRESSIVE pixel reduction: 28*28*4=3136 min, 28*28*16=12544 max
        # Most OCR crops need <=16 visual tokens. This is the single biggest
        # speed lever after batching (was 28*28*256=200704 before = 16x reduction)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=28 * 28 * 4,     # 3136 - safe minimum
            max_pixels=28 * 28 * 16,    # 12544 - 16 visual tokens max
        )
        # Decoder-only models need left-padding for correct batch generation
        self.processor.tokenizer.padding_side = 'left'

        self.device = self.model.device
        self._prompt_text = self._build_prompt()

        print(f"[QwenOCR] Model loaded in {time.time() - load_start:.1f}s on {self.device}")
        print(f"[QwenOCR] Speed config: max_pixels=12544, max_new_tokens=15, batch_size={self.batch_size}")
        if self.hash_cache:
            print(f"[QwenOCR] Perceptual hash cache: ENABLED (max_size={self.hash_cache.max_size}, hamming_threshold={self.hash_cache.hamming_threshold})")

    def _build_prompt(self) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Read all text in this image exactly as it appears. Output only the text, nothing else."},
            ]
        }]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _prepare_crop(crop: Image.Image) -> Image.Image:
        """Prepare a crop for inference with minimum dimension and aspect ratio enforcement.

        - Minimum 56px per dimension (2x safety margin for processor resize)
        - Maximum 14:1 aspect ratio (processor resize-to-zero at ~16:1 with max_pixels=12544)
        """
        if crop.width < 2 or crop.height < 2:
            return None

        crop_rgb = crop.convert("RGB")

        # Cap extreme aspect ratios: the Qwen processor rounds dimensions to multiples
        # of 28 after scaling to fit pixel budget. At ~16:1 ratio the short dimension
        # rounds to 0. We cap at 14:1 with padding.
        MAX_RATIO = 14.0
        w, h = crop_rgb.width, crop_rgb.height
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > MAX_RATIO:
            if w > h:
                new_h = max(int(w / MAX_RATIO), 56)
                padded = Image.new("RGB", (w, new_h), (0, 0, 0))
                padded.paste(crop_rgb, (0, (new_h - h) // 2))
                crop_rgb = padded
            else:
                new_w = max(int(h / MAX_RATIO), 56)
                padded = Image.new("RGB", (new_w, h), (0, 0, 0))
                padded.paste(crop_rgb, ((new_w - w) // 2, 0))
                crop_rgb = padded

        # Enforce minimum dimensions
        MIN_DIM = 56
        w, h = crop_rgb.width, crop_rgb.height
        if w < MIN_DIM or h < MIN_DIM:
            scale = max(MIN_DIM / w, MIN_DIM / h, 1.0)
            new_w = max(int(w * scale), MIN_DIM)
            new_h = max(int(h * scale), MIN_DIM)
            crop_rgb = crop_rgb.resize((new_w, new_h), Image.LANCZOS)

        return crop_rgb

    @staticmethod
    def _crop_to_jpeg_base64(crop: Image.Image, quality: int = 85) -> str:
        """Compress crop to JPEG base64 for vLLM API transfer. 5-10x smaller than PNG."""
        buf = io.BytesIO()
        crop.save(buf, format='JPEG', quality=quality)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    @torch.inference_mode()
    def recognize_crop(self, crop: Image.Image) -> str:
        """Recognize text in a single cropped image region."""
        prepared = self._prepare_crop(crop)
        if prepared is None:
            return ""

        # Check hash cache first
        if self.hash_cache:
            cached = self.hash_cache.lookup(prepared)
            if cached is not None:
                return cached

        if self.vllm_url:
            # Single crop via vLLM (synchronous wrapper, safe in FastAPI async context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run, self._vllm_ocr_single(prepared)
                        ).result()
                else:
                    result = loop.run_until_complete(self._vllm_ocr_single(prepared))
            except RuntimeError:
                result = asyncio.run(self._vllm_ocr_single(prepared))
        else:
            result = self._local_recognize_single(prepared)

        # Store in cache
        if self.hash_cache and result:
            self.hash_cache.store(prepared, result)

        return result

    @torch.inference_mode()
    def _local_recognize_single(self, crop_rgb: Image.Image) -> str:
        """Local HF inference for a single crop."""
        inputs = self.processor(
            text=[self._prompt_text],
            images=[crop_rgb],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=15,    # Game UI text is short (was 128)
            do_sample=False,       # Greedy = no sampling overhead
            num_beams=1,
        )

        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    @torch.inference_mode()
    def _local_recognize_batch(self, crops: List[Image.Image]) -> List[str]:
        """Batched local HF inference for multiple crops at once.

        Processes crops in batches to keep GPU saturated instead of
        sequential single-crop inference.
        """
        if not crops:
            return []

        # Process all crops through the processor at once
        prompts = [self._prompt_text] * len(crops)
        inputs = self.processor(
            text=prompts,
            images=crops,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=15,
            do_sample=False,
            num_beams=1,
        )

        # With left-padding, all sequences are right-aligned so the actual content
        # ends at position input_ids.shape[1]. We can use a uniform slice safely
        # since generate() pads all outputs to the same length.
        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        texts = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # End-to-end OCR (Qwen detect + recognise — replaces PaddleOCR step)
    # ------------------------------------------------------------------

    # Canonical Qwen2.5-VL OCR prompt (from QwenLM/Qwen2.5-VL/cookbooks/ocr.ipynb).
    # The "line-level" variant is the one the team trained on; "word-level" splits
    # text into per-word boxes and is noisier on game UI which mixes labels and
    # values on the same line.
    FULLIMG_OCR_PROMPT = (
        "Spotting all the text in the image with line-level, "
        "and output in JSON format."
    )

    # Pixel budget for full-image OCR. The Qwen2.5-VL cookbook suggests
    # 2048*28*28 (~1.6M pixels) for best legibility on UI screenshots, but
    # ViT attention is O(n²) and on a 24 GB GPU shared with another Qwen copy
    # (from the uvicorn re-import quirk) that OOMs at 2019 visual tokens.
    # 1024 visual tokens = 1024*28*28 = 802816 pixels = ~1067x756 effective
    # resolution after smart_resize on 1920x1080 input. Still legible for
    # 14-18 px game UI text. Adjust via env var if you have headroom:
    #   QWEN_FULL_OCR_MAX_TOKENS=1536  -> larger images, more VRAM
    #   QWEN_FULL_OCR_MAX_TOKENS=768   -> faster, may miss small text
    _FULLIMG_DEFAULT_MAX_TOKENS = 1024
    _FULLIMG_DEFAULT_MIN_TOKENS = 256

    def _get_fullimg_processor(self):
        """Lazy-load a separate processor for full-image OCR.

        Default self.processor caps at 12544 pixels — fine for tiny crops, way
        too small for a full 1920x1080 screenshot. We keep both processors so
        the cropped-recognition path stays fast while end-to-end OCR has enough
        pixel budget to read game UI text.
        """
        if self._fullimg_processor is not None:
            return self._fullimg_processor
        import os as _os
        from transformers import AutoProcessor
        max_tokens = int(_os.environ.get(
            'QWEN_FULL_OCR_MAX_TOKENS', self._FULLIMG_DEFAULT_MAX_TOKENS,
        ))
        min_tokens = int(_os.environ.get(
            'QWEN_FULL_OCR_MIN_TOKENS', self._FULLIMG_DEFAULT_MIN_TOKENS,
        ))
        print(
            f"[QwenOCR-full] processor min_pixels={min_tokens}*28*28={min_tokens*28*28}, "
            f"max_pixels={max_tokens}*28*28={max_tokens*28*28}",
            flush=True,
        )
        self._fullimg_processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=min_tokens * 28 * 28,
            max_pixels=max_tokens * 28 * 28,
        )
        # Decoder-only models prefer left-padding for batch generation.
        self._fullimg_processor.tokenizer.padding_side = 'left'
        return self._fullimg_processor

    @staticmethod
    def _parse_grounding_json(raw: str) -> List[Dict]:
        """Parse Qwen2.5-VL grounding JSON output.

        Tolerates: leading ```json fences, trailing prose, single-quote dicts
        (the cookbook itself uses single quotes in its example output), and
        truncated trailing items when max_new_tokens is hit.
        """
        import re
        import json
        s = raw.strip()
        # Strip a markdown fence if present.
        m = re.search(r'```(?:json)?\s*(.+?)\s*```', s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        # Try strict JSON first.
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        # Find the first JSON array span and parse just that.
        m = re.search(r'\[.*\]', s, re.DOTALL)
        if m:
            blob = m.group(0)
            try:
                data = json.loads(blob)
                if isinstance(data, list):
                    return data
            except Exception:
                # Single-quote → double-quote (lossy but matches cookbook samples).
                try:
                    import ast
                    data = ast.literal_eval(blob)
                    if isinstance(data, list):
                        return data
                except Exception:
                    pass
        # Last-ditch: scrape all `bbox_2d`/`text_content` items individually.
        items = []
        item_re = re.compile(
            r'\{\s*[\'"]bbox_2d[\'"]\s*:\s*\[\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*,'
            r'\s*([\-\d\.]+)\s*,\s*([\-\d\.]+)\s*\]\s*,\s*'
            r'[\'"]text_content[\'"]\s*:\s*[\'"]((?:[^\'"]|\\.)*)[\'"]\s*\}',
            re.DOTALL,
        )
        for x1, y1, x2, y2, text in item_re.findall(s):
            try:
                items.append({
                    'bbox_2d': [float(x1), float(y1), float(x2), float(y2)],
                    'text_content': text,
                })
            except Exception:
                continue
        return items

    class _StopOnJsonClose:
        """Stop generation as soon as the model emits the closing `]` of the
        outer JSON array. Saves the typical ~100 trailing tokens (whitespace,
        end-of-message tokens) that the model would otherwise generate.

        Implemented as a transformers StoppingCriteria. We decode only the
        tail of the generated tokens each step (8 tokens is plenty to catch
        `]\\n` or `]}`) and check for a string match — the `]` token can
        merge with adjacent whitespace in different ways (`"]"`, `"]\\n"`,
        `" ]\\n"`, `"]\\n\\n"`), so token-id matching is fragile.
        """
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            tail = input_ids[0, -8:]
            try:
                text = self.tokenizer.decode(tail, skip_special_tokens=True)
            except Exception:
                return False
            stripped = text.rstrip()
            # JSON close on the whole array; tolerate either `]` alone or `}]`
            # (when the LAST item is an object closing right before the array
            # close, which is the normal Qwen output shape).
            return stripped.endswith(']')

    @torch.inference_mode()
    def detect_and_recognize(
        self,
        image: Image.Image,
        max_new_tokens: int = 2048,
    ) -> "tuple[List[str], List[List[int]]]":
        """End-to-end OCR via Qwen2.5-VL grounding.

        Returns
        -------
        (texts, bboxes_pixel)
            ``texts``  — list of recognised strings, one per detected line.
            ``bboxes_pixel`` — matching list of [x1, y1, x2, y2] in **original
            image pixel space**. The model emits coords in the smart-resized
            tensor space; we rescale here using image_grid_thw so callers get
            coordinates that line up with the input image.
        """
        # Beacons: confirm we entered detect_and_recognize and where it dies.
        # Stdout from request handlers isn't reaching our log; this gives us
        # ground truth from inside the function.
        import os as _os
        _beacon = _os.environ.get(
            'QWEN_OCR_BEACON_FILE',
            'F:/Raptor-X-V2/omniparser-server/parse-test/logs/qwen_full_ocr_beacon.txt',
        )

        def _b(msg: str) -> None:
            try:
                with open(_beacon, 'a', encoding='utf-8') as f:
                    f.write(f"{time.time():.3f} {msg}\n")
            except Exception:
                pass

        _b(f"ENTER detect_and_recognize image={image.size}")

        if self.vllm_url:
            # Routing through vLLM not yet implemented for grounding; fall back
            # to local HF if model is loaded, else raise.
            if self.model is None:
                raise RuntimeError(
                    "Qwen full-image OCR via vLLM is not yet supported; "
                    "start the server without --vllm_url to enable it."
                )

        try:
            proc = self._get_fullimg_processor()
            _b("got fullimg processor")
            image_rgb = image.convert("RGB")
            orig_w, orig_h = image_rgb.size

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.FULLIMG_OCR_PROMPT},
                ],
            }]
            prompt_text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            _b(f"prompt_text built len={len(prompt_text)}")
            inputs = proc(
                text=[prompt_text],
                images=[image_rgb],
                return_tensors="pt",
            ).to(self.device)
            _b(f"inputs prepared keys={list(inputs.keys())} input_ids_shape={tuple(inputs.input_ids.shape)}")

            t0 = time.perf_counter()
            # Greedy decoding: do_sample=False is required with FA2 + bf16 —
            # the cookbook's `do_sample=True, temperature=1e-6` sampling path
            # blows up `torch.multinomial` when FA2 produces underflow probs in
            # bf16 (CUDA device-side assert in _sample). Greedy sidesteps
            # multinomial entirely; for OCR this matches the documented
            # behaviour at temperature=0 anyway.
            from transformers import StoppingCriteriaList, StoppingCriteria
            class _Stop(StoppingCriteria):
                def __init__(self, helper): self.helper = helper
                def __call__(self, input_ids, scores, **kw): return self.helper(input_ids, scores, **kw)
            stop = StoppingCriteriaList([_Stop(self._StopOnJsonClose(proc.tokenizer))])
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                stopping_criteria=stop,
            )
            t1 = time.perf_counter()
            _b(f"generate done {t1-t0:.2f}s gen_ids_shape={tuple(gen_ids.shape)}")
            out_ids = gen_ids[:, inputs.input_ids.shape[1]:]
            raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
            _b(f"decoded raw len={len(raw)}")
        except Exception as _exc:
            import traceback as _tb
            _b(f"EXCEPTION: {type(_exc).__name__}: {_exc}")
            _b("TRACEBACK:\n" + _tb.format_exc())
            raise
        # Debug: dump raw output to a file (uvicorn captures stdout in a way that
        # makes interactive debugging hard). Each call overwrites the same file.
        try:
            import os as _os
            _dbg_path = _os.environ.get(
                'QWEN_OCR_DEBUG_FILE',
                'F:/Raptor-X-V2/omniparser-server/parse-test/logs/qwen_full_ocr_last_raw.txt',
            )
            with open(_dbg_path, 'w', encoding='utf-8') as f:
                f.write(raw)
        except Exception:
            pass
        # Also try stdout (will appear if buffering is friendly).
        _preview = raw if len(raw) <= 600 else raw[:600] + "...<truncated>"
        print(f"[QwenOCR-full] RAW OUTPUT >>>{_preview}<<<", flush=True)

        # Rescale resized-space coords back to original image pixels.
        # image_grid_thw is shape (1, 3) = (T, H_grid, W_grid); each grid cell
        # is 14x14 (the patch size). Resized image dims = grid * 14.
        try:
            grid_thw = inputs['image_grid_thw'][0].tolist()
            resized_h = grid_thw[1] * 14
            resized_w = grid_thw[2] * 14
        except Exception:
            resized_h, resized_w = orig_h, orig_w
        sx = orig_w / max(resized_w, 1)
        sy = orig_h / max(resized_h, 1)

        items = self._parse_grounding_json(raw)
        texts: List[str] = []
        bboxes: List[List[int]] = []
        for item in items:
            bb = item.get('bbox_2d') or item.get('bbox') or item.get('box')
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            try:
                x1 = max(0, min(orig_w - 1, int(round(float(bb[0]) * sx))))
                y1 = max(0, min(orig_h - 1, int(round(float(bb[1]) * sy))))
                x2 = max(0, min(orig_w,     int(round(float(bb[2]) * sx))))
                y2 = max(0, min(orig_h,     int(round(float(bb[3]) * sy))))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            text = (item.get('text_content') or item.get('text') or '').strip()
            if not text:
                continue
            texts.append(text)
            bboxes.append([x1, y1, x2, y2])

        print(
            f"[QwenOCR-full] detect+recognise on {self.device_str}: "
            f"{t1-t0:.2f}s, raw_chars={len(raw)}, "
            f"items_parsed={len(items)}, items_kept={len(texts)}, "
            f"resized={resized_w}x{resized_h}, orig={orig_w}x{orig_h}"
        )
        return texts, bboxes

    async def _vllm_ocr_single(self, crop: Image.Image) -> str:
        """Single crop OCR via vLLM API."""
        import aiohttp
        crop_b64 = self._crop_to_jpeg_base64(crop)
        payload = {
            "model": self.model_path,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                    {"type": "text", "text": "Read all text in this image exactly as it appears. Output only the text, nothing else."}
                ]
            }],
            "max_tokens": 15,
            "temperature": 0,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.vllm_url}/v1/chat/completions", json=payload) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()

    async def _vllm_ocr_batch(self, crops: List[Image.Image]) -> List[str]:
        """Fire ALL crops at vLLM simultaneously. vLLM's continuous batching
        handles the concurrency internally with PagedAttention.

        This is where the 79s -> 1.5s magic happens.
        """
        import aiohttp

        semaphore = asyncio.Semaphore(64)  # Match vLLM --max-num-seqs

        async def ocr_one(session: aiohttp.ClientSession, crop: Image.Image) -> str:
            async with semaphore:
                crop_b64 = self._crop_to_jpeg_base64(crop)
                payload = {
                    "model": self.model_path,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                            {"type": "text", "text": "Read all text in this image exactly as it appears. Output only the text, nothing else."}
                        ]
                    }],
                    "max_tokens": 15,
                    "temperature": 0,
                }
                try:
                    async with session.post(
                        f"{self.vllm_url}/v1/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    print(f"[QwenOCR] vLLM request failed: {e}")
                    return ""

        async with aiohttp.ClientSession() as session:
            tasks = [ocr_one(session, crop) for crop in crops]
            return await asyncio.gather(*tasks)

    def recognize_regions(self, image: Image.Image, bboxes: List[List[int]]) -> List[str]:
        """Recognize text in multiple regions - maximum speed path.

        Strategy:
        1. Check perceptual hash cache for each crop (0ms per hit)
        2. Uncached crops go to batch inference (local HF or vLLM)
        3. Store new results in cache for next frame
        """
        image_rgb = image.convert("RGB")
        t0 = time.perf_counter()

        # Phase 1: Prepare all crops and check cache
        crops = []
        crop_indices = []  # Maps crop list index -> original bbox index
        results = [""] * len(bboxes)
        cache_hits = 0

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, image_rgb.width))
            y1 = max(0, min(y1, image_rgb.height))
            x2 = max(0, min(x2, image_rgb.width))
            y2 = max(0, min(y2, image_rgb.height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb.crop((x1, y1, x2, y2))
            prepared = self._prepare_crop(crop)
            if prepared is None:
                continue

            # Check cache
            if self.hash_cache:
                cached = self.hash_cache.lookup(prepared)
                if cached is not None:
                    results[i] = cached
                    cache_hits += 1
                    continue

            crops.append(prepared)
            crop_indices.append(i)

        t1 = time.perf_counter()

        # Phase 2: Batch inference on uncached crops
        if crops:
            if self.vllm_url:
                # vLLM: fire all at once asynchronously
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already in an async context (FastAPI)
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            batch_results = pool.submit(
                                asyncio.run, self._vllm_ocr_batch(crops)
                            ).result()
                    else:
                        batch_results = loop.run_until_complete(self._vllm_ocr_batch(crops))
                except RuntimeError:
                    batch_results = asyncio.run(self._vllm_ocr_batch(crops))
            else:
                # Local HF: process in batches, with fallback to single-crop on error
                batch_results = []
                for batch_start in range(0, len(crops), self.batch_size):
                    batch = crops[batch_start:batch_start + self.batch_size]
                    try:
                        batch_texts = self._local_recognize_batch(batch)
                        batch_results.extend(batch_texts)
                    except Exception as e:
                        print(f"[QwenOCR] Batch failed ({e}), falling back to single-crop")
                        for crop in batch:
                            try:
                                text = self._local_recognize_single(crop)
                                batch_results.append(text)
                            except Exception:
                                batch_results.append("")
                    # Free KV cache VRAM between batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Map results back and store in cache
            for pos, (idx, text) in enumerate(zip(crop_indices, batch_results)):
                results[idx] = text
                if self.hash_cache and text:
                    self.hash_cache.store(crops[pos], text)

        t2 = time.perf_counter()
        total = len(bboxes)
        inferred = len(crops)
        print(f"[QwenOCR] {total} regions: {cache_hits} cache hits, {inferred} inferred "
              f"| prep={t1-t0:.3f}s, inference={t2-t1:.3f}s, total={t2-t0:.3f}s")
        if self.hash_cache:
            print(f"[QwenOCR] Cache stats: {self.hash_cache.stats()}")

        return results
