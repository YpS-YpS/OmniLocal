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

    def __init__(self, max_size: int = 2048, hamming_threshold: int = 5):
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
            # FP16: best quality, ~6GB VRAM
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
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
            print(f"[QwenOCR] Perceptual hash cache: ENABLED (max_size=2048, hamming_threshold=5)")

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
