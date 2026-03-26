from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box, predict_yolo
import torch
import time
from PIL import Image
import io
import base64
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Dual-GPU support: Qwen OCR on the bigger GPU, YOLO+Florence2 on the smaller
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.dual_gpu = False

        # Check if user explicitly set GPU assignments
        user_set_ocr = 'gpu_ocr' in config and config['gpu_ocr'] != 'cuda:0'
        user_set_detect = 'gpu_detect' in config and config['gpu_detect'] != 'cuda:0'

        if num_gpus >= 2 and not user_set_ocr and not user_set_detect:
            # Auto-detect: put Qwen (heavy) on the GPU with more VRAM
            vram = []
            for i in range(num_gpus):
                mem_free, mem_total = torch.cuda.mem_get_info(i)
                name = torch.cuda.get_device_name(i)
                vram.append((mem_total, i, name))
                print(f'[Omniparser] GPU {i}: {name}, {mem_total/1024**3:.1f}GB total')
            vram.sort(reverse=True)  # Biggest first
            big_gpu = f'cuda:{vram[0][1]}'
            small_gpu = f'cuda:{vram[1][1]}'
            self.gpu_ocr = big_gpu      # Qwen on the big GPU (needs 6GB+ model + KV cache)
            self.gpu_detect = small_gpu  # YOLO+Florence2 on smaller GPU (needs ~3GB)
            self.dual_gpu = True
            print(f'[Omniparser] Auto-assigned: OCR -> {big_gpu} ({vram[0][2]}), '
                  f'Detection -> {small_gpu} ({vram[1][2]})')
        elif num_gpus >= 2:
            self.gpu_ocr = config.get('gpu_ocr', 'cuda:0')
            self.gpu_detect = config.get('gpu_detect', 'cuda:1')
            self.dual_gpu = self.gpu_ocr != self.gpu_detect
        else:
            self.gpu_ocr = 'cuda:0' if num_gpus > 0 else 'cpu'
            self.gpu_detect = self.gpu_ocr
            if config.get('gpu_detect', 'cuda:0') != config.get('gpu_ocr', 'cuda:0'):
                print(f'[Omniparser] Only {num_gpus} GPU(s), forcing single-GPU mode')

        print(f'[Omniparser] Device config: ocr={self.gpu_ocr}, detection={self.gpu_detect}, dual={self.dual_gpu}')

        # Load YOLO
        self.som_model = get_yolo_model(model_path=config['som_model_path'])

        # Load Florence2 on detection GPU
        caption_device = self.gpu_detect if self.gpu_detect != 'cpu' else self.device
        self.caption_model_processor = get_caption_model_processor(
            model_name=config['caption_model_name'],
            model_name_or_path=config['caption_model_path'],
            device=caption_device
        )

        # Initialize Qwen2.5-VL OCR on OCR GPU
        self.qwen_ocr = None
        if config.get('use_qwen_ocr', False):
            from util.qwen_ocr import QwenOCR
            qwen_model_path = config.get('qwen_model_path', 'Qwen/Qwen2.5-VL-3B-Instruct')
            vllm_url = config.get('vllm_url', None)
            use_hash_cache = config.get('use_hash_cache', True)
            ocr_batch_size = config.get('ocr_batch_size', 8)

            ocr_device = self.gpu_ocr if self.gpu_ocr != 'cpu' else self.device
            quantize = config.get('quantize', None)
            self.qwen_ocr = QwenOCR(
                model_path=qwen_model_path,
                device=ocr_device,
                use_hash_cache=use_hash_cache,
                vllm_url=vllm_url,
                batch_size=ocr_batch_size,
                quantize=quantize,
            )

        # Thread pool for parallel GPU execution
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Determine active OCR engine
        if self.qwen_ocr is not None:
            ocr_engine = 'Qwen2.5-VL-3B (DE2)'
        elif config.get('use_paddleocr', False):
            ocr_engine = 'PaddleOCR (legacy)'
        else:
            ocr_engine = 'EasyOCR (legacy)'

        print(f'[Omniparser] ╔══════════════════════════════════════════════╗')
        print(f'[Omniparser] ║  OCR Engine: {ocr_engine:<32s}║')
        print(f'[Omniparser] ║  Dual GPU:   {"YES" if self.dual_gpu else "NO":<32s}║')
        print(f'[Omniparser] ║  Hash Cache: {"YES" if config.get("use_hash_cache", True) else "NO":<32s}║')
        print(f'[Omniparser] ║  vLLM:       {"YES" if config.get("vllm_url") else "NO":<32s}║')
        print(f'[Omniparser] ╚══════════════════════════════════════════════╝')

    def _run_yolo(self, image, box_threshold, scale_img, imgsz):
        """Run YOLO detection on the detection GPU. Called from thread."""
        t0 = time.perf_counter()
        if not imgsz:
            w, h = image.size
            imgsz = (h, w)
        xyxy, logits, phrases = predict_yolo(
            model=self.som_model,
            image=image,
            box_threshold=box_threshold,
            imgsz=imgsz,
            scale_img=scale_img,
            iou_threshold=0.1,
            device=self.gpu_detect,
        )
        t1 = time.perf_counter()
        print(f'[Omniparser] YOLO detection on {self.gpu_detect}: {t1-t0:.3f}s ({len(xyxy)} boxes)')
        return xyxy, logits, phrases

    def _run_ocr(self, image, text_threshold, use_paddleocr, qwen_ocr_instance):
        """Run OCR (PaddleOCR detection + Qwen recognition) on the OCR GPU. Called from thread."""
        t0 = time.perf_counter()
        try:
            (text, ocr_bbox), _ = check_ocr_box(
                image, display_img=False, output_bb_format='xyxy',
                easyocr_args={'text_threshold': text_threshold},
                use_paddleocr=use_paddleocr, qwen_ocr=qwen_ocr_instance
            )
        except Exception as e:
            print(f'[Omniparser] OCR failed on {self.gpu_ocr}: {e} — returning empty results')
            return [], []
        t1 = time.perf_counter()
        print(f'[Omniparser] OCR on {self.gpu_ocr}: {t1-t0:.3f}s ({len(text)} texts)')
        return text, ocr_bbox

    def parse(self, image_base64: str, override_config: dict = None):
        if override_config is None:
            override_config = {}

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        print('image size:', image.size)

        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        use_paddleocr = override_config.get('use_paddleocr', self.config.get('use_paddleocr', True))
        use_qwen_ocr = override_config.get('use_qwen_ocr', self.config.get('use_qwen_ocr', False))
        iou_threshold = override_config.get('IOU_THRESHOLD', self.config.get('IOU_THRESHOLD', 0.1))
        box_threshold = override_config.get('BOX_TRESHOLD', self.config.get('BOX_TRESHOLD', 0.05))
        text_threshold = override_config.get('text_threshold', 0.8)
        use_local_semantics = override_config.get('use_local_semantics', True)
        scale_img = override_config.get('scale_img', False)
        imgsz = override_config.get('imgsz', None)

        qwen_ocr_instance = self.qwen_ocr if use_qwen_ocr and self.qwen_ocr is not None else None
        ocr_mode = 'qwen_vlm' if qwen_ocr_instance else ('paddleocr' if use_paddleocr else 'easyocr')
        print(f'OCR config: mode={ocr_mode}, box_threshold={box_threshold}, dual_gpu={self.dual_gpu}')

        t_start = time.perf_counter()

        # ============================================================
        # PARALLEL EXECUTION: YOLO (gpu_detect) || OCR (gpu_ocr)
        # These are independent -- YOLO finds UI element boxes,
        # OCR finds text boxes. They don't need each other's results.
        # ============================================================
        if self.dual_gpu:
            # PIL Images are NOT thread-safe -- copy for each thread
            image_for_yolo = image.copy()
            image_for_ocr = image.copy()

            # Fire both at the same time on different GPUs
            yolo_future = self._executor.submit(
                self._run_yolo, image_for_yolo, box_threshold, scale_img, imgsz
            )
            ocr_future = self._executor.submit(
                self._run_ocr, image_for_ocr, text_threshold, use_paddleocr, qwen_ocr_instance
            )

            # Wait for both to complete
            yolo_result = yolo_future.result()
            text, ocr_bbox = ocr_future.result()
        else:
            # Single GPU: run sequentially (can't share GPU between threads safely)
            text, ocr_bbox = self._run_ocr(image, text_threshold, use_paddleocr, qwen_ocr_instance)
            yolo_result = self._run_yolo(image, box_threshold, scale_img, imgsz)

        t_parallel = time.perf_counter()

        # ============================================================
        # MERGE + FLORENCE2 CAPTIONING (sequential, needs both results)
        # ============================================================
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, self.som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=use_local_semantics,
            iou_threshold=iou_threshold,
            scale_img=scale_img,
            imgsz=imgsz,
            batch_size=128,
            precomputed_yolo=yolo_result,  # Skip YOLO re-run, already done in parallel
        )

        t_end = time.perf_counter()
        print(f'[Omniparser] Timing: parallel(YOLO+OCR)={t_parallel-t_start:.3f}s, '
              f'merge+caption={t_end-t_parallel:.3f}s, total={t_end-t_start:.3f}s')

        return dino_labled_img, parsed_content_list
