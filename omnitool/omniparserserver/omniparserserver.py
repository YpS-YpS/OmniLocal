'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --IOU_THRESHOLD 0.1 --use_paddleocr

# Maximum Speed Mode (vLLM + dual GPU, auto-detects GPU assignment):
# python -m omniparserserver --use_qwen_ocr --vllm_url http://localhost:8100 --port 8000

# Maximum Speed Mode (local HF batched + dual GPU, auto-detects GPU assignment):
# python -m omniparserserver --use_qwen_ocr --ocr_batch_size 8 --port 8000

# Single GPU speed mode (force single GPU even with multiple GPUs):
# python -m omniparserserver --use_qwen_ocr --use_hash_cache --ocr_batch_size 8 --no-dual-gpu --port 8000
'''

import os
# Determinism env vars must be set BEFORE any CUDA/torch import. cuBLAS reads
# CUBLAS_WORKSPACE_CONFIG at first cuBLAS handle creation — if torch initializes
# CUDA before this line, the value is ignored. See pytorch/pytorch docs notes on
# randomness + ray-project/ray#47690 for the leading-colon requirement.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('FLAGS_cudnn_deterministic', 'True')
os.environ.setdefault('FLAGS_cudnn_exhaustive_search', 'False')
# Reduce VRAM fragmentation; helps the full-image Qwen OCR path which
# allocates large attention buffers on top of the steady-state model.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn

# Lock torch determinism after env vars but before model construction. warn_only
# because Florence-2 uses bilinear interpolate which has no deterministic CUDA
# kernel — we accept a warning rather than crash the server on first call.
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Dinosaur Eyes 2 API - Maximum Speed Edition')

    # Model paths
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt')
    parser.add_argument('--caption_model_name', type=str, default='florence2')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence')
    parser.add_argument('--qwen_model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda', help='Default device')
    parser.add_argument('--gpu_ocr', type=str, default=None, help='GPU for Qwen OCR (auto-detects biggest GPU if not set)')
    parser.add_argument('--gpu_detect', type=str, default=None, help='GPU for YOLO + Florence2 (auto-detects smallest GPU if not set)')
    parser.add_argument('--no-dual-gpu', action='store_true', help='Force single-GPU mode even with multiple GPUs')

    # Detection thresholds
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05)
    parser.add_argument('--IOU_THRESHOLD', type=float, default=0.1)

    # OCR engine selection
    parser.add_argument('--use_paddleocr', action='store_true', help='Use PaddleOCR')
    parser.add_argument('--use_qwen_ocr', action='store_true', help='Use Qwen2.5-VL for OCR')

    # Speed optimizations
    parser.add_argument('--vllm_url', type=str, default=None,
                        help='vLLM server URL (e.g. http://localhost:8100). Enables async batch OCR.')
    parser.add_argument('--use_hash_cache', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable/disable perceptual hash cache (default: on)')
    parser.add_argument('--ocr_batch_size', type=int, default=8,
                        help='Batch size for local HF Qwen inference (default: 8)')
    parser.add_argument('--quantize', type=str, default=None, choices=['int4', 'int8'],
                        help='Quantize Qwen model: int4 (~2GB VRAM) or int8 (~4GB VRAM)')

    # Server config
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-reload', action='store_true')

    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()

# Avoid loading Omniparser twice when uvicorn re-imports this module.
# `python -m omniparserserver` runs the file as `__main__`; uvicorn.run()
# then imports the file AGAIN under the name `omniparserserver` to construct
# `app` — so module-level model loading would happen twice and double VRAM.
# We share a single instance via sys.modules so both namespaces see the same
# loaded models.
_OMNIP_KEY = '_DE2_OMNIPARSER_SINGLETON'
if _OMNIP_KEY in os.environ:
    # Should not happen via env; placeholder for future.
    pass
import builtins as _b
if hasattr(_b, _OMNIP_KEY):
    omniparser = getattr(_b, _OMNIP_KEY)
    print('[DE2] Reusing already-loaded Omniparser instance (skipped duplicate load).', flush=True)
else:
    omniparser = Omniparser(config)
    setattr(_b, _OMNIP_KEY, omniparser)

class ParseRequest(BaseModel):
    base64_image: str
    # YOLO icon detector confidence floor. Lower → more icons (slider explosion
    # on gradient bars). Higher → fewer (might miss real icons). Default 0.05.
    box_threshold: Optional[float] = None
    # IoU threshold for the merge step (remove_overlap_new). Lower → more
    # aggressive dedupe of overlapping boxes. Higher → keep more. Default 0.1.
    iou_threshold: Optional[float] = None
    use_paddleocr: Optional[bool] = None
    use_qwen_ocr: Optional[bool] = None
    # qwen_full_ocr: when True (and Qwen OCR is loaded), Qwen2.5-VL detects AND
    # recognises text in a single grounding call, fully replacing PaddleOCR's
    # detection step. Slower per request (~10-60s vs ~0.5s) but avoids
    # PaddleOCR failure modes. Use sparingly, ideally only on screens where
    # the hybrid path is known to club labels.
    qwen_full_ocr: Optional[bool] = None
    # OCR text-recognition confidence threshold. ONLY active in pure-Paddle
    # path (use_paddleocr=True without qwen_ocr). Has NO effect when
    # use_qwen_ocr=True because Qwen handles recognition without this knob.
    text_threshold: Optional[float] = None
    # PaddleOCR DBNet text-detection knobs (active in BOTH hybrid and
    # pure-Paddle paths; this is the layer that produces the bbox polygons
    # before recognition runs):
    #   det_db_thresh        — pixel-binarisation threshold (default 0.3)
    #                          lower → more pixels considered text → larger
    #                          regions but more noise.
    #   det_db_box_thresh    — minimum confidence to keep a detected box
    #                          (default 0.6 in PaddleOCR's runtime, 0.5 in
    #                          some docs). Lower → keep more marginal boxes.
    #                          Useful when small UI labels get dropped.
    #   det_db_unclip_ratio  — box expansion factor (default 1.5). Lower
    #                          → tighter boxes (fewer mergers between
    #                          adjacent labels — fixes Cyberpunk top-tab
    #                          clubbing). Higher → fatter boxes that may
    #                          engulf glow / stylised glyph edges.
    #   det_db_score_mode    — 'fast' (bbox-mean) or 'slow' (polygon-mean).
    #                          'slow' is more stable for irregular text.
    det_db_thresh: Optional[float] = None
    det_db_box_thresh: Optional[float] = None
    det_db_unclip_ratio: Optional[float] = None
    det_db_score_mode: Optional[str] = None
    use_local_semantics: Optional[bool] = None
    scale_img: Optional[bool] = None
    imgsz: Optional[int] = None

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()

    override_config = {}
    if parse_request.box_threshold is not None:
        override_config['BOX_TRESHOLD'] = parse_request.box_threshold
    if parse_request.iou_threshold is not None:
        override_config['IOU_THRESHOLD'] = parse_request.iou_threshold
    if parse_request.use_paddleocr is not None:
        override_config['use_paddleocr'] = parse_request.use_paddleocr
    if parse_request.use_qwen_ocr is not None:
        override_config['use_qwen_ocr'] = parse_request.use_qwen_ocr
    if parse_request.qwen_full_ocr is not None:
        override_config['qwen_full_ocr'] = parse_request.qwen_full_ocr
    if parse_request.det_db_thresh is not None:
        override_config['det_db_thresh'] = parse_request.det_db_thresh
    if parse_request.det_db_box_thresh is not None:
        override_config['det_db_box_thresh'] = parse_request.det_db_box_thresh
    if parse_request.det_db_unclip_ratio is not None:
        override_config['det_db_unclip_ratio'] = parse_request.det_db_unclip_ratio
    if parse_request.det_db_score_mode is not None:
        override_config['det_db_score_mode'] = parse_request.det_db_score_mode
    if parse_request.text_threshold is not None:
        override_config['text_threshold'] = parse_request.text_threshold
    if parse_request.use_local_semantics is not None:
        override_config['use_local_semantics'] = parse_request.use_local_semantics
    if parse_request.scale_img is not None:
        override_config['scale_img'] = parse_request.scale_img
    if parse_request.imgsz is not None:
        override_config['imgsz'] = parse_request.imgsz

    if override_config:
        print(f'Using override config: {override_config}')

    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image, override_config)
    latency = time.time() - start
    print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency, 'config_used': override_config or 'defaults'}

@app.get("/probe/")
async def root():
    return {"message": "Dinosaur Eyes 2 API ready"}

@app.get("/cache_stats/")
async def cache_stats():
    """Check perceptual hash cache performance."""
    if omniparser.qwen_ocr and omniparser.qwen_ocr.hash_cache:
        return omniparser.qwen_ocr.hash_cache.stats()
    return {"message": "Hash cache not enabled"}

if __name__ == "__main__":
    use_reload = not getattr(args, 'no_reload', False)
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=use_reload)
