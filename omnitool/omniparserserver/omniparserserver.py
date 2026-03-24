'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --IOU_THRESHOLD 0.1 --use_paddleocr

# Maximum Speed Mode (vLLM + dual GPU):
# python -m omniparserserver --use_qwen_ocr --vllm_url http://localhost:8100 --gpu_detect cuda:1 --port 8000

# Maximum Speed Mode (local HF batched + dual GPU):
# python -m omniparserserver --use_qwen_ocr --gpu_detect cuda:1 --ocr_batch_size 8 --port 8000

# Single GPU speed mode:
# python -m omniparserserver --use_qwen_ocr --use_hash_cache --ocr_batch_size 8 --port 8000
'''

import sys
import os
import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)
from util.omniparser import Omniparser

def parse_arguments():
    parser = argparse.ArgumentParser(description='Omniparser API - Maximum Speed Edition')

    # Model paths
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt')
    parser.add_argument('--caption_model_name', type=str, default='florence2')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence')
    parser.add_argument('--qwen_model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')

    # Device configuration
    parser.add_argument('--device', type=str, default='cuda', help='Default device')
    parser.add_argument('--gpu_ocr', type=str, default='cuda:0', help='GPU for Qwen OCR (RTX 4090)')
    parser.add_argument('--gpu_detect', type=str, default='cuda:0', help='GPU for YOLO + Florence2 (RTX 4080)')

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
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str
    box_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    use_paddleocr: Optional[bool] = None
    use_qwen_ocr: Optional[bool] = None
    text_threshold: Optional[float] = None
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
    return {"message": "Omniparser API ready"}

@app.get("/cache_stats/")
async def cache_stats():
    """Check perceptual hash cache performance."""
    if omniparser.qwen_ocr and omniparser.qwen_ocr.hash_cache:
        return omniparser.qwen_ocr.hash_cache.stats()
    return {"message": "Hash cache not enabled"}

if __name__ == "__main__":
    use_reload = not getattr(args, 'no_reload', False)
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=use_reload)
