'''
python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05 --IOU_THRESHOLD 0.1 --use_paddleocr
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
    parser = argparse.ArgumentParser(description='Omniparser API')
    parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
    parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
    parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
    parser.add_argument('--IOU_THRESHOLD', type=float, default=0.1, help='IoU threshold for non-maximum suppression')
    parser.add_argument('--use_paddleocr', action='store_true', help='Use PaddleOCR instead of EasyOCR')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload (recommended for production/service manager)')
    args = parser.parse_args()
    return args

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str
    # Per-request OCR configuration overrides (optional - defaults to server startup config)
    box_threshold: Optional[float] = None  # YOLO detection confidence threshold
    iou_threshold: Optional[float] = None  # IOU threshold for overlap removal
    use_paddleocr: Optional[bool] = None   # True = PaddleOCR, False = EasyOCR
    text_threshold: Optional[float] = None # OCR confidence threshold for text detection
    use_local_semantics: Optional[bool] = None  # Use caption model for icon labeling
    scale_img: Optional[bool] = None       # Scale image before processing
    imgsz: Optional[int] = None            # Image size for YOLO model

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    print('start parsing...')
    start = time.time()

    # Build override config from request (only include non-None values)
    override_config = {}
    if parse_request.box_threshold is not None:
        override_config['BOX_TRESHOLD'] = parse_request.box_threshold
    if parse_request.iou_threshold is not None:
        override_config['IOU_THRESHOLD'] = parse_request.iou_threshold
    if parse_request.use_paddleocr is not None:
        override_config['use_paddleocr'] = parse_request.use_paddleocr
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

if __name__ == "__main__":
    # Use reload=False when --no-reload is passed (recommended for service manager)
    use_reload = not getattr(args, 'no_reload', False)
    uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=use_reload)


# '''
# python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda --BOX_TRESHOLD 0.05
# '''

# import sys
# import os
# import time
# from fastapi import FastAPI
# from pydantic import BaseModel
# import argparse
# import uvicorn
# root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(root_dir)
# from util.omniparser import Omniparser

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Omniparser API')
#     parser.add_argument('--som_model_path', type=str, default='../../weights/icon_detect/model.pt', help='Path to the som model')
#     parser.add_argument('--caption_model_name', type=str, default='florence2', help='Name of the caption model')
#     parser.add_argument('--caption_model_path', type=str, default='../../weights/icon_caption_florence', help='Path to the caption model')
#     parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
#     parser.add_argument('--BOX_TRESHOLD', type=float, default=0.05, help='Threshold for box detection')
#     parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
#     parser.add_argument('--port', type=int, default=8000, help='Port for the API')
#     args = parser.parse_args()
#     return args

# args = parse_arguments()
# config = vars(args)

# app = FastAPI()
# omniparser = Omniparser(config)

# class ParseRequest(BaseModel):
#     base64_image: str

# @app.post("/parse/")
# async def parse(parse_request: ParseRequest):
#     print('start parsing...')
#     start = time.time()
#     dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
#     latency = time.time() - start
#     print('time:', latency)
#     return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

# @app.get("/probe/")
# async def root():
#     return {"message": "Omniparser API ready"}

# if __name__ == "__main__":
#     uvicorn.run("omniparserserver:app", host=args.host, port=args.port, reload=True)