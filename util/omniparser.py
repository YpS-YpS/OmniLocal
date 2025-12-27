from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
import torch
from PIL import Image
import io
import base64
from typing import Dict
class Omniparser(object):
    def __init__(self, config: Dict):
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.som_model = get_yolo_model(model_path=config['som_model_path'])
        self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
        print('Omniparser initialized!!!')

    def parse(self, image_base64: str, override_config: dict = None):
        """
        Parse an image and detect UI elements.

        Args:
            image_base64: Base64 encoded image
            override_config: Optional per-request config overrides. Supported keys:
                - BOX_TRESHOLD: YOLO detection confidence (default: from startup config)
                - IOU_THRESHOLD: IOU threshold for overlap removal (default: 0.1)
                - use_paddleocr: Use PaddleOCR vs EasyOCR (default: True)
                - text_threshold: OCR confidence threshold (default: 0.8)
                - use_local_semantics: Use caption model for icons (default: True)
                - scale_img: Scale image before processing (default: False)
                - imgsz: Image size for YOLO (default: None = use original)
        """
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

        # Merge startup config with per-request overrides (overrides take precedence)
        use_paddleocr = override_config.get('use_paddleocr', self.config.get('use_paddleocr', True))
        iou_threshold = override_config.get('IOU_THRESHOLD', self.config.get('IOU_THRESHOLD', 0.1))
        box_threshold = override_config.get('BOX_TRESHOLD', self.config.get('BOX_TRESHOLD', 0.05))
        text_threshold = override_config.get('text_threshold', 0.8)
        use_local_semantics = override_config.get('use_local_semantics', True)
        scale_img = override_config.get('scale_img', False)
        imgsz = override_config.get('imgsz', None)

        print(f'OCR config: use_paddleocr={use_paddleocr}, text_threshold={text_threshold}, box_threshold={box_threshold}, iou_threshold={iou_threshold}')

        (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': text_threshold}, use_paddleocr=use_paddleocr)
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
            batch_size=128
        )

        return dino_labled_img, parsed_content_list

# from util.utils import get_som_labeled_img, get_caption_model_processor, get_yolo_model, check_ocr_box
# import torch
# from PIL import Image
# import io
# import base64
# from typing import Dict
# class Omniparser(object):
#     def __init__(self, config: Dict):
#         self.config = config
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#         self.som_model = get_yolo_model(model_path=config['som_model_path'])
#         self.caption_model_processor = get_caption_model_processor(model_name=config['caption_model_name'], model_name_or_path=config['caption_model_path'], device=device)
#         print('Omniparser initialized!!!')

#     def parse(self, image_base64: str):
#         image_bytes = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_bytes))
#         print('image size:', image.size)
        
#         box_overlay_ratio = max(image.size) / 3200
#         draw_bbox_config = {
#             'text_scale': 0.8 * box_overlay_ratio,
#             'text_thickness': max(int(2 * box_overlay_ratio), 1),
#             'text_padding': max(int(3 * box_overlay_ratio), 1),
#             'thickness': max(int(3 * box_overlay_ratio), 1),
#         }

#         (text, ocr_bbox), _ = check_ocr_box(image, display_img=False, output_bb_format='xyxy', easyocr_args={'text_threshold': 0.8}, use_paddleocr=False)
#         dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image, self.som_model, BOX_TRESHOLD = self.config['BOX_TRESHOLD'], output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=self.caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.1, scale_img=False, batch_size=128)

#         return dino_labled_img, parsed_content_list