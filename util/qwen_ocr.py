import torch
from PIL import Image
from typing import List, Optional
import time


class QwenOCR:
    """Qwen2.5-VL-2B based text recognition for stylized/game UI fonts.

    Uses Qwen2.5-VL-2B-Instruct as a vision-language model to read text
    from cropped image regions. Dramatically more accurate than PaddleOCR/EasyOCR
    on artistic, stylized, or game-specific fonts because it reasons about
    visual appearance + language context jointly.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print(f"[QwenOCR] Loading model from {model_path} on {device}...")
        load_start = time.time()

        if device == "cpu":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float32, device_map="cpu"
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map=device
            )
        self.model.eval()

        # Use lower min_pixels for small text crops, cap max for speed
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=28 * 28 * 4,     # ~56x56 min
            max_pixels=28 * 28 * 256,   # ~448x448 max
        )

        self.device = self.model.device
        self._prompt_text = self._build_prompt()

        print(f"[QwenOCR] Model loaded in {time.time() - load_start:.1f}s on {self.device}")

    def _build_prompt(self) -> str:
        """Build the chat template prompt once for reuse."""
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

    @torch.inference_mode()
    def recognize_crop(self, crop: Image.Image) -> str:
        """Recognize text in a single cropped image region.

        Args:
            crop: PIL Image of the cropped text region

        Returns:
            Recognized text string
        """
        if crop.width < 2 or crop.height < 2:
            return ""

        crop_rgb = crop.convert("RGB")

        # Qwen2.5-VL requires minimum 28px per dimension
        MIN_DIM = 28
        if crop_rgb.width < MIN_DIM or crop_rgb.height < MIN_DIM:
            scale = max(MIN_DIM / crop_rgb.width, MIN_DIM / crop_rgb.height, 1.0)
            new_w = max(int(crop_rgb.width * scale), MIN_DIM)
            new_h = max(int(crop_rgb.height * scale), MIN_DIM)
            crop_rgb = crop_rgb.resize((new_w, new_h), Image.LANCZOS)

        inputs = self.processor(
            text=[self._prompt_text],
            images=[crop_rgb],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )

        # Decode only the generated tokens (skip input)
        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()

    def recognize_regions(self, image: Image.Image, bboxes: List[List[int]]) -> List[str]:
        """Recognize text in multiple regions of a full image.

        Args:
            image: Full PIL Image (screenshot)
            bboxes: List of [x1, y1, x2, y2] bounding boxes in pixel coordinates

        Returns:
            List of recognized text strings, one per bbox
        """
        image_rgb = image.convert("RGB")
        texts = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Clamp to image bounds
            x1 = max(0, min(x1, image_rgb.width))
            y1 = max(0, min(y1, image_rgb.height))
            x2 = max(0, min(x2, image_rgb.width))
            y2 = max(0, min(y2, image_rgb.height))

            if x2 <= x1 or y2 <= y1:
                texts.append("")
                continue

            crop = image_rgb.crop((x1, y1, x2, y2))
            text = self.recognize_crop(crop)
            texts.append(text)

        return texts
