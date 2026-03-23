import logging
import os
import base64

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

from agent.llm_utils.utils import is_image_path, encode_image

###############################################################################
# 1) LOAD THE QWEN MODEL WITH vLLM, ONCE AT IMPORT TIME
###############################################################################
#MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" - UNCOMMENT FOR 3B MODEL
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

logging.info(f"Loading vLLM-based Qwen model from: {MODEL_PATH} ...")

# Create the vLLM LLM instance
# limit_mm_per_prompt is recommended in Qwen’s doc, to limit # of images/video
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct", # Change to 3B if using the 3B model
    max_model_len=32768,                  # or 65536, see what fits
    limit_mm_per_prompt={"image": 4},     # if you only need 1-2 images
    dtype="bfloat16",                      # or "bfloat16" if your GPU supports it
    gpu_memory_utilization=0.95           # let vLLM use ~95% of VRAM
)

# Also create the processor for building prompts + handling images
processor = AutoProcessor.from_pretrained(MODEL_PATH)

logging.info("vLLM Qwen model loaded successfully.")

###############################################################################
# 2) DEFINE THE INFERENCE FUNCTION, ANALOGOUS TO run_local_qwen_interleaved
###############################################################################
def run_vllm_qwen_interleaved(
    messages: list,
    system: str,
    model_name: str,
    api_key: str,      # Not used locally, for signature compatibility
    max_tokens=2048,
    temperature=0.0,
    provider_base_url: str = None,  # ignored locally
):
    """
    vLLM-based Qwen 2.5-VL inference with multi-modal input.
    Returns (generated_text, token_usage).
    """

    # -------------------------------------------------------------------------
    # 1) Convert your agent's messages into Qwen-style chat messages
    # -------------------------------------------------------------------------
    qwen_messages = []
    
    if system.strip():
        qwen_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system.strip()}]
        })

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            qwen_content = []
            for cnt in msg.get("content", []):
                if isinstance(cnt, str):
                    if is_image_path(cnt):
                        img_b64 = encode_image(cnt)
                        qwen_content.append({
                            "type": "image",
                            "image": f"data:image/png;base64,{img_b64}",
                            # optionally min_pixels / max_pixels
                            # "min_pixels": 224 * 224,
                            # "max_pixels": 1280 * 28 * 28
                        })
                    else:
                        qwen_content.append({"type": "text", "text": cnt})
                else:
                    # If it's not a string, treat it as text
                    qwen_content.append({"type": "text", "text": str(cnt)})
            qwen_messages.append({"role": role, "content": qwen_content})
        elif isinstance(msg, str):
            qwen_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": msg}]
            })

    # -------------------------------------------------------------------------
    # 2) Build the prompt using Qwen's apply_chat_template
    # -------------------------------------------------------------------------
    prompt = processor.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 3) Prepare multi-modal data from messages
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        qwen_messages,
        return_video_kwargs=True
    )

    # Build dict for vLLM
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    # 4) Construct final llm_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # pass FPS or other settings from video_kwargs if needed
        "mm_processor_kwargs": video_kwargs,
    }

    # 5) Build SamplingParams
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # repetition_penalty=1.05, # optional
        # top_p=0.9, top_k=40, etc.
        # You can also define stop_token_ids or other vLLM features
    )

    # 6) Run vLLM inference
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    # vLLM returns a list of RequestOutput objects, each with .outputs
    # We usually have exactly 1 request, so outputs[0]
    # Then outputs[0].outputs is a list of candidates; we typically use index 0
    if not outputs or not outputs[0].outputs:
        # safety check
        generated_text = ""
    else:
        generated_text = outputs[0].outputs[0].text

    # 7) (Optional) Count tokens. vLLM does track usage,
    #    but as of now there's no official usage object. You can approximate:
    token_usage = 0
    # For example:
    # token_usage = len(processor.tokenizer(generated_text)["input_ids"])

    return generated_text, token_usage
