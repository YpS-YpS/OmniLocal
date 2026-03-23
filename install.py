"""OmniParser + Qwen OCR installer.

Called by install.bat after venv is created and activated.
Handles: pip packages, CUDA detection, weights download, model caching.
"""

import os
import sys
import subprocess
import shutil
import urllib.request
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Google Drive file IDs ────────────────────────────────────────────
# Update these if you re-upload to a different location.
GDRIVE_FILES = {
    "weights": {
        "id": "1Otyc6swsZkzNyDHdPvPIXbyCky6QhNkg",
        "dest": os.path.join(SCRIPT_DIR, "weights.zip"),
        "desc": "Model weights (YOLO + Florence2, ~1.1 GB)",
    },
    # Optional: flash_attn wheel for advanced users
    "flash_attn": {
        "id": "1n0l5gP4xmtABP7LsfuIrROf1J3_CKTrg",
        "dest": os.path.join(SCRIPT_DIR, "flash_attn.whl"),
        "desc": "Flash Attention wheel (~128 MB, optional)",
    },
}

# ── HuggingFace models to pre-cache ─────────────────────────────────
HF_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",     # Qwen OCR recognition
    "microsoft/Florence-2-base",          # Florence2 processor files
]


def run(cmd, desc=None, check=True):
    """Run a command and print output."""
    if desc:
        print(f"\n{'─'*50}")
        print(f"  {desc}")
        print(f"{'─'*50}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed: {cmd}")
        return False
    return True


def gdrive_download(file_id, dest, desc=""):
    """Download a file from Google Drive with large-file bypass."""
    if os.path.exists(dest):
        print(f"  [SKIP] {desc} — already downloaded")
        return True

    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    print(f"  Downloading {desc}...")
    print(f"  URL: {url}")

    # Use curl (built into Windows 10+) for progress bar
    result = subprocess.run(
        f'curl -L -o "{dest}" "{url}"',
        shell=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Download failed for {desc}")
        return False
    print(f"  [OK] Downloaded: {dest}")
    return True


def check_weights():
    """Check if weights are present, download if not."""
    weights_dir = os.path.join(SCRIPT_DIR, "weights")
    yolo_model = os.path.join(weights_dir, "icon_detect", "model.pt")
    florence_model = os.path.join(weights_dir, "icon_caption_florence", "model.safetensors")

    if os.path.exists(yolo_model) and os.path.exists(florence_model):
        print("  [OK] Weights already present")
        return True

    # Download weights zip
    winfo = GDRIVE_FILES["weights"]
    if not gdrive_download(winfo["id"], winfo["dest"], winfo["desc"]):
        print("\n  [ERROR] Could not download weights.")
        print("  Manual download: https://drive.google.com/file/d/" + winfo["id"])
        return False

    # Extract
    print("  Extracting weights...")
    import zipfile
    with zipfile.ZipFile(winfo["dest"], "r") as zf:
        zf.extractall(SCRIPT_DIR)
    os.remove(winfo["dest"])
    print("  [OK] Weights extracted")
    return True


def cache_hf_models():
    """Pre-download HuggingFace models to local cache."""
    cache_dir = os.path.join(SCRIPT_DIR, ".cache", "huggingface")
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        print("  [SKIP] huggingface_hub not installed, models will download on first run")
        return True

    # Check if token is available
    api = HfApi()
    token = api.token

    for model_id in HF_MODELS:
        print(f"  Caching {model_id}...")
        try:
            snapshot_download(
                model_id,
                cache_dir=os.path.join(cache_dir, "hub"),
                token=token,
            )
            print(f"  [OK] {model_id} cached")
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                print(f"  [WARN] {model_id} requires authentication.")
                print(f"         Run: python -c \"from huggingface_hub import login; login()\"")
                print(f"         Then re-run install.bat")
            elif "404" in err:
                print(f"  [WARN] {model_id} not found. Check model ID.")
            else:
                print(f"  [WARN] Could not cache {model_id}: {err[:200]}")
                print(f"         Model will download on first server start.")
    return True


def main():
    print(f"Python: {sys.version}")
    print(f"Install dir: {SCRIPT_DIR}")

    # ── 1. Upgrade pip ────────────────────────────────────────────────
    if not run("python -m pip install --upgrade pip", "Upgrading pip"):
        return 1

    # ── 2. Install PyTorch with CUDA ──────────────────────────────────
    # Check if torch already installed with CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[OK] PyTorch {torch.__version__} already installed with CUDA")
        else:
            print(f"\n[WARN] PyTorch {torch.__version__} installed but CUDA not available")
            raise ImportError("Reinstall needed")
    except ImportError:
        if not run(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
            "Installing PyTorch with CUDA 12.8",
        ):
            # Fallback to cu124
            print("  Trying CUDA 12.4...")
            if not run(
                "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
                "Installing PyTorch with CUDA 12.4 (fallback)",
            ):
                print("[ERROR] Could not install PyTorch with CUDA.")
                print("        Install manually: https://pytorch.org/get-started/locally/")
                return 1

    # ── 3. Install requirements ───────────────────────────────────────
    req_file = os.path.join(SCRIPT_DIR, "requirements.txt")
    if not run(f'pip install -r "{req_file}"', "Installing requirements"):
        return 1

    # ── 4. Install Qwen OCR requirements ──────────────────────────────
    qwen_req = os.path.join(SCRIPT_DIR, "requirements-qwen-ocr.txt")
    if os.path.exists(qwen_req):
        if not run(f'pip install -r "{qwen_req}"', "Installing Qwen OCR requirements"):
            return 1

    # ── 5. Download weights ───────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Checking model weights")
    print(f"{'─'*50}")
    check_weights()

    # ── 6. Pre-cache HuggingFace models ───────────────────────────────
    print(f"\n{'─'*50}")
    print("  Pre-caching HuggingFace models")
    print(f"{'─'*50}")
    cache_hf_models()

    # ── 7. Verify installation ────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Verifying installation")
    print(f"{'─'*50}")

    checks = [
        ("PyTorch", "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
        ("Transformers", "import transformers; print(f'  Transformers {transformers.__version__}')"),
        ("PaddleOCR", "import paddleocr; print(f'  PaddleOCR {paddleocr.__version__}')"),
        ("Ultralytics", "import ultralytics; print(f'  Ultralytics {ultralytics.__version__}')"),
        ("Qwen VL Utils", "import qwen_vl_utils; print('  qwen-vl-utils OK')"),
    ]

    all_ok = True
    for name, cmd in checks:
        result = subprocess.run(f'python -c "{cmd}"', shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] {name}: not installed")
            all_ok = False
        else:
            print(result.stdout.strip())

    # Optional: flash_attn
    result = subprocess.run(
        'python -c "import flash_attn; print(f\'  Flash Attention {flash_attn.__version__}\')"',
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("  Flash Attention: not installed (optional, SDPA used instead)")

    if all_ok:
        print("\n[OK] All core packages verified.")
    else:
        print("\n[WARN] Some packages missing. Server may still work.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
