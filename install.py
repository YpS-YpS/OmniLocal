"""Dinosaur Eyes 2 Installer.

Handles: Python validation, pip packages, CUDA detection, weights download,
flash attention (from local .whl), HuggingFace model caching.
Generates installation log.

Standalone: install.bat -> install.py
RPX plugin: rpx_setup executor -> python install.py [--silent] [--log-dir <path>]
"""

import os
import sys
import subprocess
import shutil
import time
import traceback
import argparse
import logging
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()

# -- Google Drive file IDs -------------------------------------------------
GDRIVE_FILES = {
    "weights": {
        "id": "1Otyc6swsZkzNyDHdPvPIXbyCky6QhNkg",
        "dest": SCRIPT_DIR / "weights.zip",
        "desc": "Model weights (YOLO + Florence2, ~1.1 GB)",
    },
}

# -- HuggingFace models to pre-cache ---------------------------------------
HF_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "microsoft/Florence-2-base",
]

# -- Verification checks ----------------------------------------------------
VERIFY_IMPORTS = [
    ("PyTorch",      "import torch; v=torch.__version__; c='YES' if torch.cuda.is_available() else 'NO'; print(f'PyTorch {v}, CUDA: {c}')"),
    ("Transformers", "import transformers; print(f'Transformers {transformers.__version__}')"),
    ("PaddleOCR",    "import paddleocr; print(f'PaddleOCR {paddleocr.__version__}')"),
    ("Ultralytics",  "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')"),
    ("aiohttp",      "import aiohttp; print(f'aiohttp {aiohttp.__version__}')"),
    ("bitsandbytes", "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')"),
]


# ==========================================================================
#  Logging
# ==========================================================================

class InstallLogger:
    """Dual-output logger: console + log file with timestamps."""

    def __init__(self, log_dir: Path = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_dir is None:
            log_dir = SCRIPT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"install_{ts}.log"
        self.fh = open(self.log_path, "w", encoding="utf-8")
        self.pass_count = 0
        self.fail_count = 0
        self.warn_count = 0
        self.steps = []

    def _ts(self):
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, level, msg):
        line = f"[{self._ts()}] [{level}] {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()

    def info(self, msg):
        print(f"  {msg}")
        self._write("INFO", msg)

    def ok(self, msg):
        print(f"  [OK] {msg}")
        self._write("PASS", msg)

    def warn(self, msg):
        print(f"  [WARN] {msg}")
        self._write("WARN", msg)
        self.warn_count += 1

    def fail(self, msg):
        print(f"  [FAIL] {msg}")
        self._write("FAIL", msg)

    def error_trace(self, msg, exc=None):
        """Log error with full stack trace and suggestion."""
        print(f"  [ERROR] {msg}")
        self._write("ERROR", msg)
        if exc:
            tb = traceback.format_exc()
            print(f"  Stack trace:\n{tb}")
            self._write("TRACE", tb)

    def step_start(self, num, total, title):
        header = f"[{num}/{total}] {title}"
        print(f"\n{'='*60}")
        print(f"  {header}")
        print(f"{'='*60}")
        self._write("STEP", header)
        self.steps.append({"title": title, "status": "running", "start": time.time()})

    def step_pass(self):
        if self.steps:
            self.steps[-1]["status"] = "PASS"
            self.steps[-1]["elapsed"] = time.time() - self.steps[-1]["start"]
            self.pass_count += 1

    def step_fail(self):
        if self.steps:
            self.steps[-1]["status"] = "FAIL"
            self.steps[-1]["elapsed"] = time.time() - self.steps[-1]["start"]
            self.fail_count += 1

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  Installation Summary")
        print(f"{'='*60}")
        for s in self.steps:
            icon = "[PASS]" if s["status"] == "PASS" else "[FAIL]"
            elapsed = s.get("elapsed", 0)
            print(f"  {icon} {s['title']} ({elapsed:.1f}s)")
        print(f"\n  Results: {self.pass_count} passed, {self.fail_count} failed, {self.warn_count} warnings")
        print(f"  Log file: {self.log_path}")
        self._write("SUMMARY", f"{self.pass_count} passed, {self.fail_count} failed, {self.warn_count} warnings")

    def close(self):
        self.fh.close()


# ==========================================================================
#  Helpers
# ==========================================================================

def run_cmd(cmd, log: InstallLogger, desc=None, check=True, timeout=600):
    """Run a shell command with real-time output visible in terminal.

    Output flows directly to stdout (no capture), so:
    - When called from rpx_setup's run_live(): output streams through the pipe
      to the terminal, preserving pip progress bars and colored output.
    - When called standalone (python install.py): output goes directly to console.
    """
    if desc:
        log.info(f"Running: {desc}")
    log._write("CMD", cmd)

    try:
        result = subprocess.run(cmd, shell=True, timeout=timeout)
        log._write("EXIT", str(result.returncode))

        if check and result.returncode != 0:
            log.fail(f"Command failed (exit {result.returncode}): {cmd}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.fail(f"Command timed out after {timeout}s: {cmd}")
        return False
    except Exception as e:
        log.error_trace(f"Command exception: {cmd}", exc=e)
        return False


def gdrive_download(file_id, dest: Path, desc, log: InstallLogger):
    """Download a file from Google Drive with large-file bypass."""
    if dest.exists() and dest.stat().st_size > 1000:
        log.ok(f"{desc} -- already downloaded ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    log.info(f"Downloading {desc}...")

    success = run_cmd(
        f'curl -L -o "{dest}" "{url}"',
        log, desc=f"curl download: {desc}", check=True, timeout=600
    )
    if not success or not dest.exists() or dest.stat().st_size < 1000:
        log.fail(f"Download failed for {desc}")
        log.info(f"  Manual download: https://drive.google.com/file/d/{file_id}")
        log.info(f"  Save to: {dest}")
        if dest.exists():
            dest.unlink()
        return False

    log.ok(f"Downloaded: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


# ==========================================================================
#  Installation Steps
# ==========================================================================

def step_pip_upgrade(log: InstallLogger):
    return run_cmd("python -m pip install --upgrade pip", log, "Upgrading pip")


def step_pytorch(log: InstallLogger):
    """Install PyTorch with CUDA. Tries 12.8 then 12.4."""
    try:
        result = subprocess.run(
            'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and "True" in result.stdout:
            ver = result.stdout.strip().split()[0]
            log.ok(f"PyTorch {ver} already installed with CUDA")
            return True
    except Exception:
        pass

    log.info("PyTorch not found or CUDA not available, installing...")

    if run_cmd(
        "pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128",
        log, "Installing PyTorch 2.8.0 + CUDA 12.8", timeout=600
    ):
        return True

    log.warn("CUDA 12.8 failed, trying CUDA 12.4...")
    log.warn("Flash Attention wheel requires CUDA 12.8 -- SDPA will be used instead with CUDA 12.4")
    if run_cmd(
        "pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu124",
        log, "Installing PyTorch 2.8.0 + CUDA 12.4 (fallback)", timeout=600
    ):
        return True

    log.fail("Could not install PyTorch with CUDA")
    log.info("  Suggestion: Install manually from https://pytorch.org/get-started/locally/")
    log.info("  Then re-run this installer.")
    return False


def step_requirements(log: InstallLogger):
    """Install pip requirements."""
    req_file = SCRIPT_DIR / "requirements.txt"
    if not req_file.exists():
        log.fail(f"requirements.txt not found at {req_file}")
        return False
    return run_cmd(f'pip install -r "{req_file}"', log, "Installing requirements.txt", timeout=600)


def step_weights(log: InstallLogger):
    """Download and extract model weights (YOLO + Florence2)."""
    weights_dir = SCRIPT_DIR / "weights"
    yolo_model = weights_dir / "icon_detect" / "model.pt"
    florence_model = weights_dir / "icon_caption_florence" / "model.safetensors"

    if yolo_model.exists() and florence_model.exists():
        log.ok("Model weights already present")
        return True

    winfo = GDRIVE_FILES["weights"]
    if not gdrive_download(winfo["id"], winfo["dest"], winfo["desc"], log):
        return False

    log.info("Extracting weights...")
    try:
        import zipfile
        with zipfile.ZipFile(winfo["dest"], "r") as zf:
            zf.extractall(SCRIPT_DIR)
        winfo["dest"].unlink()
        log.ok("Weights extracted")

        # Verify
        if yolo_model.exists() and florence_model.exists():
            log.ok(f"Verified: YOLO ({yolo_model.stat().st_size / 1024 / 1024:.0f} MB), "
                   f"Florence2 ({florence_model.stat().st_size / 1024 / 1024:.0f} MB)")
            return True
        else:
            log.fail("Extraction completed but expected files not found")
            log.info(f"  Expected: {yolo_model}")
            log.info(f"  Expected: {florence_model}")
            return False
    except Exception as e:
        log.error_trace("Failed to extract weights", exc=e)
        return False


def step_flash_attention(log: InstallLogger):
    """Install Flash Attention from local .whl file (optional).

    The wheel is downloaded by rpx_setup (setup.py) before this runs.
    If no wheel is found locally, skip gracefully — SDPA is used instead.
    """
    # Check if already installed
    result = subprocess.run(
        'python -c "import flash_attn; print(flash_attn.__version__)"',
        shell=True, capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log.ok(f"Flash Attention {result.stdout.strip()} already installed")
        return True

    # Check Python version -- the pre-built wheel is cp312 (Python 3.12) only
    if sys.version_info[:2] != (3, 12):
        log.warn(f"Flash Attention wheel requires Python 3.12 "
                 f"(you have {sys.version_info.major}.{sys.version_info.minor})")
        log.info("  SDPA will be used instead (no performance impact for inference)")
        return True  # Not a hard failure

    # Look for existing .whl files in project directory
    existing_wheels = list(SCRIPT_DIR.glob("flash_attn*.whl"))
    if not existing_wheels:
        log.warn("No flash_attn .whl found in project directory (optional -- SDPA will be used instead)")
        log.info("  The wheel should be downloaded by rpx_setup (setup.py).")
        log.info("  You can also place a flash_attn*.whl file here manually.")
        return True  # Not a hard failure

    wheel_path = existing_wheels[0]
    log.info(f"Found wheel: {wheel_path.name}")

    if run_cmd(f'pip install "{wheel_path}"', log, "Installing Flash Attention wheel", timeout=300):
        log.ok("Flash Attention installed")
        return True
    else:
        log.warn("Flash Attention install failed (optional -- SDPA will be used instead)")
        log.info("  This is normal if your CUDA version doesn't match the wheel.")
        return True  # Not a hard failure


def step_hf_cache(log: InstallLogger):
    """Pre-download HuggingFace models to local cache."""
    cache_dir = SCRIPT_DIR / ".cache" / "huggingface"
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        from huggingface_hub import snapshot_download, HfApi
    except ImportError:
        log.warn("huggingface_hub not installed, models will download on first run")
        return True

    api = HfApi()
    token = api.token
    all_ok = True

    for model_id in HF_MODELS:
        log.info(f"Caching {model_id}...")
        try:
            snapshot_download(
                model_id,
                cache_dir=str(cache_dir / "hub"),
                token=token,
            )
            log.ok(f"{model_id} cached")
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                log.warn(f"{model_id} requires authentication")
                log.info('  Run: python -c "from huggingface_hub import login; login()"')
                log.info("  Then re-run installer.")
            elif "404" in err:
                log.warn(f"{model_id} not found -- check model ID")
            else:
                log.warn(f"Could not cache {model_id}: {err[:200]}")
                log.info("  Model will download on first server start.")
            all_ok = False

    return all_ok


def step_verify(log: InstallLogger):
    """Verify all core packages are importable."""
    all_ok = True

    for name, cmd in VERIFY_IMPORTS:
        result = subprocess.run(
            f'python -c "{cmd}"', shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            log.warn(f"{name}: not importable")
            if result.stderr.strip():
                last = result.stderr.strip().split("\n")[-1]
                log.info(f"  Reason: {last[:200]}")
            all_ok = False
        else:
            log.ok(result.stdout.strip())

    # Flash attention (optional)
    result = subprocess.run(
        'python -c "import flash_attn; print(f\'Flash Attention {flash_attn.__version__}\')"',
        shell=True, capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log.ok(result.stdout.strip())
    else:
        log.info("Flash Attention: not installed (optional, SDPA used instead)")

    return all_ok


# ==========================================================================
#  Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Dinosaur Eyes 2 Installer")
    parser.add_argument("--silent", action="store_true", help="Non-interactive mode")
    parser.add_argument("--log-dir", type=str, default=None, help="Custom log directory")
    parser.add_argument("--skip-hf-cache", action="store_true", help="Skip HuggingFace model caching")
    parser.add_argument("--skip-flash-attn", action="store_true", help="Skip Flash Attention")
    parser.add_argument("--skip-weights", action="store_true", help="Skip weights download (already handled by rpx_setup)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    log = InstallLogger(log_dir)

    log.info(f"Python: {sys.version}")
    log.info(f"Install dir: {SCRIPT_DIR}")
    log.info(f"Platform: {sys.platform}")
    log._write("INFO", f"Arguments: {vars(args)}")

    steps = [
        ("Upgrade pip",              step_pip_upgrade),
        ("Install PyTorch + CUDA",   step_pytorch),
        ("Install requirements",     step_requirements),
        ("Download model weights",   step_weights),
        ("Install Flash Attention",  step_flash_attention),
        ("Cache HuggingFace models", step_hf_cache),
        ("Verify installation",      step_verify),
    ]

    # Filter optional steps
    if args.skip_flash_attn:
        steps = [(t, f) for t, f in steps if "Flash" not in t]
    if args.skip_hf_cache:
        steps = [(t, f) for t, f in steps if "HuggingFace" not in t]
    if args.skip_weights:
        steps = [(t, f) for t, f in steps if "weights" not in t.lower()]

    total = len(steps)
    t_start = time.time()

    for i, (title, func) in enumerate(steps):
        log.step_start(i + 1, total, title)
        try:
            success = func(log)
            if success:
                log.step_pass()
            else:
                log.step_fail()
        except Exception as e:
            log.error_trace(f"Unexpected error in step '{title}'", exc=e)
            log.step_fail()

    elapsed = time.time() - t_start
    log.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.print_summary()
    log.close()

    return 1 if log.fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
