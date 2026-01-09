@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   OmniParser Installation Script
echo   Python 3.12 + CUDA 12.8 + PyTorch 2.8.0
echo ============================================
echo.

:: Check Python version
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.12
    pause
    exit /b 1
)

:: Check if Python 3.12
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Detected Python %PYVER%
echo %PYVER% | findstr /b "3.12" >nul
if errorlevel 1 (
    echo [WARNING] Python 3.12 recommended. You have %PYVER%
    echo           Flash attention wheel may not be compatible.
    echo.
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 1
)

echo.
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing frozen requirements (this may take a while)...
pip install -r requirements-frozen.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo [3/4] Installing flash-attention for faster inference...
echo       Downloading wheel from GitHub releases...

:: Check if wheel already exists
set WHEEL_FILE=flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl
if exist "%WHEEL_FILE%" (
    echo       Found existing wheel file, using it...
) else (
    echo       Downloading flash_attn wheel (122 MB)...
    curl -L -o "%WHEEL_FILE%" "https://github.com/bdashore3/flash-attention/releases/download/v2.8.3/%WHEEL_FILE%"
    if errorlevel 1 (
        echo [WARNING] Could not download flash_attn wheel.
        echo           OmniParser will still work, but slower.
        goto :skip_flash
    )
)

pip install "%WHEEL_FILE%"
if errorlevel 1 (
    echo [WARNING] Failed to install flash_attn.
    echo           OmniParser will still work, but slower.
)
:skip_flash

echo.
echo [4/4] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>nul || echo Flash Attention: Not installed (optional)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo NEXT STEPS:
echo   1. Download model weights from:
echo      https://huggingface.co/microsoft/OmniParser
echo.
echo   2. Place weights in the 'weights' folder:
echo      weights/
echo        icon_detect/
echo        icon_caption_florence/
echo.
echo   3. Run OmniParser server:
echo      python omnitool/omniparserserver/omniparserserver.py --port 8100 --use_paddleocr --no-reload
echo.
pause
