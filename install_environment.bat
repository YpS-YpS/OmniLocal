@echo off
REM ============================================================
REM OmniParser Environment Setup Script
REM ============================================================
REM Requirements:
REM   - Python 3.12 installed and in PATH
REM   - NVIDIA GPU with CUDA 12.8 compatible drivers
REM   - flash_attn wheel file in same directory as this script
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo         OmniParser Environment Setup
echo ============================================================
echo.

REM Check Python version
python --version 2>nul | findstr /C:"3.12" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.12 is required but not found in PATH.
    echo Please install Python 3.12 and try again.
    pause
    exit /b 1
)
echo [OK] Python 3.12 detected

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Check for flash_attn wheel file
set "WHEEL_FILE=%SCRIPT_DIR%flash_attn-2.8.3+cu128torch2.7.0cxx11abifalse-cp312-cp312-win_amd64.whl"
if not exist "%WHEEL_FILE%" (
    echo.
    echo [WARNING] flash_attn wheel file not found in script directory.
    echo Expected: %WHEEL_FILE%
    echo.
    echo The script will attempt to download it from GitHub...
    set "DOWNLOAD_WHEEL=1"
) else (
    echo [OK] flash_attn wheel file found
    set "DOWNLOAD_WHEEL=0"
)

echo.
echo ============================================================
echo Step 1: Installing PyTorch 2.7.1 with CUDA 12.8
echo ============================================================
echo.

pip install torch==2.7.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)
echo [OK] PyTorch installed successfully

echo.
echo ============================================================
echo Step 2: Installing flash_attn
echo ============================================================
echo.

if "%DOWNLOAD_WHEEL%"=="1" (
    echo Downloading flash_attn wheel from GitHub...
    pip install "https://github.com/bdashore3/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"
) else (
    echo Installing flash_attn from local wheel file...
    pip install "%WHEEL_FILE%"
)

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install flash_attn
    pause
    exit /b 1
)
echo [OK] flash_attn installed successfully

echo.
echo ============================================================
echo Step 3: Installing project requirements
echo ============================================================
echo.

if exist "%SCRIPT_DIR%requirements.txt" (
    pip install -r "%SCRIPT_DIR%requirements.txt"
    if %errorlevel% neq 0 (
        echo [WARNING] Some requirements may have failed to install
    ) else (
        echo [OK] Requirements installed successfully
    )
) else (
    echo [WARNING] requirements.txt not found, skipping...
)

echo.
echo ============================================================
echo Step 4: Verifying installation
echo ============================================================
echo.

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print(f'flash_attn: {flash_attn.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo ============================================================
echo                 Installation Complete!
echo ============================================================
echo.
echo You can now run: start_omni_server.bat
echo.
pause
