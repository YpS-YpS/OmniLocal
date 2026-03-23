@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>nul

echo ============================================
echo   OmniParser + Qwen OCR Installer
echo   Portable Setup - Clone, Install, Done
echo ============================================
echo.

:: ─── Step 1: Find Python ─────────────────────────────────────────────
set PYTHON_CMD=
set PYTHON_VER=

:: Try 'python' first
python --version >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
    set PYTHON_CMD=python
    goto :check_version
)

:: Try 'python3'
python3 --version >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('python3 --version 2^>^&1') do set PYTHON_VER=%%v
    set PYTHON_CMD=python3
    goto :check_version
)

:: Try Windows Python Launcher 'py'
py --version >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=2 delims= " %%v in ('py --version 2^>^&1') do set PYTHON_VER=%%v
    set PYTHON_CMD=py
    goto :check_version
)

echo [ERROR] Python not found!
echo.
echo Please install Python 3.10, 3.11, or 3.12 from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:check_version
echo Found: %PYTHON_CMD% %PYTHON_VER%

:: Extract major.minor
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

:: Check Python >= 3.10 and <= 3.12
if %PY_MAJOR% LSS 3 goto :bad_version
if %PY_MAJOR% GTR 3 goto :bad_version
if %PY_MINOR% LSS 10 goto :bad_version
if %PY_MINOR% GTR 12 goto :bad_version
echo [OK] Python %PYTHON_VER% is supported.
goto :create_venv

:bad_version
echo.
echo [ERROR] Python %PYTHON_VER% is not supported.
echo         Requires Python 3.10, 3.11, or 3.12.
echo         Download from: https://www.python.org/downloads/
pause
exit /b 1

:: ─── Step 2: Create virtual environment ──────────────────────────────
:create_venv
echo.
if exist "%~dp0venv\Scripts\activate.bat" (
    echo [OK] Virtual environment already exists.
) else (
    echo [1/3] Creating virtual environment...
    %PYTHON_CMD% -m venv "%~dp0venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv
call "%~dp0venv\Scripts\activate.bat"
echo [OK] Virtual environment activated.

:: ─── Step 3: Run installer ───────────────────────────────────────────
echo.
echo [2/3] Running installer (this will take a while on first run)...
echo.

:: Set HF cache to project-local directory
set HF_HOME=%~dp0.cache\huggingface
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

python "%~dp0install.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed. Check output above.
    pause
    exit /b 1
)

:: ─── Done ────────────────────────────────────────────────────────────
echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo To start the server:
echo   start_omni_server.bat
echo.
echo To parse a single image:
echo   venv\Scripts\activate.bat
echo   python parse_image.py your_screenshot.png
echo.
pause
