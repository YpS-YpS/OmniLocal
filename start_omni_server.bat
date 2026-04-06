@echo off
REM Dinosaur Eyes 2 Server — Maximum Speed Edition
REM Supports: Standard mode, Qwen OCR, and MAXIMUM SPEED (vLLM + dual GPU)

setlocal enabledelayedexpansion

REM Store the script directory
set "SCRIPT_DIR=%~dp0"

REM Set HuggingFace cache to project-local directory
set HF_HOME=%SCRIPT_DIR%.cache\huggingface
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

REM Get number of instances from user
set /p NUM_INSTANCES="Enter number of Dinosaur Eyes 2 instances to start (1-10) [default: 1]:"
if "%NUM_INSTANCES%"=="" set NUM_INSTANCES=1
if %NUM_INSTANCES% LSS 1 set NUM_INSTANCES=1
if %NUM_INSTANCES% GTR 10 set NUM_INSTANCES=10

REM Get OCR engine choice
echo.
echo =========================================
echo   OCR Engine Options
echo =========================================
echo   1. PaddleOCR (standard text)
echo   2. EasyOCR
echo   3. Qwen2.5-VL local (batched, hash cache, ~5-10s per frame)
echo   4. MAXIMUM SPEED: Qwen2.5-VL + vLLM (requires Docker, ~1-2s per frame)
echo   5. MAXIMUM SPEED: Qwen2.5-VL local + dual GPU (~5-8s per frame)
echo.
set /p OCR_CHOICE="Select OCR engine [3]: "

if "%OCR_CHOICE%"=="" set OCR_CHOICE=3

if "%OCR_CHOICE%"=="1" (
    set "OCR_FLAG=--use_paddleocr"
    set "OCR_NAME=PaddleOCR"
) else if "%OCR_CHOICE%"=="2" (
    set "OCR_FLAG="
    set "OCR_NAME=EasyOCR"
) else if "%OCR_CHOICE%"=="3" (
    set "OCR_FLAG=--use_qwen_ocr --use_hash_cache --ocr_batch_size 8 --no-dual-gpu"
    set "OCR_NAME=Qwen2.5-VL [Speed]"
) else if "%OCR_CHOICE%"=="4" (
    set "OCR_FLAG=--use_qwen_ocr --vllm_url http://localhost:8100 --use_hash_cache"
    set "OCR_NAME=MAX SPEED [vLLM]"
    echo.
    echo Make sure vLLM is running! Use start_vllm.bat first.
    echo NOTE: vLLM GPU is pinned in docker-compose.vllm.yml - verify it matches your biggest GPU.
    echo.
) else if "%OCR_CHOICE%"=="5" (
    set "OCR_FLAG=--use_qwen_ocr --use_hash_cache --ocr_batch_size 8"
    set "OCR_NAME=MAX SPEED [Dual GPU]"
) else (
    set "OCR_FLAG=--use_qwen_ocr --use_hash_cache --ocr_batch_size 8 --no-dual-gpu"
    set "OCR_NAME=Qwen2.5-VL [Speed]"
)

echo.
echo Starting %NUM_INSTANCES% Dinosaur Eyes 2 instance(s) with %OCR_NAME%...
echo.

REM Check if Windows Terminal is available
where wt >nul 2>nul
if %errorlevel% neq 0 (
    echo Windows Terminal not found. Falling back to separate windows...
    for /L %%i in (1,1,%NUM_INSTANCES%) do (
        set /a PORT=8000+%%i-1
        start "DinoEyes2 !PORT!" cmd /k "cd /d !SCRIPT_DIR!omnitool\omniparserserver && python -m omniparserserver !OCR_FLAG! --port !PORT!"
    )
    goto :end
)

REM Launch instances using Windows Terminal
set /a LAST_PORT=8000+%NUM_INSTANCES%-1
echo Launching Windows Terminal with %NUM_INSTANCES% tab(s)...
echo Ports: 8000 to %LAST_PORT%
echo.

if %NUM_INSTANCES%==1 (
    wt -w 0 nt --title "DinoEyes2 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000"
    goto :end
)

if %NUM_INSTANCES%==2 (
    wt -w 0 nt --title "DinoEyes2 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "DinoEyes2 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001"
    goto :end
)

if %NUM_INSTANCES%==3 (
    wt -w 0 nt --title "DinoEyes2 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "DinoEyes2 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "DinoEyes2 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002"
    goto :end
)

if %NUM_INSTANCES%==4 (
    wt -w 0 nt --title "DinoEyes2 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "DinoEyes2 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "DinoEyes2 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002" ; nt --title "DinoEyes2 8003 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8003"
    goto :end
)

if %NUM_INSTANCES% GEQ 5 (
    wt -w 0 nt --title "DinoEyes2 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "DinoEyes2 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "DinoEyes2 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002" ; nt --title "DinoEyes2 8003 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8003" ; nt --title "DinoEyes2 8004 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8004"

    REM Launch remaining instances (6-10) in additional tabs
    if %NUM_INSTANCES% GEQ 6 (
        for /L %%i in (6,1,%NUM_INSTANCES%) do (
            set /a PORT=8000+%%i-1
            timeout /t 1 /nobreak >nul
            wt -w 0 nt --title "DinoEyes2 !PORT! [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port !PORT!"
        )
    )
    goto :end
)

:end
echo.
echo Servers starting... Check Windows Terminal tabs for status.
timeout /t 3 /nobreak >nul
exit
