@echo off
REM Launch vLLM server for maximum-speed Qwen2.5-VL-3B OCR
REM Requires: Docker Desktop with WSL2 backend + NVIDIA Container Toolkit
REM GPU 0 (RTX 4090) runs vLLM, GPU 1 (RTX 4080) runs OmniParser detection

echo ============================================
echo  vLLM OCR Server - Maximum Speed Mode
echo ============================================
echo.
echo GPU Assignment:
echo   GPU 0 (RTX 4090) = vLLM Qwen2.5-VL-3B OCR engine
echo   GPU 1 (RTX 4080) = OmniParser YOLO + Florence2
echo.
echo Configuration:
echo   - Prefix caching: ENABLED
echo   - CUDA graphs: -O2
echo   - Max concurrent sequences: 64
echo   - Max pixels per crop: 12544 (16 visual tokens)
echo   - Max output tokens: 15
echo   - PagedAttention: ENABLED
echo.

REM Check Docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Docker not found. Install Docker Desktop with WSL2 backend.
    echo https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

REM Check NVIDIA Docker runtime
docker info 2>nul | findstr /i "nvidia" >nul 2>nul
if %errorlevel% neq 0 (
    echo WARNING: NVIDIA container runtime not detected.
    echo Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    echo.
)

echo Starting vLLM server...
docker compose -f docker-compose.vllm.yml up -d

echo.
echo Waiting for vLLM to load model (this takes 30-60s first time)...
echo.

:wait_loop
timeout /t 5 /nobreak >nul
curl -s http://localhost:8100/health >nul 2>nul
if %errorlevel% neq 0 (
    echo   Still loading...
    goto wait_loop
)

echo.
echo ============================================
echo  vLLM server is READY at http://localhost:8100
echo ============================================
echo.
echo Now start OmniParser with vLLM mode:
echo   start_omni_server.bat (select option 4: Maximum Speed)
echo.
echo Or manually:
echo   python -m omniparserserver --use_qwen_ocr --vllm_url http://localhost:8100 --gpu_detect cuda:1 --port 8000
echo.
pause
