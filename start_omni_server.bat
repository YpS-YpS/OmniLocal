@echo off
REM Start Omniparser Server instances in Windows Terminal tabs
REM Prompts for number of instances, OCR engine choice, and creates them on consecutive ports starting from 8000

setlocal enabledelayedexpansion

REM Store the script directory
set "SCRIPT_DIR=%~dp0"

REM Get number of instances from user
set /p NUM_INSTANCES="Enter number of Omniparser instances to start (1-10) [default: 2]: "

REM Validate input
if "%NUM_INSTANCES%"=="" set NUM_INSTANCES=2
if %NUM_INSTANCES% LSS 1 set NUM_INSTANCES=1
if %NUM_INSTANCES% GTR 10 set NUM_INSTANCES=10

REM Get OCR engine choice
echo.
echo OCR Engine Options:
echo   1. PaddleOCR (default - better for standard text)
echo   2. EasyOCR (better for stylized/gaming fonts)
echo.
set /p OCR_CHOICE="Select OCR engine [1]: "

if "%OCR_CHOICE%"=="" set OCR_CHOICE=1
if "%OCR_CHOICE%"=="1" (
    set "OCR_FLAG=--use_paddleocr"
    set "OCR_NAME=PaddleOCR"
) else (
    set "OCR_FLAG="
    set "OCR_NAME=EasyOCR"
)

echo.
echo Starting %NUM_INSTANCES% Omniparser server instance(s) with %OCR_NAME%...
echo NOTE: Per-request OCR config can override this default.
echo.

REM Check if Windows Terminal is available
where wt >nul 2>nul
if %errorlevel% neq 0 (
    echo Windows Terminal not found. Please install Windows Terminal for multi-tab support.
    echo Falling back to separate windows...

    for /L %%i in (1,1,%NUM_INSTANCES%) do (
        set /a PORT=8000+%%i-1
        start "Omniparser !PORT!" cmd /k "cd /d !SCRIPT_DIR!omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port !PORT!"
    )
    goto :end
)

REM Launch instances using Windows Terminal
set /a LAST_PORT=8000+%NUM_INSTANCES%-1
echo Launching Windows Terminal with %NUM_INSTANCES% tab(s)...
echo Ports: 8000 to %LAST_PORT%
echo.

REM Build command dynamically based on instance count
if %NUM_INSTANCES%==1 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000"
    goto :end
)

if %NUM_INSTANCES%==2 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "Omniparser 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001"
    goto :end
)

if %NUM_INSTANCES%==3 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "Omniparser 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "Omniparser 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002"
    goto :end
)

if %NUM_INSTANCES%==4 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "Omniparser 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "Omniparser 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002" ; nt --title "Omniparser 8003 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8003"
    goto :end
)

if %NUM_INSTANCES%==5 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "Omniparser 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "Omniparser 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002" ; nt --title "Omniparser 8003 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8003" ; nt --title "Omniparser 8004 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8004"
    goto :end
)

if %NUM_INSTANCES% GEQ 6 (
    wt -w 0 nt --title "Omniparser 8000 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8000" ; nt --title "Omniparser 8001 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8001" ; nt --title "Omniparser 8002 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8002" ; nt --title "Omniparser 8003 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8003" ; nt --title "Omniparser 8004 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8004" ; nt --title "Omniparser 8005 [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port 8005"

    REM Launch remaining instances (7-10) in additional tabs
    for /L %%i in (7,1,%NUM_INSTANCES%) do (
        set /a PORT=8000+%%i-1
        timeout /t 1 /nobreak >nul
        wt -w 0 nt --title "Omniparser !PORT! [%OCR_NAME%]" cmd /k "cd /d %SCRIPT_DIR%omnitool\omniparserserver && python -m omniparserserver %OCR_FLAG% --port !PORT!"
    )
    goto :end
)

:end
echo.
echo Servers starting... Check Windows Terminal tabs for status.
timeout /t 3 /nobreak >nul
exit
