@echo off
title DeepFake Investigation System Launcher
color 0B
cls

echo ======================================================================
echo           DEEPFAKE INVESTIGATION SYSTEM - 1-CLICK LAUNCHER
echo ======================================================================
echo.
echo [1/3] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detected.
echo.
echo [2/3] Checking dependencies from requirements.txt...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. Continuing anyway...
) else (
    echo [OK] Dependencies verified.
)

echo.
echo [3/3] Launching Deepfake Investigation System Web Server...
echo.
echo ----------------------------------------------------------------------
echo  Server Address : http://127.0.0.1:5000
echo  Dashboard      : http://127.0.0.1:5000/dashboard
echo  API Endpoint   : http://127.0.0.1:5000/api/predict
echo ----------------------------------------------------------------------
echo.
echo Opening browser in 3 seconds...
start "" http://127.0.0.1:5000

python app.py

pause
