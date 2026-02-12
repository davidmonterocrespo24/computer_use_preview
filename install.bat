@echo off
REM Installation script for Windows

echo =====================================
echo Browser Agent Installation Script
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

echo [4/5] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [5/5] Installing Playwright browsers...
playwright install chromium
if errorlevel 1 (
    echo [ERROR] Failed to install Playwright browsers
    pause
    exit /b 1
)

echo.
echo =====================================
echo Installation Complete!
echo =====================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Edit .env and add your API key
echo 3. Run: python main.py --query "your query here"
echo.
echo Example:
echo   python main.py --query "Search Google for Python tutorials"
echo.

pause
