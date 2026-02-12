#!/bin/bash
# Installation script for Linux/Mac

set -e

echo "====================================="
echo "Browser Agent Installation Script"
echo "====================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.10+ first"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
python3 -m venv venv

echo "[2/5] Activating virtual environment..."
source venv/bin/activate

echo "[3/5] Upgrading pip..."
pip install --upgrade pip

echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt

echo "[5/5] Installing Playwright browsers..."
playwright install chromium

echo ""
echo "====================================="
echo "Installation Complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env"
echo "2. Edit .env and add your API key"
echo "3. Run: python main.py --query 'your query here'"
echo ""
echo "Example:"
echo "  python main.py --query 'Search Google for Python tutorials'"
echo ""
