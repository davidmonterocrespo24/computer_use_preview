#!/bin/bash
# Run tests script for Linux/Mac

echo "========================================"
echo "Running Browser Agent Tests"
echo "========================================"
echo ""

# Activate virtual environment if it exists
if [ -f venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run tests
echo "Running tests..."
pytest tests/ -v --tb=short

echo ""
echo "========================================"
echo "Test Results Complete"
echo "========================================"
echo ""

# Ask about coverage
read -p "Run with coverage report? (y/n): " COVERAGE
if [ "$COVERAGE" = "y" ] || [ "$COVERAGE" = "Y" ]; then
    echo ""
    echo "Generating coverage report..."
    pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo ""
    echo "Coverage report generated in htmlcov/index.html"
    echo ""
    
    read -p "Open coverage report in browser? (y/n): " OPEN
    if [ "$OPEN" = "y" ] || [ "$OPEN" = "Y" ]; then
        # Try different browser commands
        if command -v xdg-open &> /dev/null; then
            xdg-open htmlcov/index.html
        elif command -v open &> /dev/null; then
            open htmlcov/index.html
        else
            echo "Please open htmlcov/index.html manually"
        fi
    fi
fi
