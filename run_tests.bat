@echo off
REM Run tests script for Windows

echo ========================================
echo Running Browser Agent Tests
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run tests
echo Running tests...
pytest tests/ -v --tb=short

echo.
echo ========================================
echo Test Results Complete
echo ========================================
echo.

REM Run with coverage (optional)
set /p COVERAGE="Run with coverage report? (y/n): "
if /i "%COVERAGE%"=="y" (
    echo.
    echo Generating coverage report...
    pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo.
    echo Coverage report generated in htmlcov/index.html
    echo.
    
    set /p OPEN="Open coverage report in browser? (y/n): "
    if /i "%OPEN%"=="y" (
        start htmlcov\index.html
    )
)

pause
