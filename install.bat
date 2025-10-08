@echo off
REM Installation script for PDF Section Detection (Windows)
REM Handles NumPy compatibility issues automatically

echo === PDF Section Detection - Installation Script ===
echo.

REM Check if we're in a virtual environment
if defined VIRTUAL_ENV (
    echo ✓ Virtual environment detected: %VIRTUAL_ENV%
) else (
    echo ⚠️  Warning: Not in a virtual environment
    echo    Consider creating one: python -m venv venv ^&^& venv\Scripts\activate
    echo.
)

REM Check Python version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo Python version: %python_version%

REM Check NumPy version
for /f "tokens=*" %%i in ('python -c "import numpy; print(numpy.__version__)" 2^>nul') do set numpy_version=%%i
if "%numpy_version%"=="" set numpy_version=not installed
echo Current NumPy version: %numpy_version%

REM Check if NumPy 2.x is installed
echo %numpy_version% | findstr /b "2." >nul
if %errorlevel%==0 (
    echo ⚠️  NumPy 2.x detected - this may cause compatibility issues
    echo    Downgrading to NumPy 1.x...
    pip install "numpy<2.0.0"
)

echo.
echo Installing dependencies...

REM Try standard requirements first
pip install -r requirements.txt
if %errorlevel%==0 (
    echo ✓ Standard installation successful
) else (
    echo ❌ Standard installation failed
    echo    Trying stable version requirements...
    
    pip install -r requirements-stable.txt
    if %errorlevel%==0 (
        echo ✓ Stable installation successful
    ) else (
        echo ❌ Installation failed
        echo    Please check the error messages above
        exit /b 1
    )
)

echo.
echo Verifying installation...

REM Test imports
python -c "try: import torch, numpy as np; from PIL import Image; import pdf2image; from transformers import AutoImageProcessor, AutoModel; from sklearn.cluster import DBSCAN; import cv2; print('✓ All required modules imported successfully'); print(f'  - PyTorch: {torch.__version__}'); print(f'  - NumPy: {np.__version__}'); print(f'  - OpenCV: {cv2.__version__}'); except ImportError as e: print(f'❌ Import error: {e}'); exit(1)"

if %errorlevel%==0 (
    echo.
    echo 🎉 Installation completed successfully!
    echo.
    echo Usage examples:
    echo   python Dino_sectiondetector.py document.pdf
    echo   python Dino_sectiondetector.py document.pdf --output-images ./annotated/
    echo.
    echo For help: python Dino_sectiondetector.py --help
) else (
    echo.
    echo ❌ Installation verification failed
    echo    Please check the error messages above
    exit /b 1
)

pause