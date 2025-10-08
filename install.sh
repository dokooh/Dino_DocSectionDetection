#!/bin/bash
# Installation script for PDF Section Detection
# Handles NumPy compatibility issues automatically

echo "=== PDF Section Detection - Installation Script ==="
echo

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider creating one: python -m venv venv && source venv/bin/activate"
    echo
fi

# Check Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Check if NumPy 2.x is already installed
numpy_version=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
echo "Current NumPy version: $numpy_version"

if [[ "$numpy_version" == 2.* ]]; then
    echo "⚠️  NumPy 2.x detected - this may cause compatibility issues"
    echo "   Downgrading to NumPy 1.x..."
    pip install "numpy<2.0.0"
fi

echo
echo "Installing dependencies..."

# Try standard requirements first
if pip install -r requirements.txt; then
    echo "✓ Standard installation successful"
else
    echo "❌ Standard installation failed"
    echo "   Trying stable version requirements..."
    
    if pip install -r requirements-stable.txt; then
        echo "✓ Stable installation successful"
    else
        echo "❌ Stable installation failed"
        echo "   Trying minimal requirements..."
        
        if pip install -r requirements-minimal.txt; then
            echo "✓ Minimal installation successful"
        else
            echo "❌ All installation attempts failed"
            echo "   Please check the error messages above"
            exit 1
        fi
    fi
fi

echo
echo "Verifying installation..."

# Test imports
python -c "
try:
    import torch
    import numpy as np
    from PIL import Image
    import pdf2image
    from transformers import AutoImageProcessor, AutoModel
    from sklearn.cluster import DBSCAN
    import cv2
    print('✓ All required modules imported successfully')
    print(f'  - PyTorch: {torch.__version__}')
    print(f'  - NumPy: {np.__version__}')
    print(f'  - OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Installation completed successfully!"
    echo
    echo "Usage examples:"
    echo "  python Dino_sectiondetector.py document.pdf"
    echo "  python Dino_sectiondetector.py document.pdf --output-images ./annotated/"
    echo
    echo "For help: python Dino_sectiondetector.py --help"
else
    echo
    echo "❌ Installation verification failed"
    echo "   Please check the error messages above"
    exit 1
fi