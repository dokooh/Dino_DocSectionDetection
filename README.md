# PDF Section Detection with DINO

A Python application that automatically detects and classifies sections, subsections, and tables in PDF documents using DINOv2 (DINO with Vision Transformers) for visual embeddings and advanced computer vision techniques.

## Features

- **PDF to Image Conversion**: Converts PDF pages to high-resolution images for processing
- **Color-Based Region Detection**: Identifies non-white (colored) regions using OpenCV morphological operations
- **DINO Embeddings**: Extracts visual embeddings for each colored region using Facebook's DINOv2 model
- **Section Classification**: Automatically classifies regions as:
  - **Sections**: Main headers and primary content blocks
  - **Subsections**: Secondary headers and nested content
  - **Tables**: Structured data and tabular content
- **Similarity Clustering**: Groups similar sections across pages using DBSCAN clustering
- **Visual Output**: Creates annotated images with bounding boxes and labels for each detected section
- **Flexible Output**: Export results to JSON format with detailed metadata

## Classification Logic

The system uses intelligent heuristics based on:
- **Size and aspect ratio**: Tables tend to be wider, headers more compact
- **Position on page**: Full-width regions are likely main sections
- **Average color intensity**: Darker colors often indicate headers
- **Relative dimensions**: Width relative to page size helps distinguish section types

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Poppler (for PDF processing)

### System Dependencies

**Windows:**
```bash
# Download and install poppler-utils from: https://blog.alivate.com.au/poppler-windows/
# Add poppler/bin to your PATH environment variable
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Dino_DocSectionDetection
```

2. Install Python dependencies:

**Option A - Standard Installation:**
```bash
pip install -r requirements.txt
```

**Option B - If you encounter NumPy compatibility issues:**
```bash
pip install -r requirements-stable.txt
```

**Option C - Manual NumPy fix (if needed):**
```bash
# First, ensure NumPy 1.x is installed
pip install "numpy<2.0.0"
# Then install other requirements
pip install -r requirements.txt
```

### NumPy Compatibility Note

This project requires NumPy 1.x due to compatibility issues with some dependencies. If you encounter errors like:
```
AttributeError: _ARRAY_API not found
```
or
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Use one of these solutions:
- Use `requirements-stable.txt` for fixed versions
- Downgrade NumPy: `pip install "numpy<2.0.0"`
- Create a fresh virtual environment and install dependencies

## Usage

### Command Line Interface

The script now supports command-line arguments for easy configuration:

#### Basic Usage
```bash
# Process a PDF with default settings
python Dino_sectiondetector.py document.pdf

# Specify output file
python Dino_sectiondetector.py document.pdf -o my_results.json

# Use a different model
python Dino_sectiondetector.py document.pdf -m facebook/dinov2-large
```

#### Advanced Usage
```bash
# Customize detection parameters
python Dino_sectiondetector.py document.pdf \
  --dpi 150 \
  --white-threshold 220 \
  --min-area 500 \
  --eps 0.3

# Skip clustering and save page images
python Dino_sectiondetector.py document.pdf \
  --no-clustering \
  --save-images ./page_images/ \
  --verbose

# Create annotated images with bounding boxes
python Dino_sectiondetector.py document.pdf \
  --output-images ./annotated/ \
  --verbose

# Full example with all parameters
python Dino_sectiondetector.py document.pdf \
  -o results.json \
  -m facebook/dinov2-large \
  --dpi 300 \
  --white-threshold 240 \
  --min-area 1000 \
  --eps 0.5 \
  --verbose
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `input_pdf` | Path to input PDF file | Required |
| `-o, --output` | Output JSON file path | `{input}_sections.json` |
| `-m, --model` | DINO model to use | `facebook/dinov2-base` |
| `--dpi` | DPI for PDF conversion | 300 |
| `--white-threshold` | White pixel threshold (0-255) | 240 |
| `--min-area` | Minimum area for region detection | 1000 |
| `--eps` | DBSCAN clustering epsilon | 0.5 |
| `--no-clustering` | Skip similarity clustering | False |
| `--verbose` | Enable detailed output | False |
| `--save-images` | Directory to save page images | None |
| `--output-images` | Directory to save annotated images | None |

### Python API Usage

You can still use the classes directly in your Python code:

```python
from Dino_sectiondetector import PDFSectionDetector

# Initialize detector with DINOv2
detector = PDFSectionDetector(model_name='facebook/dinov2-base')

# Process PDF with custom parameters
sections = detector.process_pdf(
    pdf_path="your_document.pdf", 
    output_json="sections_output.json",
    dpi=300,
    white_threshold=240,
    min_area=1000,
    eps=0.5,
    enable_clustering=True
)

# Access individual sections
for section in sections:
    print(f"Page {section.page_num}: {section.section_type}")
    print(f"  Position: {section.bbox}")
    print(f"  Color: {section.avg_color}")
```

### Advanced Configuration

```python
# Use larger model for better accuracy (requires more memory)
detector = PDFSectionDetector(model_name='facebook/dinov2-large')

# Process with custom parameters
sections = detector.process_pdf(
    pdf_path="document.pdf",
    dpi=150,                    # Lower DPI for faster processing  
    white_threshold=220,        # More sensitive color detection
    min_area=500,              # Detect smaller regions
    eps=0.3,                   # Tighter clustering
    enable_clustering=True
)
```

### Model Options

- `facebook/dinov2-base`: Balanced performance and speed (default)
- `facebook/dinov2-large`: Higher accuracy, more memory usage
- `facebook/dinov2-giant`: Best accuracy, highest resource requirements

## Output Format

The system outputs a `Section` object for each detected region containing:

```python
@dataclass
class Section:
    page_num: int                           # Page number (1-indexed)
    bbox: Tuple[int, int, int, int]        # Bounding box (x1, y1, x2, y2)
    section_type: str                      # 'section', 'subsection', or 'table'
    embedding: np.ndarray                  # DINOv2 visual embedding
    avg_color: Tuple[int, int, int]        # Average RGB color
```

### JSON Output

When using `output_json` parameter, results are saved in this format:

```json
{
  "sections": [
    {
      "page": 1,
      "bbox": [100, 200, 500, 250],
      "type": "section",
      "color": [45, 85, 125],
      "embedding": [0.123, -0.456, ...]
    }
  ]
}
```

## Algorithm Details

### 1. PDF Processing
- Converts PDF pages to high-resolution images (300 DPI default)
- Maintains original aspect ratios and quality

### 2. Region Detection
- Creates binary masks for non-white pixels (threshold: 240)
- Applies morphological operations (close/open) to clean noise
- Extracts contours and filters by minimum area (1000 pixels)

### 3. Feature Extraction
- Crops regions and processes through DINOv2
- Extracts 768-dimensional embeddings (base model)
- Calculates average RGB color for each region

### 4. Classification
- Uses size, position, and color heuristics
- Aspect ratio analysis (tables: >2.0 ratio)
- Relative width analysis (sections: >80% page width)
- Color intensity thresholding

### 5. Clustering
- Normalizes embeddings using L2 norm
- Applies DBSCAN with cosine distance metric
- Groups visually similar sections across pages

### 6. Visualization
- Draws color-coded bounding boxes around detected sections
- Uses different colors for section types (red=sections, green=subsections, blue=tables)
- Includes cluster-based coloring when similarity clustering is enabled
- Adds legends and labels for easy interpretation

## Performance Notes

- **GPU acceleration**: Automatically uses CUDA if available
- **Memory usage**: ~2-4GB for base model, more for larger models
- **Processing time**: ~1-3 seconds per page (GPU), ~5-10 seconds (CPU)
- **Accuracy**: Best results on documents with colored headers/sections

## Limitations

- Designed for documents with colored sections/headers
- May struggle with purely text-based documents
- Performance depends on PDF quality and color contrast
- Requires sufficient color differentiation from white background

## Troubleshooting

### Common Issues

1. **NumPy Compatibility Error (`_ARRAY_API not found`)**
   ```bash
   # Solution 1: Use stable requirements
   pip install -r requirements-stable.txt
   
   # Solution 2: Downgrade NumPy
   pip install "numpy<2.0.0"
   
   # Solution 3: Fresh environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-stable.txt
   ```

2. **"No module named 'pdf2image'"**
   - Ensure poppler is installed and in PATH
   - Restart terminal after installation

3. **CUDA out of memory**
   - Use smaller model: `facebook/dinov2-base`
   - Reduce DPI: `dpi=150`
   - Process pages individually

4. **Poor detection results**
   - Adjust `white_threshold` parameter
   - Check PDF quality and color contrast
   - Try different DPI settings

5. **Module import errors with transformers**
   - Ensure you're using compatible versions from `requirements-stable.txt`
   - Avoid mixing NumPy 1.x and 2.x in the same environment

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- Facebook AI Research for the DINOv2 model
- HuggingFace Transformers library
- OpenCV community for computer vision tools