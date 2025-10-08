try:
    import torch
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import pdf2image
    from transformers import AutoImageProcessor, AutoModel
    from sklearn.cluster import DBSCAN
    from dataclasses import dataclass
    from typing import List, Tuple
    import cv2
    import argparse
    import json
    import os
    import sys
    import random
    import tempfile
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns
except ImportError as e:
    if "_ARRAY_API not found" in str(e) or "NumPy" in str(e):
        print("❌ NumPy Compatibility Error Detected!")
        print("=" * 50)
        print("This error occurs when NumPy 2.x is installed but some")
        print("dependencies were compiled with NumPy 1.x.")
        print("\nQuick fixes:")
        print("1. pip install -r requirements-stable.txt")
        print("2. pip install 'numpy<2.0.0'")
        print("3. Create a fresh virtual environment")
        print("\nFor detailed instructions, see README.md")
        print("=" * 50)
        sys.exit(1)
    else:
        print(f"❌ Import Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)

@dataclass
class Section:
    """Represents a detected section or subsection"""
    page_num: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    section_type: str  # 'section', 'subsection', 'table'
    embedding: np.ndarray
    avg_color: Tuple[int, int, int]

class PDFSectionDetector:
    def __init__(self, model_name='facebook/dinov2-base'):
        """
        Initialize the PDF section detector
        
        Args:
            model_name: HuggingFace model name (dinov2-base, dinov2-large, dinov2-giant)
                       For DINOv3, use 'facebook/dinov3-base' when available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load DINOv2 model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Create temp directory for embedding visualizations
        self.temp_dir = tempfile.mkdtemp(prefix="dino_embeddings_")
        print(f"Embedding visualizations will be saved to: {self.temp_dir}")
    
    def _safe_int_conversion(self, value):
        """Safely convert numpy scalar or other types to Python int"""
        if hasattr(value, 'item'):  # numpy scalar
            return int(value.item())
        elif isinstance(value, (np.integer, np.floating)):
            return int(value)
        else:
            return int(value)
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """Convert PDF pages to images"""
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        
        # Ensure all images are in RGB mode
        rgb_images = []
        for i, img in enumerate(images):
            if img.mode != 'RGB':
                print(f"Converting page {i+1} from {img.mode} to RGB")
                img = img.convert('RGB')
            rgb_images.append(img)
        
        return rgb_images
    
    def detect_colored_regions(self, image: Image.Image, 
                               white_threshold: int = 240,
                               min_area: int = 1000) -> List[Tuple[int, int, int, int]]:
        """
        Detect non-white (colored) regions in the image
        
        Args:
            image: PIL Image
            white_threshold: Pixels with all RGB values above this are considered white
            min_area: Minimum area to filter noise
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2) for colored regions
        """
        try:
            img_array = np.array(image)
            
            # Debug: Print image info
            # print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
            
            # Ensure we have a 3-channel image (RGB)
            if len(img_array.shape) == 2:
                # Grayscale image - convert to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA image - remove alpha channel
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 3 and img_array.shape[2] != 3:
                # Unexpected number of channels - force to RGB
                if img_array.shape[2] == 1:
                    img_array = np.repeat(img_array, 3, axis=2)
                else:
                    img_array = img_array[:, :, :3]
            
            # Ensure img_array has the right shape
            if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                raise ValueError(f"Unexpected image array shape: {img_array.shape}")
            
            # Create mask for non-white pixels
            # Check if any RGB channel is below threshold
            non_white_mask = np.any(img_array < white_threshold, axis=2).astype(np.uint8) * 255
            
        except Exception as e:
            print(f"Error in detect_colored_regions: {e}")
            print(f"Image mode: {image.mode}, size: {image.size}")
            if 'img_array' in locals():
                print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
            raise
        
        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        non_white_mask = cv2.morphologyEx(non_white_mask, cv2.MORPH_CLOSE, kernel)
        non_white_mask = cv2.morphologyEx(non_white_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(non_white_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        bboxes = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, x + w, y + h))
        
        return bboxes
    
    def get_region_embedding(self, image: Image.Image, 
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract DINO embedding for a specific region"""
        try:
            x1, y1, x2, y2 = bbox
            region = image.crop((x1, y1, x2, y2))
            
            # Ensure region is not empty
            if region.size[0] == 0 or region.size[1] == 0:
                raise ValueError(f"Empty region from bbox {bbox}")
                
            # Process image
            inputs = self.processor(images=region, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Ensure embedding is the right shape
            embedding_flat = embedding.flatten()
            if embedding_flat.size == 0:
                raise ValueError("Empty embedding generated")
                
            return embedding_flat
            
        except Exception as e:
            print(f"Error getting embedding for bbox {bbox}: {e}")
            # Return a default embedding with the expected size (768 for dinov2-base)
            return np.zeros(768, dtype=np.float32)
    
    def get_average_color(self, image: Image.Image, 
                         bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Get average color of a region"""
        x1, y1, x2, y2 = bbox
        region = np.array(image.crop((x1, y1, x2, y2)))
        
        # Handle different image formats
        if len(region.shape) == 2:
            # Grayscale - return as RGB
            avg_color = region.mean()
            return (int(avg_color), int(avg_color), int(avg_color))
        elif len(region.shape) == 3:
            if region.shape[2] == 4:
                # RGBA - use only RGB channels
                region = region[:, :, :3]
            # Calculate average color across spatial dimensions
            avg_color = region.mean(axis=(0, 1))
            
            # Ensure we have a numpy array and convert to int
            if isinstance(avg_color, np.ndarray):
                avg_color = avg_color.astype(int)
                # Ensure we have exactly 3 values
                if avg_color.size >= 3:
                    return tuple(avg_color.flatten()[:3])
                elif avg_color.size == 1:
                    val = int(avg_color.item())
                    return (val, val, val)
                else:
                    # Handle 2 values or other unexpected cases
                    vals = avg_color.flatten()
                    return (int(vals[0]), int(vals[0] if len(vals) == 1 else vals[1]), int(vals[0]))
            else:
                # Handle scalar values
                val = int(avg_color)
                return (val, val, val)
        else:
            # Fallback for unexpected array shapes
            return (128, 128, 128)  # Gray default
    
    def classify_section_type(self, bbox: Tuple[int, int, int, int], 
                             avg_color: Tuple[int, int, int],
                             image_width: int) -> str:
        """
        Classify section type based on size, position, and color
        
        Args:
            bbox: Bounding box
            avg_color: Average RGB color
            image_width: Width of the page image
            
        Returns:
            Section type: 'section', 'subsection', or 'table'
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Color intensity (darker colors might indicate headers)
        # Ensure avg_color is a tuple of ints, not numpy array
        if isinstance(avg_color, (list, tuple)) and len(avg_color) >= 3:
            color_intensity = sum(avg_color[:3]) / 3
        else:
            # Fallback for unexpected formats
            color_intensity = 128  # Neutral gray
        
        # Heuristics for classification
        relative_width = width / image_width
        
        # Tables tend to be wider and have more structure
        if aspect_ratio > 2 and height > 100:
            return 'table'
        # Full-width colored sections are likely main sections
        elif relative_width > 0.8 and height < 100:
            return 'section'
        # Narrower colored regions might be subsections
        elif relative_width < 0.6:
            return 'subsection'
        # Default based on color intensity
        elif color_intensity < 200:
            return 'section'
        else:
            return 'subsection'
    
    def process_page(self, image: Image.Image, page_num: int, 
                     white_threshold: int = 240, min_area: int = 1000) -> List[Section]:
        """Process a single page and detect sections"""
        print(f"Processing page {page_num}...")
        
        # Detect colored regions
        bboxes = self.detect_colored_regions(image, white_threshold, min_area)
        print(f"  Found {len(bboxes)} colored regions")
        
        sections = []
        for bbox in bboxes:
            # Get embedding
            embedding = self.get_region_embedding(image, bbox)
            
            # Get average color
            avg_color = self.get_average_color(image, bbox)
            
            # Classify section type
            section_type = self.classify_section_type(bbox, avg_color, image.width)
            
            section = Section(
                page_num=page_num,
                bbox=bbox,
                section_type=section_type,
                embedding=embedding,
                avg_color=avg_color
            )
            sections.append(section)
        
        return sections
    
    def cluster_similar_sections(self, sections: List[Section], 
                                eps: float = 0.5) -> List[int]:
        """
        Cluster sections with similar embeddings
        
        Args:
            sections: List of detected sections
            eps: DBSCAN epsilon parameter
            
        Returns:
            List of cluster labels
        """
        if len(sections) == 0:
            return []
        
        # Stack embeddings
        try:
            embeddings = np.vstack([s.embedding for s in sections])
        except ValueError as e:
            print(f"Error stacking embeddings: {e}")
            print(f"Embedding shapes: {[s.embedding.shape for s in sections[:5]]}")  # Show first 5
            raise
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        return labels
    
    def draw_sections_on_image(self, image: Image.Image, sections: List[Section], 
                              labels: List[int] = None, output_path: str = None) -> Image.Image:
        """
        Draw bounding boxes and labels on the image for detected sections
        
        Args:
            image: PIL Image to draw on
            sections: List of detected sections for this page
            labels: Optional cluster labels for color coding
            output_path: Optional path to save the annotated image
            
        Returns:
            PIL Image with drawn bounding boxes
        """
        # Create a copy of the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Define colors for different section types
        type_colors = {
            'section': (255, 0, 0),      # Red
            'subsection': (0, 255, 0),   # Green
            'table': (0, 0, 255)         # Blue
        }
        
        # Generate colors for clusters if labels are provided
        cluster_colors = {}
        if labels is not None and len(labels) > 0:  # Fix: Proper array check
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Noise points
                    cluster_colors[label] = (128, 128, 128)  # Gray
                else:
                    # Generate random bright colors for clusters
                    # Convert numpy scalar to Python int for random.seed()
                    seed_value = int(label) if hasattr(label, 'item') else label
                    random.seed(seed_value)  # Consistent colors for same cluster
                    cluster_colors[label] = (
                        random.randint(100, 255),
                        random.randint(100, 255),
                        random.randint(100, 255)
                    )
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw bounding boxes for each section
        for i, section in enumerate(sections):
            x1, y1, x2, y2 = section.bbox
            
            # Choose color based on cluster or section type
            if labels is not None and len(labels) > 0 and i < len(labels):  # Fix: Proper array check
                color = cluster_colors.get(labels[i], (255, 255, 255))
                label_text = f"C{labels[i]}" if labels[i] != -1 else "N"
            else:
                color = type_colors.get(section.section_type, (255, 255, 255))
                label_text = section.section_type[0].upper()
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw section type label
            label_bg_color = tuple(min(255, c + 50) for c in color)
            text_bbox = draw.textbbox((0, 0), label_text, font=font) if font else (0, 0, 20, 15)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position label at top-left of bounding box
            label_x = max(0, x1)
            label_y = max(0, y1 - text_height - 5)
            
            # Draw label background
            draw.rectangle([label_x, label_y, label_x + text_width + 4, label_y + text_height + 2], 
                          fill=label_bg_color, outline=color)
            
            # Draw label text
            if font:
                draw.text((label_x + 2, label_y + 1), label_text, fill=(0, 0, 0), font=font)
            else:
                draw.text((label_x + 2, label_y + 1), label_text, fill=(0, 0, 0))
            
            # Draw section info (optional, for detailed view)
            info_text = f"{section.section_type}: {section.avg_color}"
            if len(sections) <= 10:  # Only show detailed info for pages with few sections
                info_y = y2 + 5
                if font:
                    draw.text((x1, info_y), info_text, fill=color, font=font)
                else:
                    draw.text((x1, info_y), info_text, fill=color)
        
        # Add legend
        self._draw_legend(draw, type_colors, font, annotated_image.width, annotated_image.height)
        
        # Save if output path is provided
        if output_path:
            annotated_image.save(output_path)
        
        return annotated_image
    
    def _draw_legend(self, draw: ImageDraw.Draw, type_colors: dict, font, 
                    image_width: int, image_height: int):
        """Draw a legend showing section type colors"""
        legend_items = list(type_colors.items())
        legend_height = len(legend_items) * 25 + 20
        legend_width = 120
        
        # Position legend at bottom-right
        legend_x = image_width - legend_width - 10
        legend_y = image_height - legend_height - 10
        
        # Draw legend background
        draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                      fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Draw legend title
        title_y = legend_y + 5
        if font:
            draw.text((legend_x + 5, title_y), "Section Types:", fill=(0, 0, 0), font=font)
        else:
            draw.text((legend_x + 5, title_y), "Section Types:", fill=(0, 0, 0))
        
        # Draw legend items
        for i, (section_type, color) in enumerate(legend_items):
            item_y = title_y + 20 + i * 20
            
            # Draw color box
            draw.rectangle([legend_x + 5, item_y, legend_x + 15, item_y + 10],
                          fill=color, outline=(0, 0, 0))
            
            # Draw text
            text = section_type.capitalize()
            if font:
                draw.text((legend_x + 20, item_y - 2), text, fill=(0, 0, 0), font=font)
            else:
                draw.text((legend_x + 20, item_y - 2), text, fill=(0, 0, 0))
    
    def process_pdf(self, pdf_path: str, output_json: str = None, 
                    dpi: int = 300, white_threshold: int = 240, 
                    min_area: int = 1000, eps: float = 0.5,
                    enable_clustering: bool = True) -> List[Section]:
        """
        Process entire PDF document
        
        Args:
            pdf_path: Path to PDF file
            output_json: Optional path to save results as JSON
            dpi: DPI for PDF to image conversion
            white_threshold: White pixel threshold for region detection
            min_area: Minimum area for region detection
            eps: DBSCAN epsilon parameter for clustering
            enable_clustering: Whether to perform similarity clustering
            
        Returns:
            List of all detected sections
        """
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi)
        
        # Process each page
        all_sections = []
        for page_num, image in enumerate(images, start=1):
            sections = self.process_page(image, page_num, white_threshold, min_area)
            all_sections.extend(sections)
        
        print(f"\nTotal sections found: {len(all_sections)}")
        
        # Cluster similar sections
        labels = []
        if enable_clustering and len(all_sections) > 0:
            labels = self.cluster_similar_sections(all_sections, eps)
            
            # Print summary
            print("\nSection Summary:")
            for i, section in enumerate(all_sections):
                cluster_id = labels[i] if i < len(labels) else -1
                print(f"Page {section.page_num}, {section.section_type.upper()}: "
                      f"BBox={section.bbox}, Color={section.avg_color}, "
                      f"Cluster={cluster_id}")
        
        # Optionally save to JSON
        if output_json:
            data = {
                'metadata': {
                    'input_file': pdf_path,
                    'dpi': dpi,
                    'white_threshold': white_threshold,
                    'min_area': min_area,
                    'clustering_eps': eps,
                    'clustering_enabled': enable_clustering,
                    'total_pages': len(images),
                    'total_sections': len(all_sections)
                },
                'sections': [
                    {
                        'page': s.page_num,
                        'bbox': s.bbox,
                        'type': s.section_type,
                        'color': s.avg_color,
                        'embedding': s.embedding.tolist(),
                        'cluster': labels[i] if len(labels) > 0 and i < len(labels) else -1  # Fix: Proper array check
                    }
                    for i, s in enumerate(all_sections)
                ]
            }
            with open(output_json, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nResults saved to {output_json}")
        
        return all_sections


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="PDF Section Detection using DINO embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Dino_sectiondetector.py document.pdf
  python Dino_sectiondetector.py document.pdf -o results.json
  python Dino_sectiondetector.py document.pdf -m facebook/dinov2-large --dpi 150
  python Dino_sectiondetector.py document.pdf --white-threshold 220 --eps 0.3
  python Dino_sectiondetector.py document.pdf --output-images ./annotated/ --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_pdf',
        help='Path to input PDF file'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output JSON file path (default: input_filename_sections.json)'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='facebook/dinov2-base',
        choices=['facebook/dinov2-base', 'facebook/dinov2-large', 'facebook/dinov2-giant'],
        help='DINO model to use (default: facebook/dinov2-base)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF to image conversion (default: 300)'
    )
    
    parser.add_argument(
        '--white-threshold',
        type=int,
        default=240,
        help='White pixel threshold (0-255, default: 240)'
    )
    
    parser.add_argument(
        '--min-area',
        type=int,
        default=1000,
        help='Minimum area for region detection (default: 1000)'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.5,
        help='DBSCAN epsilon parameter for clustering (default: 0.5)'
    )
    
    parser.add_argument(
        '--no-clustering',
        action='store_true',
        help='Skip similarity clustering step'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--save-images',
        help='Directory to save processed page images (optional)'
    )
    
    parser.add_argument(
        '--output-images',
        help='Directory to save annotated images with bounding boxes (optional)'
    )
    
    parser.add_argument(
        '--save-embedding-viz',
        action='store_true',
        help='Save embedding visualization images to temp folder'
    )
    
    parser.add_argument(
        '--temp-dir',
        help='Custom temporary directory for visualizations (default: auto-generated)'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments"""
    # Check if input PDF exists
    if not os.path.exists(args.input_pdf):
        print(f"Error: Input PDF file '{args.input_pdf}' not found.")
        sys.exit(1)
    
    # Check if input is a PDF file
    if not args.input_pdf.lower().endswith('.pdf'):
        print(f"Error: Input file '{args.input_pdf}' is not a PDF file.")
        sys.exit(1)
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
        args.output = f"{base_name}_sections.json"
    
    # Validate parameter ranges
    if not 0 <= args.white_threshold <= 255:
        print("Error: white-threshold must be between 0 and 255.")
        sys.exit(1)
    
    if args.dpi < 50 or args.dpi > 600:
        print("Warning: DPI outside recommended range (50-600). This may affect performance.")
    
    if args.min_area < 100:
        print("Warning: Very small min-area may result in noise detection.")
    
    # Create save_images directory if specified
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
    
    # Create output_images directory if specified
    if args.output_images:
        os.makedirs(args.output_images, exist_ok=True)
    
    return args


# Enhanced Example usage with command line arguments
if __name__ == "__main__":
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    
    # Print configuration
    print("PDF Section Detection Configuration:")
    print(f"  Input PDF: {args.input_pdf}")
    print(f"  Output JSON: {args.output}")
    print(f"  Model: {args.model}")
    print(f"  DPI: {args.dpi}")
    print(f"  White threshold: {args.white_threshold}")
    print(f"  Min area: {args.min_area}")
    print(f"  Clustering eps: {args.eps}")
    print(f"  Skip clustering: {args.no_clustering}")
    if args.save_images:
        print(f"  Save images to: {args.save_images}")
    if args.output_images:
        print(f"  Save annotated images to: {args.output_images}")
    print("-" * 50)
    
    try:
        # Initialize detector
        print(f"Loading model: {args.model}...")
        detector = PDFSectionDetector(model_name=args.model)
        
        # Convert PDF to images
        images = detector.pdf_to_images(args.input_pdf, dpi=args.dpi)
        
        # Process each page
        all_sections = []
        page_sections_map = {}  # Map page numbers to their sections
        
        for page_num, image in enumerate(images, start=1):
            if args.verbose:
                print(f"\nProcessing page {page_num}/{len(images)}...")
            
            # Override detection parameters
            original_detect = detector.detect_colored_regions
            def custom_detect(img, white_threshold=args.white_threshold):
                return original_detect(img, white_threshold)
            
            # Process page with custom parameters and visualization
            sections = []
            bboxes = custom_detect(image)
            
            if args.verbose:
                print(f"  Found {len(bboxes)} colored regions")
            
            # Save color detection visualization if requested
            if args.save_embedding_viz:
                color_viz_path = detector.visualize_color_detection(
                    image, bboxes, page_num, args.white_threshold
                )
                if args.verbose:
                    print(f"  Color detection visualization: {color_viz_path}")
            
            for bbox_idx, bbox in enumerate(bboxes):
                try:
                    # Filter by minimum area
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    if area < args.min_area:
                        continue
                    
                    if args.verbose:
                        print(f"    Processing bbox {bbox_idx}: {bbox}")
                    
                    # Get embedding
                    embedding = detector.get_region_embedding(image, bbox)
                    if args.verbose:
                        print(f"    Got embedding shape: {embedding.shape}")
                    
                    # Get average color
                    avg_color = detector.get_average_color(image, bbox)
                    if args.verbose:
                        print(f"    Got avg_color: {avg_color} (type: {type(avg_color)})")
                    
                    # Classify section type
                    section_type = detector.classify_section_type(bbox, avg_color, image.width)
                    if args.verbose:
                        print(f"    Classified as: {section_type}")
                    
                    section = Section(
                        page_num=page_num,
                        bbox=bbox,
                        section_type=section_type,
                        embedding=embedding,
                        avg_color=avg_color
                    )
                    sections.append(section)
                    
                    # Save embedding visualization if requested
                    if args.save_embedding_viz:
                        embedding_viz_path = detector.visualize_embedding_processing(
                            image, bbox, embedding, page_num, bbox_idx
                        )
                        if args.verbose:
                            print(f"    Embedding visualization: {embedding_viz_path}")
                    
                except Exception as e:
                    print(f"Error processing bbox {bbox_idx} on page {page_num}: {e}")
                    print(f"  BBox: {bbox}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # Store sections for this page
            page_sections_map[page_num] = sections
            all_sections.extend(sections)
            
            # Save page image if requested
            if args.save_images:
                image_path = os.path.join(args.save_images, f"page_{page_num:03d}.png")
                image.save(image_path)
                if args.verbose:
                    print(f"  Saved page image: {image_path}")
        
        print(f"\nTotal sections found: {len(all_sections)}")
        
        # Cluster similar sections (unless disabled)
        labels = []
        if not args.no_clustering and len(all_sections) > 0:
            print("Clustering similar sections...")
            labels = detector.cluster_similar_sections(all_sections, eps=args.eps)
            
            # Print clustering summary
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"  Found {n_clusters} clusters with {n_noise} noise points")
            
            # Save clustering visualization if requested
            if args.save_embedding_viz:
                clustering_viz_path = detector.visualize_clustering_results(all_sections, labels)
                if clustering_viz_path:
                    print(f"  Clustering visualization saved: {clustering_viz_path}")
        
        # Create annotated images if requested
        if args.output_images:
            print("\nCreating annotated images...")
            
            # Create a mapping from section index to label for each page
            section_idx = 0
            for page_num, image in enumerate(images, start=1):
                page_sections = page_sections_map.get(page_num, [])
                
                if page_sections:
                    # Get labels for sections on this page
                    page_labels = []
                    if len(labels) > 0:  # Fix: Use len() instead of direct truthiness check
                        page_labels = labels[section_idx:section_idx + len(page_sections)]
                    
                    # Create annotated image
                    output_path = os.path.join(args.output_images, f"annotated_page_{page_num:03d}.png")
                    annotated_image = detector.draw_sections_on_image(
                        image, page_sections, page_labels, output_path
                    )
                    
                    if args.verbose:
                        print(f"  Created annotated image: {output_path}")
                    
                    section_idx += len(page_sections)
                else:
                    # Create image even if no sections found
                    output_path = os.path.join(args.output_images, f"annotated_page_{page_num:03d}.png")
                    image.save(output_path)
                    
                    if args.verbose:
                        print(f"  Saved page without annotations: {output_path}")
            
            print(f"✓ Annotated images saved to {args.output_images}")
        
        # Print detailed summary
        if args.verbose or len(all_sections) <= 20:
            print("\nDetailed Section Summary:")
            for i, section in enumerate(all_sections):
                cluster_id = labels[i] if i < len(labels) else -1
                print(f"  Page {section.page_num}, {section.section_type.upper()}: "
                      f"BBox={section.bbox}, Color={section.avg_color}, "
                      f"Cluster={cluster_id}")
        
        # Save results to JSON
        print(f"\nSaving results to: {args.output}")
        data = {
            'metadata': {
                'input_file': args.input_pdf,
                'model': args.model,
                'dpi': args.dpi,
                'white_threshold': args.white_threshold,
                'min_area': args.min_area,
                'clustering_eps': args.eps,
                'clustering_enabled': not args.no_clustering,
                'total_pages': len(images),
                'total_sections': len(all_sections)
            },
            'sections': [
                {
                    'page': s.page_num,
                    'bbox': s.bbox,
                    'type': s.section_type,
                    'color': s.avg_color,
                    'embedding': s.embedding.tolist(),
                    'cluster': labels[i] if len(labels) > 0 and i < len(labels) else -1  # Fix: Proper array check
                }
                for i, s in enumerate(all_sections)
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Processing complete! Results saved to {args.output}")
        
        # Print summary statistics
        section_types = {}
        for section in all_sections:
            section_types[section.section_type] = section_types.get(section.section_type, 0) + 1
        
        print("\nSection Type Summary:")
        for stype, count in section_types.items():
            print(f"  {stype.capitalize()}s: {count}")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    def visualize_color_detection(self, image: Image.Image, bboxes: List[Tuple[int, int, int, int]], 
                                 page_num: int, white_threshold: int, save_dir: str = None) -> str:
        """
        Visualize the color detection process
        
        Args:
            image: Original page image
            bboxes: Detected bounding boxes
            page_num: Page number
            white_threshold: White threshold used
            save_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization
        """
        if save_dir is None:
            save_dir = self.temp_dir
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Color Detection Process - Page {page_num}', fontsize=16)
        
        # 1. Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Page')
        axes[0, 0].axis('off')
        
        # 2. Color detection mask
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Create mask for non-white pixels
            non_white_mask = np.any(img_array[:, :, :3] < white_threshold, axis=2)
            axes[0, 1].imshow(non_white_mask, cmap='gray')
            axes[0, 1].set_title(f'Non-White Mask\n(threshold={white_threshold})')
        else:
            axes[0, 1].text(0.5, 0.5, 'Cannot create mask\nfor this image format', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Mask Creation Failed')
        axes[0, 1].axis('off')
        
        # 3. Detected regions with bounding boxes
        axes[1, 0].imshow(image)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                           edgecolor=plt.cm.tab10(i % 10), facecolor='none')
            axes[1, 0].add_patch(rect)
            # Add region number
            axes[1, 0].text(x1+5, y1+15, str(i), color='white', fontweight='bold', 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=plt.cm.tab10(i % 10)))
        axes[1, 0].set_title(f'Detected Regions ({len(bboxes)} found)')
        axes[1, 0].axis('off')
        
        # 4. Region size distribution
        if bboxes:
            areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in bboxes]
            axes[1, 1].hist(areas, bins=min(10, len(areas)), alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Region Area Distribution')
            axes[1, 1].set_xlabel('Area (pixels²)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"""Statistics:
Total regions: {len(bboxes)}
Min area: {min(areas):,} px²
Max area: {max(areas):,} px²
Mean area: {np.mean(areas):,.0f} px²
Median area: {np.median(areas):,.0f} px²"""
            axes[1, 1].text(0.98, 0.98, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, 'No regions detected', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('No Data')
        
        # Save the visualization
        output_path = os.path.join(save_dir, f'color_detection_p{page_num:03d}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path