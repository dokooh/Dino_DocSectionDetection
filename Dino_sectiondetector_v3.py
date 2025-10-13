import sys
import os

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
    import random
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
    def __init__(self, model_name='facebook/dinov3-base'):
        """
        Initialize the PDF section detector with DINOv3
        
        Args:
            model_name: HuggingFace model name for DINOv3
                       Supported models:
                       - facebook/dinov3-base (default)
                       - facebook/dinov3-large 
                       - facebook/dinov3-giant
                       
        Note: DINOv3 offers improved performance and efficiency over DINOv2
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Validate DINOv3 model name
        valid_dinov3_models = [
            'facebook/dinov3-base',
            'facebook/dinov3-large', 
            'facebook/dinov3-giant'
        ]
        
        if model_name not in valid_dinov3_models:
            print(f"⚠️  Warning: '{model_name}' is not a recognized DINOv3 model.")
            print(f"Valid DINOv3 models: {valid_dinov3_models}")
            print("Falling back to facebook/dinov3-base")
            model_name = 'facebook/dinov3-base'
        
        print(f"Loading DINOv3 model: {model_name}")
        
        try:
            # Load DINOv3 model and processor
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Get embedding dimension for DINOv3 models
            self.embedding_dim = self._get_embedding_dimension(model_name)
            print(f"DINOv3 embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"❌ Error loading DINOv3 model '{model_name}': {e}")
            print("\nThis might be because:")
            print("1. The model is not yet available on HuggingFace")
            print("2. You need to update transformers: pip install --upgrade transformers")
            print("3. Network connectivity issues")
            print("\nFalling back to DINOv2-base as backup...")
            
            # Fallback to DINOv2
            fallback_model = 'facebook/dinov2-base'
            self.processor = AutoImageProcessor.from_pretrained(fallback_model)
            self.model = AutoModel.from_pretrained(fallback_model).to(self.device)
            self.model.eval()
            self.embedding_dim = 768
            print(f"Using fallback model: {fallback_model}")
        
        self.model_name = model_name
    
    def _get_embedding_dimension(self, model_name: str) -> int:
        """Get the embedding dimension for different DINOv3 models"""
        dinov3_dims = {
            'facebook/dinov3-base': 768,
            'facebook/dinov3-large': 1024,
            'facebook/dinov3-giant': 1536
        }
        return dinov3_dims.get(model_name, 768)  # Default to 768 if unknown
    
    def _safe_int_conversion(self, value):
        """Safely convert numpy scalar or other types to Python int"""
        if hasattr(value, 'item'):  # numpy scalar
            return int(value.item())
        elif isinstance(value, (np.integer, np.floating)):
            return int(value)
        else:
            return int(value)
    
    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable by converting numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def _filter_small_bboxes(self, bboxes: List[Tuple[int, int, int, int]], 
                            min_width: int = 50, min_height: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Filter out bounding boxes that are too small in width or height
        
        Args:
            bboxes: List of bounding boxes (x1, y1, x2, y2)
            min_width: Minimum width threshold in pixels
            min_height: Minimum height threshold in pixels
            
        Returns:
            Filtered list of bounding boxes
        """
        filtered_bboxes = []
        removed_count = 0
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Keep bounding boxes that meet minimum size requirements
            if width >= min_width and height >= min_height:
                filtered_bboxes.append(bbox)
            else:
                removed_count += 1
        
        if removed_count > 0:
            print(f"  Removed {removed_count} small bounding boxes (min_width={min_width}, min_height={min_height})")
        
        return filtered_bboxes
    
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
                               min_area: int = 1000,
                               min_width: int = 50,
                               min_height: int = 20) -> List[Tuple[int, int, int, int]]:
        """
        Detect non-white (colored) regions in the image
        
        Args:
            image: PIL Image
            white_threshold: Pixels with all RGB values above this are considered white
            min_area: Minimum area to filter noise
            min_width: Minimum width for bounding boxes in pixels
            min_height: Minimum height for bounding boxes in pixels
            
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
                # Ensure all coordinates are Python integers
                bbox = (int(x), int(y), int(x + w), int(y + h))
                bboxes.append(bbox)
        
        # Post-processing: Remove small bounding boxes based on dimensions
        filtered_bboxes = self._filter_small_bboxes(bboxes, min_width=min_width, min_height=min_height)
        
        return filtered_bboxes
    
    def get_region_embedding(self, image: Image.Image, 
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract DINOv3 embedding for a specific region"""
        try:
            x1, y1, x2, y2 = bbox
            region = image.crop((x1, y1, x2, y2))
            
            # Ensure region is not empty
            if region.size[0] == 0 or region.size[1] == 0:
                raise ValueError(f"Empty region from bbox {bbox}")
            
            # Resize very small regions to ensure good embedding quality
            min_size = 224  # DINOv3 typically works better with larger images
            if region.size[0] < min_size or region.size[1] < min_size:
                # Calculate resize ratio to maintain aspect ratio
                scale = max(min_size / region.size[0], min_size / region.size[1])
                new_size = (int(region.size[0] * scale), int(region.size[1] * scale))
                region = region.resize(new_size, Image.Resampling.LANCZOS)
            
            # Process image with DINOv3 processor
            inputs = self.processor(images=region, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings using DINOv3
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # For DINOv3, we can use the CLS token or average pool the patch embeddings
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    # Use CLS token (first token) - this is the standard approach
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Alternative: use pooler output if available
                    embedding = outputs.pooler_output.cpu().numpy()
                else:
                    # Fallback: try to get some form of hidden state
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
                    else:
                        raise ValueError("Could not extract embedding from DINOv3 model output")
            
            # Ensure embedding is the right shape
            embedding_flat = embedding.flatten()
            if embedding_flat.size == 0:
                raise ValueError("Empty embedding generated")
            
            # Verify expected dimension
            if embedding_flat.size != self.embedding_dim:
                print(f"⚠️  Warning: Expected embedding dim {self.embedding_dim}, got {embedding_flat.size}")
            
            return embedding_flat.astype(np.float32)
            
        except Exception as e:
            print(f"Error getting DINOv3 embedding for bbox {bbox}: {e}")
            # Return a default embedding with the expected size
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
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
        Enhanced for DINOv3 with improved heuristics
        
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
        
        # Enhanced heuristics for classification with DINOv3
        relative_width = width / image_width
        area = width * height
        
        # More sophisticated classification logic
        
        # Large horizontal structures are likely tables
        if aspect_ratio > 3 and area > 20000:
            return 'table'
        
        # Very wide sections spanning most of the page width
        if relative_width > 0.85 and height < 150:
            # Could be main section headers
            if color_intensity < 180:  # Darker colors
                return 'section'
            else:
                return 'subsection'
        
        # Medium-width sections
        elif 0.3 < relative_width <= 0.85:
            if height > 200:
                return 'table'  # Likely content blocks or tables
            elif color_intensity < 150:
                return 'section'
            else:
                return 'subsection'
        
        # Narrow sections
        elif relative_width <= 0.3:
            if height > 300:
                return 'table'  # Likely sidebar or column
            else:
                return 'subsection'  # Small headers or labels
        
        # Fallback based on area and color
        elif area > 50000:
            return 'table'
        elif color_intensity < 200:
            return 'section'
        else:
            return 'subsection'
    
    def process_page(self, image: Image.Image, page_num: int, 
                     white_threshold: int = 240, min_area: int = 1000,
                     min_width: int = 50, min_height: int = 20) -> List[Section]:
        """Process a single page and detect sections using DINOv3"""
        print(f"Processing page {page_num} with DINOv3...")
        
        # Detect colored regions
        bboxes = self.detect_colored_regions(image, white_threshold, min_area, min_width, min_height)
        print(f"  Found {len(bboxes)} colored regions")
        
        sections = []
        for i, bbox in enumerate(bboxes):
            try:
                # Get DINOv3 embedding
                embedding = self.get_region_embedding(image, bbox)
                
                # Get average color
                avg_color = self.get_average_color(image, bbox)
                
                # Classify section type with enhanced logic
                section_type = self.classify_section_type(bbox, avg_color, image.width)
                
                section = Section(
                    page_num=page_num,
                    bbox=bbox,
                    section_type=section_type,
                    embedding=embedding,
                    avg_color=avg_color
                )
                sections.append(section)
                
            except Exception as e:
                print(f"  ⚠️  Error processing region {i} on page {page_num}: {e}")
                continue
        
        print(f"  Successfully processed {len(sections)} sections with DINOv3 embeddings")
        return sections
    
    def cluster_similar_sections(self, sections: List[Section], 
                                eps: float = 0.3) -> List[int]:
        """
        Cluster sections with similar DINOv3 embeddings
        
        Args:
            sections: List of detected sections
            eps: DBSCAN epsilon parameter (adjusted for DINOv3)
            
        Returns:
            List of cluster labels
        """
        if len(sections) == 0:
            return []
        
        print(f"Clustering {len(sections)} sections using DINOv3 embeddings...")
        
        # Stack embeddings
        try:
            embeddings = np.vstack([s.embedding for s in sections])
            print(f"  Embedding matrix shape: {embeddings.shape}")
        except ValueError as e:
            print(f"Error stacking DINOv3 embeddings: {e}")
            print(f"Embedding shapes: {[s.embedding.shape for s in sections[:5]]}")  # Show first 5
            raise
        
        # Normalize embeddings (important for cosine similarity)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        # Cluster using DBSCAN with cosine metric
        # Note: DINOv3 embeddings may cluster differently than DINOv2
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Print clustering results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"  DINOv3 clustering result: {n_clusters} clusters, {n_noise} noise points")
        
        return labels
    
    def draw_sections_on_image(self, image: Image.Image, sections: List[Section], 
                              labels: List[int] = None, output_path: str = None) -> Image.Image:
        """
        Draw bounding boxes and labels on the image for detected sections
        Enhanced visualization for DINOv3 results
        
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
        
        # Define colors for different section types (enhanced palette)
        type_colors = {
            'section': (255, 0, 0),      # Red
            'subsection': (0, 255, 0),   # Green  
            'table': (0, 0, 255)         # Blue
        }
        
        # Generate colors for clusters if labels are provided
        cluster_colors = {}
        if labels is not None and len(labels) > 0:
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:  # Noise points
                    cluster_colors[label] = (128, 128, 128)  # Gray
                else:
                    # Generate random bright colors for clusters
                    seed_value = int(label) if hasattr(label, 'item') else label
                    random.seed(seed_value)  # Consistent colors for same cluster
                    cluster_colors[label] = (
                        random.randint(100, 255),
                        random.randint(100, 255),
                        random.randint(100, 255)
                    )
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)  # Slightly larger font
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw bounding boxes for each section
        for i, section in enumerate(sections):
            x1, y1, x2, y2 = section.bbox
            
            # Choose color based on cluster or section type
            if labels is not None and len(labels) > 0 and i < len(labels):
                color = cluster_colors.get(labels[i], (255, 255, 255))
                label_text = f"C{labels[i]}" if labels[i] != -1 else "N"
            else:
                color = type_colors.get(section.section_type, (255, 255, 255))
                label_text = section.section_type[0].upper()
            
            # Draw bounding box with thicker lines for better visibility
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw section type label with DINOv3 indicator
            dinov3_text = f"D3-{label_text}"
            label_bg_color = tuple(min(255, c + 50) for c in color)
            text_bbox = draw.textbbox((0, 0), dinov3_text, font=font) if font else (0, 0, 25, 18)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position label at top-left of bounding box
            label_x = max(0, x1)
            label_y = max(0, y1 - text_height - 8)
            
            # Draw label background
            draw.rectangle([label_x, label_y, label_x + text_width + 6, label_y + text_height + 4], 
                          fill=label_bg_color, outline=color, width=2)
            
            # Draw label text
            if font:
                draw.text((label_x + 3, label_y + 2), dinov3_text, fill=(0, 0, 0), font=font)
            else:
                draw.text((label_x + 3, label_y + 2), dinov3_text, fill=(0, 0, 0))
            
            # Draw section info (optional, for detailed view)
            info_text = f"{section.section_type}: {section.avg_color}"
            if len(sections) <= 10:  # Only show detailed info for pages with few sections
                info_y = y2 + 5
                if font:
                    draw.text((x1, info_y), info_text, fill=color, font=font)
                else:
                    draw.text((x1, info_y), info_text, fill=color)
        
        # Add enhanced legend with DINOv3 branding
        self._draw_legend(draw, type_colors, font, annotated_image.width, annotated_image.height)
        
        # Add DINOv3 watermark
        watermark_text = f"DINOv3 ({self.model_name})"
        if font:
            watermark_bbox = draw.textbbox((0, 0), watermark_text, font=font)
            watermark_width = watermark_bbox[2] - watermark_bbox[0]
            draw.text((10, 10), watermark_text, fill=(50, 50, 50), font=font)
        
        # Save if output path is provided
        if output_path:
            annotated_image.save(output_path)
            print(f"  Saved DINOv3 annotated image: {output_path}")
        
        return annotated_image
    
    def _draw_legend(self, draw: ImageDraw.Draw, type_colors: dict, font, 
                    image_width: int, image_height: int):
        """Draw a legend showing section type colors with DINOv3 branding"""
        legend_items = list(type_colors.items())
        legend_height = len(legend_items) * 25 + 40  # Extra space for DINOv3 title
        legend_width = 150  # Wider for DINOv3 text
        
        # Position legend at bottom-right
        legend_x = image_width - legend_width - 10
        legend_y = image_height - legend_height - 10
        
        # Draw legend background
        draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
                      fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        # Draw DINOv3 title
        title_y = legend_y + 5
        dinov3_title = "DINOv3 Sections:"
        if font:
            draw.text((legend_x + 5, title_y), dinov3_title, fill=(0, 0, 0), font=font)
        else:
            draw.text((legend_x + 5, title_y), dinov3_title, fill=(0, 0, 0))
        
        # Draw legend items
        for i, (section_type, color) in enumerate(legend_items):
            item_y = title_y + 25 + i * 20
            
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
                    min_area: int = 1000, eps: float = 0.3,
                    enable_clustering: bool = True) -> List[Section]:
        """
        Process entire PDF document using DINOv3
        
        Args:
            pdf_path: Path to PDF file
            output_json: Optional path to save results as JSON
            dpi: DPI for PDF to image conversion
            white_threshold: White pixel threshold for region detection
            min_area: Minimum area for region detection
            eps: DBSCAN epsilon parameter for clustering (adjusted for DINOv3)
            enable_clustering: Whether to perform similarity clustering
            
        Returns:
            List of all detected sections
        """
        print(f"🚀 Starting PDF processing with DINOv3 ({self.model_name})")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path, dpi)
        
        # Process each page
        all_sections = []
        for page_num, image in enumerate(images, start=1):
            sections = self.process_page(image, page_num, white_threshold, min_area)
            all_sections.extend(sections)
        
        print(f"\n✅ Total sections found with DINOv3: {len(all_sections)}")
        
        # Cluster similar sections
        labels = []
        if enable_clustering and len(all_sections) > 0:
            labels = self.cluster_similar_sections(all_sections, eps)
            
            # Print summary
            print("\nDINOv3 Section Summary:")
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
                    'model': self.model_name,
                    'model_type': 'DINOv3',
                    'embedding_dimension': self.embedding_dim,
                    'dpi': int(dpi),
                    'white_threshold': int(white_threshold),
                    'min_area': int(min_area),
                    'clustering_eps': float(eps),
                    'clustering_enabled': enable_clustering,
                    'total_pages': len(images),
                    'total_sections': len(all_sections)
                },
                'sections': [
                    {
                        'page': int(s.page_num),
                        'bbox': [int(coord) for coord in s.bbox],  # Convert all bbox coordinates to int
                        'type': s.section_type,
                        'color': [int(c) for c in s.avg_color],  # Convert all color values to int
                        'embedding': s.embedding.tolist(),
                        'cluster': int(labels[i]) if len(labels) > 0 and i < len(labels) and labels[i] != -1 else -1
                    }
                    for i, s in enumerate(all_sections)
                ]
            }
            with open(output_json, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n💾 DINOv3 results saved to {output_json}")
        
        return all_sections


def parse_arguments():
    """Parse command line arguments for DINOv3"""
    parser = argparse.ArgumentParser(
        description="PDF Section Detection using DINOv3 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Dino_sectiondetector_v3.py document.pdf
  python Dino_sectiondetector_v3.py document.pdf -o results.json
  python Dino_sectiondetector_v3.py document.pdf -m facebook/dinov3-large --dpi 150
  python Dino_sectiondetector_v3.py document.pdf --white-threshold 220 --eps 0.2
  python Dino_sectiondetector_v3.py document.pdf --min-width 100 --min-height 30
  python Dino_sectiondetector_v3.py document.pdf --output-images ./annotated_v3/ --verbose
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
        help='Output JSON file path (default: input_filename_sections_v3.json)'
    )
    
    parser.add_argument(
        '-m', '--model',
        default='facebook/dinov3-base',
        choices=['facebook/dinov3-base', 'facebook/dinov3-large', 'facebook/dinov3-giant'],
        help='DINOv3 model to use (default: facebook/dinov3-base)'
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
        '--min-width',
        type=int,
        default=50,
        help='Minimum width for bounding boxes in pixels (default: 50)'
    )
    
    parser.add_argument(
        '--min-height',
        type=int,
        default=20,
        help='Minimum height for bounding boxes in pixels (default: 20)'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='DBSCAN epsilon parameter for clustering (default: 0.3, optimized for DINOv3)'
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
    
    # Set default output path if not provided (with v3 suffix)
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
        args.output = f"{base_name}_sections_v3.json"
    
    # Validate parameter ranges
    if not 0 <= args.white_threshold <= 255:
        print("Error: white-threshold must be between 0 and 255.")
        sys.exit(1)
    
    if args.dpi < 50 or args.dpi > 600:
        print("Warning: DPI outside recommended range (50-600). This may affect performance.")
    
    if args.min_area < 100:
        print("Warning: Very small min-area may result in noise detection.")
    
    if args.min_width < 10:
        print("Warning: Very small min-width may not filter effectively.")
    
    if args.min_height < 5:
        print("Warning: Very small min-height may not filter effectively.")
    
    # Create save_images directory if specified
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
    
    # Create output_images directory if specified
    if args.output_images:
        os.makedirs(args.output_images, exist_ok=True)
    
    return args


# Enhanced Example usage with command line arguments for DINOv3
if __name__ == "__main__":
    # Parse and validate arguments
    args = parse_arguments()
    args = validate_arguments(args)
    
    # Print configuration
    print("🚀 PDF Section Detection Configuration (DINOv3):")
    print(f"  Input PDF: {args.input_pdf}")
    print(f"  Output JSON: {args.output}")
    print(f"  DINOv3 Model: {args.model}")
    print(f"  DPI: {args.dpi}")
    print(f"  White threshold: {args.white_threshold}")
    print(f"  Min area: {args.min_area}")
    print(f"  Min width: {args.min_width}")
    print(f"  Min height: {args.min_height}")
    print(f"  Clustering eps: {args.eps}")
    print(f"  Skip clustering: {args.no_clustering}")
    if args.save_images:
        print(f"  Save images to: {args.save_images}")
    if args.output_images:
        print(f"  Save annotated images to: {args.output_images}")
    print("=" * 60)
    
    try:
        # Initialize DINOv3 detector
        print(f"🔥 Initializing DINOv3 detector...")
        detector = PDFSectionDetector(model_name=args.model)
        
        # Convert PDF to images
        images = detector.pdf_to_images(args.input_pdf, dpi=args.dpi)
        
        # Process each page
        all_sections = []
        page_sections_map = {}  # Map page numbers to their sections
        
        for page_num, image in enumerate(images, start=1):
            if args.verbose:
                print(f"\n📄 Processing page {page_num}/{len(images)} with DINOv3...")
            
            # Process page with custom parameters
            sections = []
            bboxes = detector.detect_colored_regions(
                image, args.white_threshold, args.min_area, 
                args.min_width, args.min_height
            )
            
            if args.verbose:
                print(f"  Found {len(bboxes)} colored regions")
            
            for bbox_idx, bbox in enumerate(bboxes):
                try:
                    # Filter by minimum area
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    if area < args.min_area:
                        continue
                    
                    if args.verbose:
                        print(f"    Processing bbox {bbox_idx}: {bbox}")
                    
                    # Get DINOv3 embedding
                    embedding = detector.get_region_embedding(image, bbox)
                    if args.verbose:
                        print(f"    Got DINOv3 embedding shape: {embedding.shape}")
                    
                    # Get average color
                    avg_color = detector.get_average_color(image, bbox)
                    if args.verbose:
                        print(f"    Got avg_color: {avg_color}")
                    
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
                    
                except Exception as e:
                    print(f"❌ Error processing bbox {bbox_idx} on page {page_num}: {e}")
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
                image_path = os.path.join(args.save_images, f"page_{page_num:03d}_v3.png")
                image.save(image_path)
                if args.verbose:
                    print(f"  💾 Saved page image: {image_path}")
        
        print(f"\n✅ Total sections found with DINOv3: {len(all_sections)}")
        
        # Cluster similar sections (unless disabled)
        labels = []
        if not args.no_clustering and len(all_sections) > 0:
            print("🔄 Clustering similar sections with DINOv3 embeddings...")
            labels = detector.cluster_similar_sections(all_sections, eps=args.eps)
            
            # Print clustering summary
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            print(f"  🎯 Found {n_clusters} clusters with {n_noise} noise points")
        
        # Create annotated images if requested
        if args.output_images:
            print("\n🎨 Creating DINOv3 annotated images...")
            
            # Create a mapping from section index to label for each page
            section_idx = 0
            for page_num, image in enumerate(images, start=1):
                page_sections = page_sections_map.get(page_num, [])
                
                if page_sections:
                    # Get labels for sections on this page
                    page_labels = []
                    if len(labels) > 0:
                        page_labels = labels[section_idx:section_idx + len(page_sections)]
                    
                    # Create annotated image
                    output_path = os.path.join(args.output_images, f"annotated_page_{page_num:03d}_v3.png")
                    annotated_image = detector.draw_sections_on_image(
                        image, page_sections, page_labels, output_path
                    )
                    
                    if args.verbose:
                        print(f"  🖼️  Created annotated image: {output_path}")
                    
                    section_idx += len(page_sections)
                else:
                    # Create image even if no sections found
                    output_path = os.path.join(args.output_images, f"annotated_page_{page_num:03d}_v3.png")
                    image.save(output_path)
                    
                    if args.verbose:
                        print(f"  📄 Saved page without annotations: {output_path}")
            
            print(f"✅ DINOv3 annotated images saved to {args.output_images}")
        
        # Print detailed summary
        if args.verbose or len(all_sections) <= 20:
            print("\n📊 Detailed DINOv3 Section Summary:")
            for i, section in enumerate(all_sections):
                cluster_id = labels[i] if i < len(labels) else -1
                print(f"  Page {section.page_num}, {section.section_type.upper()}: "
                      f"BBox={section.bbox}, Color={section.avg_color}, "
                      f"Cluster={cluster_id}")
        
        # Save results to JSON
        print(f"\n💾 Saving DINOv3 results to: {args.output}")
        data = {
            'metadata': {
                'input_file': args.input_pdf,
                'model': args.model,
                'model_type': 'DINOv3',
                'embedding_dimension': detector.embedding_dim,
                'dpi': int(args.dpi),
                'white_threshold': int(args.white_threshold),
                'min_area': int(args.min_area),
                'min_width': int(args.min_width),
                'min_height': int(args.min_height),
                'clustering_eps': float(args.eps),
                'clustering_enabled': not args.no_clustering,
                'total_pages': len(images),
                'total_sections': len(all_sections)
            },
            'sections': [
                {
                    'page': int(s.page_num),
                    'bbox': [int(coord) for coord in s.bbox],  # Convert all bbox coordinates to int
                    'type': s.section_type,
                    'color': [int(c) for c in s.avg_color],  # Convert all color values to int
                    'embedding': s.embedding.tolist(),
                    'cluster': int(labels[i]) if len(labels) > 0 and i < len(labels) and labels[i] != -1 else -1
                }
                for i, s in enumerate(all_sections)
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"🎉 DINOv3 processing complete! Results saved to {args.output}")
        
        # Print summary statistics
        section_types = {}
        for section in all_sections:
            section_types[section.section_type] = section_types.get(section.section_type, 0) + 1
        
        print("\n📈 DINOv3 Section Type Summary:")
        for stype, count in section_types.items():
            print(f"  {stype.capitalize()}s: {count}")
        
        print(f"\n🔥 Powered by {detector.model_name} - DINOv3 Technology")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during DINOv3 processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)