pip install torch torchvision
pip install transformers
pip install pdf2image
pip install opencv-python
pip install scikit-learn
pip install Pillow numpy

Key Features:

PDF to Image Conversion: Converts PDF pages to high-resolution images
Color-Based Region Detection: Identifies non-white (colored) regions using OpenCV
DINO Embeddings: Extracts visual embeddings for each colored region
Section Classification: Classifies regions as sections, subsections, or tables based on:

Size and aspect ratio
Position on page
Average color


Similarity Clustering: Groups similar sections across pages using DBSCAN