# Image Triangulation
<!-- ![Demo](https://pzpzpzp1.github.io/assets/out2.gif) -->
![Demo](assets/video.gif)


This repo contains tools/algorithms to transform an image into a triangulation using
* reynold's transport thm for continuous optimization
* mesh topology edits for discrete optimization
* saliency map for user specifications

The end result is a mosaic like filtering of the initial image.

See [manuscript](https://drive.google.com/file/d/1ZqjFKV4kbAaghAXSow4ooyxp1CD9hMCe/view?usp=sharing) for algorithmic details. For an overview of available strategies, see `enums.py`

# Environment setup
1. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

For default functionality, run `python batchrun1.py`. 
The default batch should finish in less than 5 minutes total.

Outputs will appear in `/output`
* see `/output/**/video.mp4` to see the process
* see `/output/**/still_B10.png` for final piece


## Basic Usage

1. **Prepare your images**:
place jpg images you want triangulated into `/images`
2. Modify the `WHITELIST` in `batchrun1.py` to include your image name
3. run `python batchrun1.py`

You will be directed to highlight "salient" regions of the image where you want high fidelity.

