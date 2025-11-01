from typing import Tuple
import numpy as np
from PIL import Image

def colorize_mask(mask: np.ndarray) -> Image.Image:
    """
    Turn a HxW int mask into a color image for visualization.
    Labels {0,1,...} map to distinct colors.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be HxW with integer class ids")
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Simple fixed palette for first 21 classes (VOC-like); repeat if needed.
    palette = [
        (0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128),
        (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0),
        (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
        (192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0),
        (0,64,128)
    ]
    for cls_id in np.unique(mask):
        color = palette[int(cls_id) % len(palette)]
        out[mask==cls_id] = color
    return Image.fromarray(out)

def overlay_mask_on_image(image: Image.Image, mask_img: Image.Image, alpha: float=0.5) -> Image.Image:
    """Blend a colorized mask on top of the original image."""
    image = image.convert("RGBA")
    mask_img = mask_img.convert("RGBA")
    return Image.blend(image, mask_img, alpha)
