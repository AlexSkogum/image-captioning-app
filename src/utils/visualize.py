import base64
import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_attention(image: Image.Image, attention: np.ndarray, grid_size=(14, 14), alpha=0.6):
    # attention: (num_pixels,) normalized
    att_map = attention.reshape(grid_size)
    att_map = att_map / (att_map.max() + 1e-8)
    att_map = np.clip(att_map, 0, 1)
    att_map = Image.fromarray(np.uint8(att_map * 255)).resize(image.size, resample=Image.BILINEAR).convert('L')
    heat = Image.new('RGBA', image.size)
    plt_heat = np.array(att_map.convert('RGBA'))
    overlay = Image.fromarray(plt_heat)
    blended = Image.blend(image.convert('RGBA'), overlay, alpha=alpha)
    return blended


def to_base64(img: Image.Image, fmt='PNG') -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode('ascii')
