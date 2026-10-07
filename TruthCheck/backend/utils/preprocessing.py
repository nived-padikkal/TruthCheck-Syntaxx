"""Image preprocessing helpers shared by detectors."""

import numpy as np
from PIL import Image


def load_normalized(path: str, size=(224, 224)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0
