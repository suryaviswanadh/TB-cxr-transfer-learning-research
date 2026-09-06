from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_uint8_array(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Convert supported image inputs into a uint8 NumPy array."""
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            img = img.convert("RGB")
            return np.asarray(img, dtype=np.uint8)

    if isinstance(image_input, Image.Image):
        rgb = image_input.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)

    if isinstance(image_input, np.ndarray):
        arr = np.asarray(image_input)
        if arr.ndim == 2:
            return arr.astype(np.uint8)

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return arr[:, :, 0].astype(np.uint8)
            if arr.shape[2] == 4:
                rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
                return rgb
            if arr.shape[2] == 3:
                return arr.astype(np.uint8)

        raise ValueError(
            "Unsupported NumPy array shape. Expected grayscale (H, W), RGB (H, W, 3), or RGBA (H, W, 4)."
        )

    raise TypeError("Unsupported image input type. Use a file path, PIL image, or NumPy array.")


def preprocess_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
    """Apply the deterministic inference preprocessing pipeline.

    Pipeline:
    1. grayscale
    2. resize to 224x224
    3. CLAHE (clipLimit=2.0, tileGridSize=(8,8))
    4. convert to 3-channel RGB
    5. tensor
    6. ImageNet normalization
    """
    image_array = _to_uint8_array(image_input)

    if image_array.ndim == 3 and image_array.shape[2] == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    elif image_array.ndim == 2:
        gray = image_array
    else:
        raise ValueError(f"Unsupported image format with shape {image_array.shape}.")

    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(resized)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = (tensor - MEAN) / STD
    return tensor
