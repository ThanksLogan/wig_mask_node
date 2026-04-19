from __future__ import annotations

import cv2
import numpy as np


def ensure_odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def refine_mask(
    mask: np.ndarray,
    close_kernel: int = 21,
    open_kernel: int = 7,
    blur_ksize: int = 15,
) -> np.ndarray:
    out = mask.copy()

    close_kernel = max(1, close_kernel)
    open_kernel = max(1, open_kernel)
    blur_ksize = ensure_odd(max(1, blur_ksize))

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, close_k)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, open_k)

    if blur_ksize > 1:
        out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)

    _, out = cv2.threshold(out, 1, 255, cv2.THRESH_BINARY)
    return out


def overlay_mask_preview(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = image_bgr.copy()
    overlay = image_bgr.copy()
    overlay[mask > 0] = (40, 140, 255)
    return cv2.addWeighted(overlay, 0.35, color, 0.65, 0)
