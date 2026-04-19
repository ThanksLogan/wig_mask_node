from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from .mask_ops import refine_mask
from .mediapipe_backend import MediaPipeWigRegionDetector
from .types import WigMaskConfig, WigMaskDebug


def _resize_for_processing(image_bgr: np.ndarray, max_edge: int) -> Tuple[np.ndarray, float]:
    if max_edge <= 0:
        return image_bgr, 1.0

    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return image_bgr, 1.0

    scale = max_edge / float(longest)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def generate_wig_region_mask(
    image_bgr: np.ndarray,
    cfg: WigMaskConfig | None = None,
) -> tuple[np.ndarray, WigMaskDebug]:
    """
    Main stable API for local scripts and ComfyUI adapters.

    Args:
        image_bgr: OpenCV-style BGR uint8 image
        cfg: Wig mask tuning config

    Returns:
        final_mask: uint8 single-channel mask (0 or 255), same size as input image
        debug: WigMaskDebug with intermediate artifacts
    """
    if cfg is None:
        cfg = WigMaskConfig()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided.")

    proc_img, scale = _resize_for_processing(image_bgr, cfg.max_processing_edge)

    detector = MediaPipeWigRegionDetector(cfg)
    base_mask_proc, points_used_proc = detector.detect_mask(proc_img, cfg)

    refined_proc = refine_mask(
        base_mask_proc,
        close_kernel=cfg.close_kernel,
        open_kernel=cfg.open_kernel,
        blur_ksize=cfg.blur_ksize,
    )

    if scale != 1.0:
        h, w = image_bgr.shape[:2]
        base_mask = cv2.resize(base_mask_proc, (w, h), interpolation=cv2.INTER_NEAREST)
        refined = cv2.resize(refined_proc, (w, h), interpolation=cv2.INTER_NEAREST)
        points_used = [(int(x / scale), int(y / scale)) for x, y in points_used_proc]
    else:
        base_mask = base_mask_proc
        refined = refined_proc
        points_used = points_used_proc

    _, base_mask = cv2.threshold(base_mask, 1, 255, cv2.THRESH_BINARY)
    _, refined = cv2.threshold(refined, 1, 255, cv2.THRESH_BINARY)

    debug = WigMaskDebug(
        base_mask=base_mask,
        refined_mask=refined,
        points_used=points_used,
        extra={
            "anchor_points": getattr(detector, "last_anchor_points", {}),
            "face_opening_bbox": getattr(detector, "last_face_opening_bbox", []),
            "template_transform": getattr(detector, "last_transform_info", {}),
        },
    )

    return refined, debug
