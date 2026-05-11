from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class WigMaskConfig:
    # MediaPipe Tasks Face Landmarker model (.task)
    # If None, loader tries:
    # 1) $MP_FACE_LANDMARKER_MODEL
    # 2) selfie_processing/models/face_landmarker.task
    face_landmarker_model_path: str | None = None
    # MediaPipe Tasks Image Segmenter model (.tflite or compatible asset)
    # Used by the newer hair-isolation workflow.
    image_segmenter_model_path: str | None = None

    # Upper-face selection relative to face height (nose->chin anchor)
    upper_face_ratio: float = 0.45
    # Push points upward to include scalp/wig zone
    forehead_expand_ratio: float = 0.22
    # Expand mask side-to-side from center
    lateral_expand_ratio: float = 0.08
    # Output mask mode: "wig" (upper wig region) or "face" (full face oval)
    mask_mode: str = "wig"

    # Template-doorway wig workflow (default for mode="wig")
    template_mask_path: str | None = None
    template_meta_path: str | None = None
    template_face_width_multiplier: float = 1.0
    template_face_height_multiplier: float = 1.0
    template_post_dilate_px: int = 0
    template_post_erode_px: int = 0
    # Shrinks the face cutout before subtraction so wig region can overlap slightly into face.
    template_face_cutout_inset_px: int = 4

    # Wig canvas shape controls (ported from example_op.py logic)
    top_expand_ratio: float = 0.55
    side_expand_ratio: float = 0.5
    # Absolute pixel floor for side expansion. Final side pad = max(face_w * side_expand_ratio, side_expand_px).
    side_expand_px: int = 0
    down_expand_ratio: float = 1.15
    side_lower_taper_ratio: float = 0.02
    bottom_inset_ratio: float = 0.10
    forehead_inset_ratio: float = 0.08
    forehead_blend_ratio: float = 0.06
    face_cutout_scale_x: float = 1.02
    face_cutout_scale_y: float = 1.00
    neck_block_width_ratio: float = 0.22
    neck_top_offset_ratio: float = 0.02

    # Post processing
    close_kernel: int = 21
    open_kernel: int = 7
    blur_ksize: int = 15

    # Segmentation-assisted hair isolation controls
    segmentation_threshold: float = 0.15
    segmentation_mask_expand_px: int = 0
    hair_face_dilate_px: int = 8
    hair_neck_block_expand_px: int = 4
    hair_seed_expand_x_ratio: float = 0.60
    hair_seed_expand_up_ratio: float = 0.95
    hair_seed_expand_down_ratio: float = 0.18
    hair_component_min_area_ratio: float = 0.0015
    hair_post_dilate_px: int = 2
    hair_blur_ksize: int = 11
    wig_mask_expand_px: int = 16
    wig_face_overlap_expand_px: int = 4
    wig_top_trim_px: int = 0
    connect_floating_face_hair: bool = True
    floating_face_hair_max_expand_px: int = 18
    fill_face_islands: bool = True
    face_island_min_area_px: int = 1
    face_island_max_area_ratio: float = 1.0
    face_island_max_center_y_ratio: float = 1.0
    face_island_exclusion_pad_px: int = 10
    face_island_ring_px: int = 14
    face_island_min_support_ratio: float = 0.12

    # Selfie hair removal output controls
    selfie_hair_fill_gray: int = 160
    selfie_hair_blur_ksize: int = 31
    selfie_hair_blur_expand_px: int = 0
    selfie_hair_blur_strength: float = 1.0
    selfie_hair_core_erode_px: int = 6
    selfie_hair_edge_feather_px: int = 9
    selfie_hair_core_blur_scale: float = 2.0

    # If > 0, resize longest edge before processing for speed/consistency
    max_processing_edge: int = 0


@dataclass
class WigMaskDebug:
    base_mask: np.ndarray
    refined_mask: np.ndarray
    points_used: List[Tuple[int, int]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
