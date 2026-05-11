from __future__ import annotations

import os
from dataclasses import replace

import cv2
import numpy as np

from .mask_ops import ensure_odd, overlay_mask_preview, refine_mask
from .mediapipe_backend import (
    _compute_nonuniform_transform,
    _geometry_from_face_mask,
    _morph_mask,
    _warp_template_mask,
    mp,
    mp_tasks,
    mp_vision,
    _MP_IMPORT_ERROR,
    MediaPipeWigRegionDetector,
)
from .types import WigMaskConfig, WigMaskDebug

CLASS_BACKGROUND = 0
CLASS_HAIR = 1
CLASS_BODY_SKIN = 2
CLASS_FACE_SKIN = 3
CLASS_CLOTHES = 4
CLASS_OTHERS = 5

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE_INDICES = [33, 133, 159, 145]
RIGHT_EYE_INDICES = [362, 263, 386, 374]
NOSE_TIP_INDEX = 1
MOUTH_LEFT_INDEX = 61
MOUTH_RIGHT_INDEX = 291
MOUTH_TOP_INDEX = 13
MOUTH_BOTTOM_INDEX = 14


def _resolve_segmenter_model_path(cfg: WigMaskConfig) -> str:
    local_default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "selfie_multiclass_256x256.tflite")
    candidates = [
        cfg.image_segmenter_model_path,
        os.getenv("MP_IMAGE_SEGMENTER_MODEL"),
        local_default,
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError(
        "Image Segmenter model not found. Set image_segmenter_model_path or "
        "MP_IMAGE_SEGMENTER_MODEL to a valid MediaPipe Image Segmenter model."
    )


def _ensure_single_channel_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.uint8)
        if arr.shape[0] == 1:
            return arr[0].astype(np.uint8)
    raise ValueError(f"Expected a single-channel mask, got shape {tuple(arr.shape)}")


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == largest_idx] = 255
    return out


def _centroid(points: np.ndarray) -> np.ndarray:
    return np.mean(points, axis=0).astype(np.float32)


def grey_out_region(image_bgr: np.ndarray, mask: np.ndarray, grey_value: int) -> np.ndarray:
    out = image_bgr.copy()
    gray = np.uint8(np.clip(grey_value, 0, 255))
    out[mask > 0] = (gray, gray, gray)
    return out


def _blend_bgr(image_bgr: np.ndarray, fill_bgr: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    out = image_bgr.astype(np.float32) * (1.0 - alpha) + fill_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_selfie_hair_blur_fill(
    image_bgr: np.ndarray,
    hair_mask: np.ndarray,
    cfg: WigMaskConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    mask = np.where(hair_mask > 0, 255, 0).astype(np.uint8)
    blur_expand_px = max(0, int(getattr(cfg, "selfie_hair_blur_expand_px", 0)))
    if blur_expand_px > 0:
        mask = _morph_mask(mask, dilate_px=blur_expand_px, erode_px=0)
    if not np.any(mask > 0):
        return image_bgr.copy(), {
            "expanded_mask": np.zeros_like(mask),
            "core_mask": np.zeros_like(mask),
            "edge_band_mask": np.zeros_like(mask),
            "edge_alpha_mask": np.zeros_like(mask),
            "soft_blur_ksize": 1,
            "strong_blur_ksize": 1,
        }

    blur_strength = max(0.1, float(getattr(cfg, "selfie_hair_blur_strength", 1.0)))
    soft_blur_ksize = ensure_odd(max(1, int(round(float(cfg.selfie_hair_blur_ksize) * blur_strength))))
    strong_blur_ksize = ensure_odd(
        max(
            soft_blur_ksize,
            int(round(soft_blur_ksize * float(cfg.selfie_hair_core_blur_scale))),
        )
    )

    core_mask = mask.copy()
    core_erode_px = max(0, int(cfg.selfie_hair_core_erode_px))
    if core_erode_px > 0:
        core_mask = _morph_mask(core_mask, dilate_px=0, erode_px=core_erode_px)
        core_mask = cv2.bitwise_and(core_mask, mask)

    edge_band_mask = cv2.subtract(mask, core_mask)
    edge_alpha_mask = edge_band_mask.copy()
    edge_feather_px = max(0, int(cfg.selfie_hair_edge_feather_px))
    if edge_feather_px > 0 and np.any(edge_band_mask > 0):
        feather_ksize = ensure_odd(edge_feather_px * 2 + 1)
        edge_alpha_mask = cv2.GaussianBlur(edge_band_mask, (feather_ksize, feather_ksize), 0)
        edge_alpha_mask[mask == 0] = 0

    soft_blurred = cv2.GaussianBlur(image_bgr, (soft_blur_ksize, soft_blur_ksize), 0)
    strong_blurred = cv2.GaussianBlur(image_bgr, (strong_blur_ksize, strong_blur_ksize), 0)

    out = image_bgr.copy()
    if np.any(edge_alpha_mask > 0):
        out = _blend_bgr(out, soft_blurred, edge_alpha_mask)
    out[core_mask > 0] = strong_blurred[core_mask > 0]
    return out, {
        "expanded_mask": mask,
        "core_mask": core_mask,
        "edge_band_mask": edge_band_mask,
        "edge_alpha_mask": edge_alpha_mask,
        "soft_blur_ksize": soft_blur_ksize,
        "strong_blur_ksize": strong_blur_ksize,
    }


class MediaPipeImageSegmenter:
    def __init__(self, cfg: WigMaskConfig) -> None:
        if mp is None or mp_tasks is None or mp_vision is None:
            raise RuntimeError(
                "mediapipe Tasks API is not available. Install requirements first. "
                f"Import error: {_MP_IMPORT_ERROR}"
            )

        model_path = _resolve_segmenter_model_path(cfg)
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=True,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)

    def segment_foreground_mask(self, image_bgr: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._segmenter.segment(mp_image)

        score = None
        if getattr(result, "confidence_masks", None):
            confidence_masks = [m.numpy_view().astype(np.float32) for m in result.confidence_masks]
            if len(confidence_masks) == 1:
                score = confidence_masks[0]
            elif len(confidence_masks) > 1:
                score = np.max(np.stack(confidence_masks[1:], axis=0), axis=0)

        if score is None and getattr(result, "category_mask", None) is not None:
            category_mask = result.category_mask.numpy_view()
            score = (category_mask > 0).astype(np.float32)

        if score is None:
            raise RuntimeError("Image segmenter returned neither confidence nor category masks.")

        score = _ensure_single_channel_mask(score)
        mask = np.where(score >= float(cfg.segmentation_threshold), 255, 0).astype(np.uint8)
        expand_px = max(0, int(cfg.segmentation_mask_expand_px))
        if expand_px > 0:
            mask = _morph_mask(mask, dilate_px=expand_px, erode_px=0)
        return mask

    def get_category_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._segmenter.segment(mp_image)
        if getattr(result, "category_mask", None) is None:
            raise RuntimeError("Segmenter did not return a category mask.")
        category_mask = result.category_mask.numpy_view()
        category_mask = _ensure_single_channel_mask(category_mask)
        return category_mask.astype(np.uint8)


def _face_detector(cfg: WigMaskConfig) -> MediaPipeWigRegionDetector:
    return MediaPipeWigRegionDetector(replace(cfg, mask_mode="face"))


def detect_face_mask_and_geometry(image_bgr: np.ndarray, cfg: WigMaskConfig) -> tuple[np.ndarray, dict, dict[str, list[int]]]:
    detector = _face_detector(cfg)
    face_mask, _ = detector.detect_mask(image_bgr, replace(cfg, mask_mode="face"))
    geom = _geometry_from_face_mask(face_mask)
    anchors = dict(getattr(detector, "last_anchor_points", {}) or {})
    return face_mask, geom, anchors


def detect_face_landmarks_pixels(image_bgr: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
    detector = _face_detector(cfg)
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector._landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("No face detected.")
    pts = []
    for lm in result.face_landmarks[0]:
        x = int(np.clip(round(lm.x * (w - 1)), 0, w - 1))
        y = int(np.clip(round(lm.y * (h - 1)), 0, h - 1))
        pts.append((x, y))
    return np.asarray(pts, dtype=np.int32)


def build_face_oval_mask_from_landmarks(image_shape: tuple[int, int, int], landmarks_px: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
    h, w = image_shape[:2]
    pts = landmarks_px[FACE_OVAL_INDICES]
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return refine_mask(
        mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )


def build_selfie_multiclass_hair_mask(category_mask: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
    mask = (category_mask == CLASS_HAIR).astype(np.uint8) * 255
    mask = refine_mask(
        mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.hair_blur_ksize),
    )
    return _largest_connected_component(mask)


def build_selfie_multiclass_face_mask(category_mask: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
    mask = (category_mask == CLASS_FACE_SKIN).astype(np.uint8) * 255
    return refine_mask(
        mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )


def get_five_face_points(landmarks_px: np.ndarray) -> np.ndarray:
    left_eye = _centroid(landmarks_px[LEFT_EYE_INDICES])
    right_eye = _centroid(landmarks_px[RIGHT_EYE_INDICES])
    nose_tip = landmarks_px[NOSE_TIP_INDEX].astype(np.float32)
    mouth_left = landmarks_px[MOUTH_LEFT_INDEX].astype(np.float32)
    mouth_right = landmarks_px[MOUTH_RIGHT_INDEX].astype(np.float32)
    return np.vstack([left_eye, right_eye, nose_tip, mouth_left, mouth_right]).astype(np.float32)


def estimate_face_affine(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.estimateAffinePartial2D(src_points, dst_points, method=cv2.LMEDS)
    if matrix is None:
        raise RuntimeError("Failed to estimate affine transform from face landmarks.")
    return matrix


def warp_mask_to_target(mask: np.ndarray, affine_matrix: np.ndarray, target_hw: tuple[int, int], cfg: WigMaskConfig) -> np.ndarray:
    target_h, target_w = target_hw
    warped = cv2.warpAffine(
        mask,
        affine_matrix,
        (target_w, target_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return refine_mask(
        warped,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )


def warp_image_to_target(image_bgr: np.ndarray, affine_matrix: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    return cv2.warpAffine(
        image_bgr,
        affine_matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _expand_mask_outside_face(mask: np.ndarray, face_mask: np.ndarray, expand_px: int) -> np.ndarray:
    if expand_px <= 0:
        return mask.copy()

    expanded = _morph_mask(mask, dilate_px=expand_px, erode_px=0)
    added = cv2.subtract(expanded, mask)
    added = cv2.subtract(added, face_mask)
    return cv2.bitwise_or(mask, added)


def _build_face_overlap_hair_mask(wig_hair_mask: np.ndarray, wig_face_mask: np.ndarray, cfg: WigMaskConfig) -> np.ndarray:
    overlap = cv2.bitwise_and(wig_hair_mask, wig_face_mask)
    expand_px = max(0, int(cfg.wig_face_overlap_expand_px))
    if expand_px > 0:
        overlap = _morph_mask(overlap, dilate_px=expand_px, erode_px=0)
    _, overlap = cv2.threshold(overlap, 1, 255, cv2.THRESH_BINARY)
    return overlap


def _build_face_feature_exclusion_mask(
    image_shape: tuple[int, int],
    landmarks_px: np.ndarray,
    face_bbox: list[int],
    cfg: WigMaskConfig,
) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    pad = max(0, int(cfg.face_island_exclusion_pad_px))

    left_eye = _centroid(landmarks_px[LEFT_EYE_INDICES]).astype(np.int32)
    right_eye = _centroid(landmarks_px[RIGHT_EYE_INDICES]).astype(np.int32)
    nose_tip = landmarks_px[NOSE_TIP_INDEX].astype(np.int32)
    mouth_pts = np.vstack(
        [
            landmarks_px[MOUTH_LEFT_INDEX],
            landmarks_px[MOUTH_RIGHT_INDEX],
            landmarks_px[MOUTH_TOP_INDEX],
            landmarks_px[MOUTH_BOTTOM_INDEX],
        ]
    ).astype(np.int32)
    mouth_center = _centroid(mouth_pts).astype(np.int32)

    eye_radius = max(4, int(round(face_w * 0.07))) + pad
    nose_radius = max(4, int(round(face_w * 0.06))) + pad
    mouth_axes = (
        max(6, int(round(face_w * 0.15))) + pad,
        max(4, int(round(face_h * 0.10))) + pad,
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, tuple(left_eye.tolist()), eye_radius, 255, -1)
    cv2.circle(mask, tuple(right_eye.tolist()), eye_radius, 255, -1)
    cv2.circle(mask, tuple(nose_tip.tolist()), nose_radius, 255, -1)
    cv2.ellipse(mask, tuple(mouth_center.tolist()), mouth_axes, 0, 0, 360, 255, -1)
    return mask


def _build_face_perimeter_mask(face_mask: np.ndarray, thickness_px: int) -> np.ndarray:
    thickness_px = max(1, int(thickness_px))
    inner = _morph_mask(face_mask, dilate_px=0, erode_px=thickness_px)
    return cv2.subtract(face_mask, inner)


def _apply_eye_line_top_trim(
    mask: np.ndarray,
    landmarks_px: np.ndarray,
    trim_px: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    trim_px = max(0, int(trim_px))
    empty = np.zeros_like(mask)
    if trim_px <= 0 or not np.any(mask > 0):
        return mask.copy(), {
            "trimmed_mask": mask.copy(),
            "trim_requirement_map": empty,
            "eye_line_mask": empty,
            "eye_line_y": -1,
        }

    left_eye = _centroid(landmarks_px[LEFT_EYE_INDICES])
    right_eye = _centroid(landmarks_px[RIGHT_EYE_INDICES])
    eye_line_y = int(round((float(left_eye[1]) + float(right_eye[1])) / 2.0))
    eye_line_y = int(np.clip(eye_line_y, 0, mask.shape[0] - 1))

    rows_with_mask = np.where(np.any(mask > 0, axis=1))[0]
    if rows_with_mask.size == 0:
        return mask.copy(), {
            "trimmed_mask": mask.copy(),
            "trim_requirement_map": empty,
            "eye_line_mask": empty,
            "eye_line_y": eye_line_y,
        }

    top_y = int(rows_with_mask[0])
    if top_y >= eye_line_y:
        eye_line_mask = np.zeros_like(mask)
        eye_line_mask[eye_line_y:eye_line_y + 1, :] = 255
        return mask.copy(), {
            "trimmed_mask": mask.copy(),
            "trim_requirement_map": empty,
            "eye_line_mask": eye_line_mask,
            "eye_line_y": eye_line_y,
        }

    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    requirement = np.zeros(mask.shape, dtype=np.float32)
    upper_span = max(1, eye_line_y - top_y)
    for y in range(top_y, eye_line_y):
        t = float(eye_line_y - y) / float(upper_span)
        eased = t * t
        requirement[y, :] = float(trim_px) * eased

    trimmed_mask = np.where((mask > 0) & (distance > requirement), 255, 0).astype(np.uint8)
    eye_line_mask = np.zeros_like(mask)
    eye_line_mask[eye_line_y:eye_line_y + 1, :] = 255
    requirement_u8 = np.clip(np.rint(requirement * (255.0 / max(1, trim_px))), 0, 255).astype(np.uint8)
    return trimmed_mask, {
        "trimmed_mask": trimmed_mask,
        "trim_requirement_map": requirement_u8,
        "eye_line_mask": eye_line_mask,
        "eye_line_y": eye_line_y,
    }


def _connect_floating_face_hair(
    base_mask: np.ndarray,
    floating_hair_mask: np.ndarray,
    selfie_face_mask: np.ndarray,
    landmarks_px: np.ndarray,
    cfg: WigMaskConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    empty = np.zeros_like(base_mask)
    if not bool(cfg.connect_floating_face_hair):
        return empty, {
            "floating_face_hair_mask": floating_hair_mask,
            "floating_face_hair_exclusion_mask": empty,
            "floating_face_hair_bridge_mask": empty,
        }

    allowed_mask = selfie_face_mask.copy()
    support_mask = cv2.bitwise_and(base_mask, allowed_mask)

    num_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(floating_hair_mask, connectivity=8)
    bridge_mask = np.zeros_like(base_mask)
    for label_idx in range(1, num_labels):
        comp = np.zeros_like(base_mask)
        comp[labels == label_idx] = 255
        if not np.any(comp > 0):
            continue

        grown = cv2.bitwise_and(comp, allowed_mask)
        if not np.any(grown > 0):
            continue

        if np.any(support_mask[grown > 0] > 0):
            continue

        connected = False
        max_expand = max(0, int(cfg.floating_face_hair_max_expand_px))
        for _ in range(max_expand):
            expanded = _morph_mask(grown, dilate_px=1, erode_px=0)
            expanded = cv2.bitwise_and(expanded, allowed_mask)
            if np.array_equal(expanded, grown):
                break
            grown = expanded
            if np.any(support_mask[grown > 0] > 0):
                connected = True
                break

        if connected:
            bridge_mask = cv2.bitwise_or(bridge_mask, cv2.subtract(grown, comp))

    _, bridge_mask = cv2.threshold(bridge_mask, 1, 255, cv2.THRESH_BINARY)
    return bridge_mask, {
        "floating_face_hair_mask": floating_hair_mask,
        "floating_face_hair_exclusion_mask": empty,
        "floating_face_hair_bridge_mask": bridge_mask,
    }


def _fill_face_islands(
    current_mask: np.ndarray,
    selfie_face_mask: np.ndarray,
    landmarks_px: np.ndarray,
    cfg: WigMaskConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    empty = np.zeros_like(current_mask)
    if not bool(cfg.fill_face_islands):
        return empty, {
            "face_island_candidate_mask": empty,
            "face_feature_exclusion_mask": empty,
            "face_island_support_mask": empty,
        }

    face_geom = _geometry_from_face_mask(selfie_face_mask)
    candidate_mask = cv2.subtract(selfie_face_mask, current_mask)
    exclusion_mask = _build_face_feature_exclusion_mask(selfie_face_mask.shape, landmarks_px, face_geom["bbox"], cfg)
    face_perimeter_mask = _build_face_perimeter_mask(
        selfie_face_mask,
        max(1, int(cfg.face_island_ring_px // 2)),
    )
    support_mask = cv2.bitwise_or(current_mask, face_perimeter_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    filled_mask = np.zeros_like(current_mask)
    for label_idx in range(1, num_labels):
        comp = np.zeros_like(current_mask)
        comp[labels == label_idx] = 255

        if np.any(exclusion_mask[comp > 0] > 0):
            continue

        ring = cv2.subtract(_morph_mask(comp, dilate_px=int(cfg.face_island_ring_px), erode_px=0), comp)
        ring_count = int(np.count_nonzero(ring))
        if ring_count == 0:
            continue

        support_ratio = float(np.count_nonzero(support_mask[ring > 0] > 0)) / float(ring_count)
        if support_ratio < float(cfg.face_island_min_support_ratio):
            continue

        filled_mask = cv2.bitwise_or(filled_mask, comp)

    _, filled_mask = cv2.threshold(filled_mask, 1, 255, cv2.THRESH_BINARY)
    return filled_mask, {
        "face_island_candidate_mask": candidate_mask,
        "face_feature_exclusion_mask": exclusion_mask,
        "face_island_support_mask": support_mask,
    }


def _neck_block_mask(shape: tuple[int, int], geom: dict, cfg: WigMaskConfig) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = geom["bbox"]
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    cx = int(round((x1 + x2) / 2.0))
    top_y = int(round(y2 - face_h * float(cfg.neck_top_offset_ratio)))
    block_w = max(1, int(round(face_w * float(cfg.neck_block_width_ratio)))) + 2 * int(cfg.hair_neck_block_expand_px)
    block_h = max(1, h - top_y)
    left = max(0, cx - block_w // 2)
    right = min(w, cx + block_w // 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    if top_y < h and right > left:
        mask[top_y:h, left:right] = 255
    return mask


def _seed_region_mask(shape: tuple[int, int], geom: dict, cfg: WigMaskConfig) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = geom["bbox"]
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    pad_x = int(round(face_w * float(cfg.hair_seed_expand_x_ratio)))
    pad_up = int(round(face_h * float(cfg.hair_seed_expand_up_ratio)))
    pad_down = int(round(face_h * float(cfg.hair_seed_expand_down_ratio)))
    sx1 = max(0, x1 - pad_x)
    sx2 = min(w, x2 + pad_x)
    sy1 = max(0, y1 - pad_up)
    sy2 = min(h, y1 + pad_down)

    seed = np.zeros((h, w), dtype=np.uint8)
    if sx2 > sx1 and sy2 > sy1:
        seed[sy1:sy2, sx1:sx2] = 255
    return seed


def _keep_seeded_components(mask: np.ndarray, seed_mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.copy()

    out = np.zeros_like(mask)
    total_area = float(mask.shape[0] * mask.shape[1])
    min_area = max(1, int(total_area * float(min_area_ratio)))
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == label_idx
        if np.any(seed_mask[component] > 0):
            out[component] = 255
    return out


def isolate_hair_mask(image_bgr: np.ndarray, cfg: WigMaskConfig | None = None) -> tuple[np.ndarray, np.ndarray, WigMaskDebug]:
    if cfg is None:
        cfg = WigMaskConfig()
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided.")

    segmenter = MediaPipeImageSegmenter(cfg)
    category_mask = segmenter.get_category_mask(image_bgr)

    hair_mask = build_selfie_multiclass_hair_mask(category_mask, cfg)
    face_mask = build_selfie_multiclass_face_mask(category_mask, cfg)

    face_cutout = face_mask.copy()
    face_dilate_px = max(0, int(cfg.hair_face_dilate_px))
    if face_dilate_px > 0:
        face_cutout = _morph_mask(face_cutout, dilate_px=face_dilate_px, erode_px=0)

    candidate = cv2.subtract(hair_mask, face_cutout)

    post_dilate_px = max(0, int(cfg.hair_post_dilate_px))
    if post_dilate_px > 0:
        candidate = _morph_mask(candidate, dilate_px=post_dilate_px, erode_px=0)

    hair_mask = refine_mask(
        candidate,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.hair_blur_ksize),
    )

    isolated = cv2.bitwise_and(image_bgr, image_bgr, mask=hair_mask)
    debug = WigMaskDebug(
        base_mask=candidate,
        refined_mask=hair_mask,
        points_used=[],
        extra={
            "category_mask": category_mask,
            "face_mask": face_mask,
            "face_cutout": face_cutout,
        },
    )
    return hair_mask, isolated, debug


def remove_selfie_hair(image_bgr: np.ndarray, cfg: WigMaskConfig | None = None, fill_mode: str = "gray") -> tuple[np.ndarray, np.ndarray, WigMaskDebug]:
    if cfg is None:
        cfg = WigMaskConfig()

    hair_mask, _isolated, debug = isolate_hair_mask(image_bgr, cfg)
    out = image_bgr.copy()

    if fill_mode == "blur":
        out, blur_debug = _apply_selfie_hair_blur_fill(image_bgr, hair_mask, cfg)
    elif fill_mode == "black":
        out[hair_mask > 0] = 0
        blur_debug = {}
    else:
        gray = np.uint8(np.clip(cfg.selfie_hair_fill_gray, 0, 255))
        out[hair_mask > 0] = (gray, gray, gray)
        blur_debug = {}

    if blur_debug:
        debug.extra.update(
            {
                "selfie_hair_blur_expanded_mask": blur_debug["expanded_mask"],
                "selfie_hair_blur_core_mask": blur_debug["core_mask"],
                "selfie_hair_blur_edge_band_mask": blur_debug["edge_band_mask"],
                "selfie_hair_blur_edge_alpha_mask": blur_debug["edge_alpha_mask"],
                "selfie_hair_soft_blur_ksize": blur_debug["soft_blur_ksize"],
                "selfie_hair_strong_blur_ksize": blur_debug["strong_blur_ksize"],
            }
        )

    return out, hair_mask, debug


def build_tryon_mask_from_wig_hair(
    wig_hair_mask: np.ndarray,
    wig_face_mask: np.ndarray,
    selfie_bgr: np.ndarray,
    cfg: WigMaskConfig | None = None,
) -> tuple[np.ndarray, WigMaskDebug]:
    if cfg is None:
        cfg = WigMaskConfig()
    if wig_hair_mask is None or wig_face_mask is None or selfie_bgr is None:
        raise ValueError("wig_hair_mask, wig_face_mask, and selfie_bgr are required.")

    wig_hair_mask = np.where(wig_hair_mask > 0, 255, 0).astype(np.uint8)
    wig_face_mask = np.where(wig_face_mask > 0, 255, 0).astype(np.uint8)
    if wig_hair_mask.shape != wig_face_mask.shape:
        raise ValueError("wig_hair_mask and wig_face_mask must have the same size.")

    selfie_face_mask, selfie_geom, anchors = detect_face_mask_and_geometry(selfie_bgr, cfg)
    wig_geom = _geometry_from_face_mask(wig_face_mask)

    template_meta = {
        "face_opening_bbox": wig_geom["bbox"],
        "anchors": wig_geom["anchors"],
    }

    composite = cv2.bitwise_or(wig_hair_mask, wig_face_mask)
    out_h, out_w = selfie_bgr.shape[:2]
    M, transform_info = _compute_nonuniform_transform(template_meta, selfie_geom, cfg)
    warped = _warp_template_mask(composite, out_w, out_h, M)

    face_cutout = selfie_face_mask.copy()
    inset_px = max(0, int(cfg.template_face_cutout_inset_px))
    if inset_px > 0:
        k = inset_px * 2 + 1
        face_cutout = cv2.erode(face_cutout, np.ones((k, k), np.uint8), iterations=1)

    final_mask = cv2.subtract(warped, face_cutout)
    final_mask = refine_mask(
        final_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )

    debug = WigMaskDebug(
        base_mask=warped,
        refined_mask=final_mask,
        points_used=[],
        extra={
            "selfie_face_mask": selfie_face_mask,
            "selfie_face_cutout": face_cutout,
            "anchor_points": anchors,
            "face_opening_bbox": selfie_geom["bbox"],
            "template_transform": transform_info,
        },
    )
    return final_mask, debug


def prepare_wig_selfie_tryon(
    wig_bgr: np.ndarray,
    selfie_bgr: np.ndarray,
    cfg: WigMaskConfig | None = None,
    fill_mode: str = "gray",
) -> dict[str, np.ndarray | WigMaskDebug]:
    if cfg is None:
        cfg = WigMaskConfig()
    if wig_bgr is None or wig_bgr.size == 0:
        raise ValueError("Empty wig image provided.")
    if selfie_bgr is None or selfie_bgr.size == 0:
        raise ValueError("Empty selfie image provided.")

    segmenter = MediaPipeImageSegmenter(cfg)

    wig_landmarks = detect_face_landmarks_pixels(wig_bgr, cfg)
    selfie_landmarks = detect_face_landmarks_pixels(selfie_bgr, cfg)

    wig_cat_mask = segmenter.get_category_mask(wig_bgr)
    selfie_cat_mask = segmenter.get_category_mask(selfie_bgr)

    wig_hair_mask = build_selfie_multiclass_hair_mask(wig_cat_mask, cfg)
    wig_face_mask_model = build_selfie_multiclass_face_mask(wig_cat_mask, cfg)
    wig_face_mask_landmarks = build_face_oval_mask_from_landmarks(wig_bgr.shape, wig_landmarks, cfg)
    wig_face_mask = cv2.bitwise_or(wig_face_mask_model, wig_face_mask_landmarks)
    wig_face_mask = refine_mask(
        wig_face_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )
    wig_face_hair_mask = cv2.bitwise_or(wig_hair_mask, wig_face_mask)
    wig_face_hair_mask = refine_mask(
        wig_face_hair_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )
    wig_transfer_source_mask = _expand_mask_outside_face(
        wig_face_hair_mask,
        wig_face_mask,
        int(cfg.wig_mask_expand_px),
    )
    wig_transfer_source_mask = refine_mask(
        wig_transfer_source_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )
    wig_face_overlap_hair_mask = _build_face_overlap_hair_mask(wig_hair_mask, wig_face_mask, cfg)
    wig_isolated_hair = cv2.bitwise_and(wig_bgr, wig_bgr, mask=wig_transfer_source_mask)

    selfie_hair_mask = build_selfie_multiclass_hair_mask(selfie_cat_mask, cfg)
    selfie_face_mask_model = build_selfie_multiclass_face_mask(selfie_cat_mask, cfg)
    selfie_face_mask_landmarks = build_face_oval_mask_from_landmarks(selfie_bgr.shape, selfie_landmarks, cfg)
    selfie_face_mask = cv2.bitwise_or(selfie_face_mask_model, selfie_face_mask_landmarks)
    selfie_face_mask = refine_mask(
        selfie_face_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )
    if fill_mode == "blur":
        selfie_hair_removed, selfie_blur_debug = _apply_selfie_hair_blur_fill(selfie_bgr, selfie_hair_mask, cfg)
    elif fill_mode == "black":
        selfie_hair_removed = selfie_bgr.copy()
        selfie_hair_removed[selfie_hair_mask > 0] = 0
        selfie_blur_debug = {}
    else:
        selfie_hair_removed = grey_out_region(selfie_bgr, selfie_hair_mask, cfg.selfie_hair_fill_gray)
        selfie_blur_debug = {}

    wig_five = get_five_face_points(wig_landmarks)
    selfie_five = get_five_face_points(selfie_landmarks)
    wig_to_selfie_affine = estimate_face_affine(wig_five, selfie_five)

    warped_wig_hair_mask = warp_mask_to_target(wig_hair_mask, wig_to_selfie_affine, selfie_bgr.shape[:2], cfg)
    warped_wig_face_mask = warp_mask_to_target(wig_face_mask, wig_to_selfie_affine, selfie_bgr.shape[:2], cfg)
    warped_wig_face_hair_mask = warp_mask_to_target(wig_transfer_source_mask, wig_to_selfie_affine, selfie_bgr.shape[:2], cfg)
    warped_wig_face_overlap_hair_mask = warp_mask_to_target(wig_face_overlap_hair_mask, wig_to_selfie_affine, selfie_bgr.shape[:2], cfg)
    warped_wig_image = warp_image_to_target(wig_bgr, wig_to_selfie_affine, selfie_bgr.shape[:2])
    warped_wig_face_hair_mask, top_trim_debug = _apply_eye_line_top_trim(
        warped_wig_face_hair_mask,
        selfie_landmarks,
        int(cfg.wig_top_trim_px),
    )

    selfie_face_cutout = selfie_face_mask.copy()
    inset_px = max(0, int(cfg.template_face_cutout_inset_px))
    if inset_px > 0:
        k = inset_px * 2 + 1
        selfie_face_cutout = cv2.erode(
            selfie_face_cutout,
            np.ones((k, k), np.uint8),
            iterations=1,
        )

    final_mask = cv2.subtract(warped_wig_face_hair_mask, selfie_face_cutout)
    final_mask = refine_mask(
        final_mask,
        close_kernel=max(1, cfg.close_kernel),
        open_kernel=max(1, cfg.open_kernel),
        blur_ksize=max(1, cfg.blur_ksize),
    )
    final_mask = cv2.subtract(final_mask, selfie_face_cutout)
    base_transfer_mask = final_mask.copy()
    preserved_face_hair_mask = cv2.bitwise_and(warped_wig_face_overlap_hair_mask, selfie_face_mask)
    floating_face_hair_bridge_mask, bridge_debug = _connect_floating_face_hair(
        base_transfer_mask,
        preserved_face_hair_mask,
        selfie_face_cutout,
        selfie_landmarks,
        cfg,
    )
    final_mask = cv2.bitwise_or(base_transfer_mask, preserved_face_hair_mask)
    final_mask = cv2.bitwise_or(final_mask, floating_face_hair_bridge_mask)
    filled_face_island_mask, island_debug = _fill_face_islands(final_mask, selfie_face_cutout, selfie_landmarks, cfg)
    final_mask = cv2.bitwise_or(final_mask, filled_face_island_mask)
    _, final_mask = cv2.threshold(final_mask, 1, 255, cv2.THRESH_BINARY)

    final_debug = WigMaskDebug(
        base_mask=warped_wig_face_hair_mask,
        refined_mask=final_mask,
        points_used=[],
        extra={
            "selfie_face_mask": selfie_face_cutout,
            "selfie_face_mask_raw": selfie_face_mask,
            "selfie_face_cutout": selfie_face_cutout,
            "warped_wig_hair_mask": warped_wig_hair_mask,
            "warped_wig_face_mask": warped_wig_face_mask,
            "warped_wig_face_hair_mask": warped_wig_face_hair_mask,
            "wig_face_overlap_hair_mask": wig_face_overlap_hair_mask,
            "warped_wig_face_overlap_hair_mask": warped_wig_face_overlap_hair_mask,
            "preserved_face_hair_mask": preserved_face_hair_mask,
            "floating_face_hair_bridge_mask": floating_face_hair_bridge_mask,
            "filled_face_island_mask": filled_face_island_mask,
            "warped_wig_image": warped_wig_image,
            "category_mask": selfie_cat_mask,
            "affine_matrix": wig_to_selfie_affine,
            **top_trim_debug,
            **bridge_debug,
            **island_debug,
        },
    )

    return {
        "wig_hair_mask": wig_hair_mask,
        "wig_face_mask": wig_face_mask,
        "wig_face_hair_mask": wig_transfer_source_mask,
        "wig_isolated_hair": wig_isolated_hair,
        "selfie_hair_removed": selfie_hair_removed,
        "selfie_hair_mask": selfie_hair_mask,
        "selfie_face_mask": selfie_face_cutout,
        "selfie_face_mask_raw": selfie_face_mask,
        "final_mask": final_mask,
        "warped_wig_hair_mask": warped_wig_hair_mask,
        "warped_wig_face_mask": warped_wig_face_mask,
        "warped_wig_face_hair_mask": warped_wig_face_hair_mask,
        "warped_wig_face_overlap_hair_mask": warped_wig_face_overlap_hair_mask,
        "warped_wig_image": warped_wig_image,
        "wig_category_mask": wig_cat_mask,
        "selfie_category_mask": selfie_cat_mask,
        "wig_debug": WigMaskDebug(
            base_mask=wig_cat_mask,
            refined_mask=wig_transfer_source_mask,
            points_used=[],
            extra={
                "face_mask": wig_face_mask,
                "raw_face_hair_mask": wig_face_hair_mask,
                "face_overlap_hair_mask": wig_face_overlap_hair_mask,
            },
        ),
        "selfie_debug": WigMaskDebug(
            base_mask=selfie_cat_mask,
            refined_mask=selfie_face_cutout,
            points_used=[],
            extra={
                "face_mask": selfie_face_cutout,
                "face_mask_raw": selfie_face_mask,
                **(
                    {
                        "selfie_hair_blur_expanded_mask": selfie_blur_debug["expanded_mask"],
                        "selfie_hair_blur_core_mask": selfie_blur_debug["core_mask"],
                        "selfie_hair_blur_edge_band_mask": selfie_blur_debug["edge_band_mask"],
                        "selfie_hair_blur_edge_alpha_mask": selfie_blur_debug["edge_alpha_mask"],
                        "selfie_hair_soft_blur_ksize": selfie_blur_debug["soft_blur_ksize"],
                        "selfie_hair_strong_blur_ksize": selfie_blur_debug["strong_blur_ksize"],
                    }
                    if selfie_blur_debug
                    else {}
                ),
            },
        ),
        "final_debug": final_debug,
    }


def overlay_dual_mask_preview(image_bgr: np.ndarray, primary_mask: np.ndarray, secondary_mask: np.ndarray | None = None) -> np.ndarray:
    preview = overlay_mask_preview(image_bgr, primary_mask)
    if secondary_mask is not None:
        overlay = preview.copy()
        overlay[secondary_mask > 0] = (80, 255, 120)
        preview = cv2.addWeighted(overlay, 0.30, preview, 0.70, 0)
    return preview
