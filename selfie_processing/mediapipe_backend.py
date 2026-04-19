from __future__ import annotations

import json
import os
from typing import List, Tuple

import cv2
import numpy as np

from .types import WigMaskConfig

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
except Exception as exc:  # pragma: no cover
    mp = None
    mp_tasks = None
    mp_vision = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None


# Face oval indices from MediaPipe Face Mesh topology.
# We filter these to upper-face points, then expand to a wig region.
FACE_OVAL_IDX = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]

NOSE_TIP_IDX = 1
CHIN_IDX = 152
FOREHEAD_CENTER_IDX = 10
LEFT_TEMPLE_IDX = 234
RIGHT_TEMPLE_IDX = 454


def _default_model_path() -> str:
    # ../.. from src/selfie_processing/mediapipe_backend.py -> selfie_processing/
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "models", "face_landmarker.task")


def _resolve_model_path(cfg: WigMaskConfig) -> str:
    candidates = [
        cfg.face_landmarker_model_path,
        os.getenv("MP_FACE_LANDMARKER_MODEL"),
        _default_model_path(),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise RuntimeError(
        "Face Landmarker model not found. Download 'face_landmarker.task' and place it at "
        "selfie_processing/models/face_landmarker.task, or set --model-path / "
        "MP_FACE_LANDMARKER_MODEL."
    )


def _processing_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_template_mask_path() -> str:
    root = _processing_root()
    jpg = os.path.join(root, "images", "template_mask.jpg")
    png = os.path.join(root, "images", "template_mask.png")
    return jpg if os.path.exists(jpg) else png


def _default_template_meta_path() -> str:
    return os.path.join(_processing_root(), "images", "template_mask_meta.json")


def _resolve_template_mask_path(cfg: WigMaskConfig) -> str:
    path = cfg.template_mask_path or _default_template_mask_path()
    if not os.path.exists(path):
        raise RuntimeError(
            f"Template mask not found: {path}. "
            "Expected images/template_mask.jpg (or template_mask.png) by default."
        )
    return path


def _resolve_template_meta_path(cfg: WigMaskConfig) -> str:
    path = cfg.template_meta_path or _default_template_meta_path()
    if not os.path.exists(path):
        raise RuntimeError(f"Template metadata not found: {path}")
    return path


def _load_binary_mask(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load template mask image: {path}")
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _contour_bbox(contour) -> list[int]:
    x, y, w, h = cv2.boundingRect(contour)
    return [int(x), int(y), int(x + w), int(y + h)]


def _validate_template_meta(meta: dict) -> None:
    if "face_opening_bbox" not in meta:
        raise ValueError("template_mask_meta.json missing 'face_opening_bbox'")
    if "anchors" not in meta:
        raise ValueError("template_mask_meta.json missing 'anchors'")
    for key in ["nose_center", "left_temple", "right_temple", "chin_bottom", "forehead_center"]:
        if key not in meta["anchors"]:
            raise ValueError(f"template_mask_meta.json missing anchors.{key}")


def _geometry_from_face_mask(mask: np.ndarray) -> dict:
    contour = _largest_contour(mask)
    if contour is None:
        raise RuntimeError("No face contour found in selfie face mask.")

    x1, y1, x2, y2 = _contour_bbox(contour)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0

    anchors = {
        "forehead_center": [cx, y1],
        "nose_center": [cx, y1 + h * 0.55],
        "chin_bottom": [cx, y2],
        "left_temple": [x1, y1 + h * 0.28],
        "right_temple": [x2, y1 + h * 0.28],
    }
    return {"bbox": [x1, y1, x2, y2], "anchors": anchors}


def _compute_nonuniform_transform(template_meta: dict, selfie_geom: dict, cfg: WigMaskConfig) -> tuple[np.ndarray, dict]:
    tx1, ty1, tx2, ty2 = template_meta["face_opening_bbox"]
    sx1, sy1, sx2, sy2 = selfie_geom["bbox"]

    template_face_w = tx2 - tx1
    template_face_h = ty2 - ty1
    selfie_face_w = sx2 - sx1
    selfie_face_h = sy2 - sy1
    if template_face_w <= 0 or template_face_h <= 0:
        raise ValueError("Invalid template face_opening_bbox dimensions.")

    scale_x = (selfie_face_w / template_face_w) * float(cfg.template_face_width_multiplier)
    scale_y = (selfie_face_h / template_face_h) * float(cfg.template_face_height_multiplier)

    tx_anchor_x, tx_anchor_y = template_meta["anchors"]["nose_center"]
    sx_anchor_x, sx_anchor_y = selfie_geom["anchors"]["nose_center"]

    translate_x = sx_anchor_x - (scale_x * tx_anchor_x)
    translate_y = sx_anchor_y - (scale_y * tx_anchor_y)

    M = np.array([[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y]], dtype=np.float32)
    info = {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "translate_x": float(translate_x),
        "translate_y": float(translate_y),
    }
    return M, info


def _warp_template_mask(template_mask: np.ndarray, output_w: int, output_h: int, M: np.ndarray) -> np.ndarray:
    warped = cv2.warpAffine(
        template_mask,
        M,
        (output_w, output_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    return warped


def _morph_mask(mask: np.ndarray, dilate_px: int = 0, erode_px: int = 0) -> np.ndarray:
    out = mask.copy()
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8), iterations=1)
    if erode_px > 0:
        k = erode_px * 2 + 1
        out = cv2.erode(out, np.ones((k, k), np.uint8), iterations=1)
    return out


def _order_polygon_clockwise(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _sample_top_arc(face_pts: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(face_pts.astype(np.float32)).reshape(-1, 2)
    hull = _order_polygon_clockwise(hull)

    y_cut = np.percentile(face_pts[:, 1], 42)
    top_pts = hull[hull[:, 1] <= y_cut]
    if len(top_pts) < 6:
        top_pts = hull[hull[:, 1] <= np.mean(face_pts[:, 1])]

    top_pts = top_pts[np.argsort(top_pts[:, 0])]

    dedup = []
    for p in top_pts:
        if not dedup or abs(float(p[0]) - float(dedup[-1][0])) > 2:
            dedup.append(p)
    top_pts = np.array(dedup, dtype=np.float32)

    if len(top_pts) < 4:
        min_xy = face_pts.min(axis=0)
        max_xy = face_pts.max(axis=0)
        cx = (min_xy[0] + max_xy[0]) / 2.0
        top_y = min_xy[1]
        top_pts = np.array(
            [
                [min_xy[0], top_y],
                [cx - (max_xy[0] - min_xy[0]) * 0.2, top_y - 10],
                [cx + (max_xy[0] - min_xy[0]) * 0.2, top_y - 10],
                [max_xy[0], top_y],
            ],
            dtype=np.float32,
        )

    return top_pts


def _resample_polyline(points: np.ndarray, num_samples: int = 30) -> np.ndarray:
    if len(points) < 2:
        return points.copy()

    seg_lens = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(cum[-1])
    if total == 0.0:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total, num_samples)
    out = []
    j = 0
    for t in targets:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        t0, t1 = float(cum[j]), float(cum[j + 1])
        p0, p1 = points[j], points[j + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        out.append(p0 * (1.0 - alpha) + p1 * alpha)
    return np.array(out, dtype=np.float32)


def _smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _bridge_points(p0: np.ndarray, p1: np.ndarray, n: int = 5) -> np.ndarray:
    """
    Build eased transition points between two anchors so polygon joins are not abrupt.
    """
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = []
    for i in range(1, n + 1):
        t = i / float(n + 1)
        e = _smoothstep(t)
        pts.append(p0 * (1.0 - e) + p1 * e)
    return np.array(pts, dtype=np.float32)


class MediaPipeWigRegionDetector:
    def __init__(self, cfg: WigMaskConfig) -> None:
        if mp is None or mp_tasks is None or mp_vision is None:
            raise RuntimeError(
                "mediapipe Tasks API is not available. Install requirements first. "
                f"Import error: {_MP_IMPORT_ERROR}"
            )

        model_path = _resolve_model_path(cfg)
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.last_anchor_points: dict[str, list[int]] = {}
        self.last_face_opening_bbox: list[int] = []
        self.last_transform_info: dict[str, float] = {}

    def detect_mask(self, image_bgr: np.ndarray, cfg: WigMaskConfig) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise RuntimeError("No face detected. Use a clear front-facing selfie.")

        lm = result.face_landmarks[0]

        def px(i: int) -> Tuple[int, int]:
            x = int(np.clip(lm[i].x * w, 0, w - 1))
            y = int(np.clip(lm[i].y * h, 0, h - 1))
            return x, y

        nose_x, nose_y = px(NOSE_TIP_IDX)
        left_temple_x, left_temple_y = px(LEFT_TEMPLE_IDX)
        right_temple_x, right_temple_y = px(RIGHT_TEMPLE_IDX)
        chin_x, chin_y = px(CHIN_IDX)
        forehead_x, forehead_y = px(FOREHEAD_CENTER_IDX)
        self.last_anchor_points = {
            "nose_center": [nose_x, nose_y],
            "left_temple": [left_temple_x, left_temple_y],
            "right_temple": [right_temple_x, right_temple_y],
            "chin_bottom": [chin_x, chin_y],
            "forehead_center": [forehead_x, forehead_y],
        }

        face_h = max(1, chin_y - nose_y)
        upper_y_limit = nose_y + int(face_h * cfg.upper_face_ratio)

        pts = np.array([px(i) for i in FACE_OVAL_IDX], dtype=np.int32)
        self.last_face_opening_bbox = [
            int(np.min(pts[:, 0])),
            int(np.min(pts[:, 1])),
            int(np.max(pts[:, 0])),
            int(np.max(pts[:, 1])),
        ]

        # Precise face contour mask from ordered oval landmarks.
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [pts], 255)

        if cfg.mask_mode == "face":
            points_used = [(int(p[0]), int(p[1])) for p in pts]
            return face_mask, points_used

        all_pts = np.array(
            [[float(np.clip(p.x * w, 0, w - 1)), float(np.clip(p.y * h, 0, h - 1))] for p in lm],
            dtype=np.float32,
        )
        if len(all_pts) < 10:
            raise RuntimeError("Insufficient landmarks for wig mask generation.")

        # Build selfie face mask from landmark hull (same plan as example_op.py).
        selfie_face_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(all_pts).astype(np.int32)
        cv2.fillConvexPoly(selfie_face_mask, hull, 255)
        selfie_geom = _geometry_from_face_mask(selfie_face_mask)

        # Load template mask + metadata.
        template_mask_path = _resolve_template_mask_path(cfg)
        template_meta_path = _resolve_template_meta_path(cfg)
        template_mask = _load_binary_mask(template_mask_path)
        with open(template_meta_path, "r", encoding="utf-8") as f:
            template_meta = json.load(f)
        _validate_template_meta(template_meta)

        # Warp template doorway mask onto selfie.
        M, transform_info = _compute_nonuniform_transform(template_meta, selfie_geom, cfg)
        mask = _warp_template_mask(template_mask, w, h, M)
        mask = _morph_mask(
            mask,
            dilate_px=max(0, int(cfg.template_post_dilate_px)),
            erode_px=max(0, int(cfg.template_post_erode_px)),
        )
        # Critical step: carve out detected selfie face opening from doorway wig mask.
        # Optional inset erodes face cutout so wig mask can overlap slightly into face.
        face_cutout = face_mask.copy()
        inset_px = max(0, int(cfg.template_face_cutout_inset_px))
        if inset_px > 0:
            k = inset_px * 2 + 1
            face_cutout = cv2.erode(face_cutout, np.ones((k, k), np.uint8), iterations=1)
        mask = cv2.subtract(mask, face_cutout)

        # Track transformed face-opening bbox for downstream/debug prints.
        tx1, ty1, tx2, ty2 = template_meta["face_opening_bbox"]
        p1 = np.array([M[0, 0] * tx1 + M[0, 2], M[1, 1] * ty1 + M[1, 2]], dtype=np.float32)
        p2 = np.array([M[0, 0] * tx2 + M[0, 2], M[1, 1] * ty2 + M[1, 2]], dtype=np.float32)
        self.last_face_opening_bbox = [
            int(np.clip(min(p1[0], p2[0]), 0, w - 1)),
            int(np.clip(min(p1[1], p2[1]), 0, h - 1)),
            int(np.clip(max(p1[0], p2[0]), 0, w - 1)),
            int(np.clip(max(p1[1], p2[1]), 0, h - 1)),
        ]

        contour = _largest_contour(mask)
        points_used = []
        if contour is not None:
            points_used = [(int(p[0][0]), int(p[0][1])) for p in contour]

        # Add useful debug payload
        self.last_transform_info = transform_info
        return mask, points_used
