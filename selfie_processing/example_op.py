import os
import json
import cv2
import numpy as np
import mediapipe as mp

# ============================================================
# CONFIG
# ============================================================
SELFIE_IMAGE_PATH = "selfie.jpg"
TEMPLATE_MASK_PATH = "template_mask.png"
TEMPLATE_META_PATH = "template_mask_meta.json"

# Optional: if you already have a binary face mask for the selfie, put it here.
# White = face, Black = background
SELFIE_FACE_MASK_PATH = None   # e.g. "selfie_face_mask.png"

# Only needed if SELFIE_FACE_MASK_PATH is None
MEDIAPIPE_MODEL_PATH = "face_landmarker.task"

OUTPUT_FINAL_MASK_PATH = "output_mask.png"
OUTPUT_SELFIE_FACE_MASK_PATH = "selfie_face_mask_debug.png"
OUTPUT_DEBUG_OVERLAY_PATH = "debug_overlay.png"

# Tuning
FACE_WIDTH_MULTIPLIER = 1.00
FACE_HEIGHT_MULTIPLIER = 1.00

# Optional final tweaks
POST_DILATE_PX = 0
POST_ERODE_PX = 0
FEATHER_PX = 0


# ============================================================
# MEDIAPIPE TASKS SETUP
# ============================================================
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# ============================================================
# HELPERS
# ============================================================
def load_binary_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask


def save_binary_mask(path, mask):
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(path, mask)


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x, y, x + w, y + h


def blur_binary_mask(mask, feather_px=0):
    if feather_px <= 0:
        return mask
    k = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(mask, (k, k), 0)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out


def morph_mask(mask, dilate_px=0, erode_px=0):
    out = mask.copy()

    if dilate_px > 0:
        k = dilate_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate(out, kernel, iterations=1)

    if erode_px > 0:
        k = erode_px * 2 + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.erode(out, kernel, iterations=1)

    return out


def normalized_landmarks_to_pixels(face_landmarks, width, height):
    pts = []
    for lm in face_landmarks:
        x = lm.x * width
        y = lm.y * height
        pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def build_face_mask_from_mediapipe(selfie_bgr, model_path):
    """
    Rough face mask from MediaPipe Tasks:
    - detect one face
    - take all landmarks
    - convex hull them
    - fill hull
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}\n"
            f"Provide SELFIE_FACE_MASK_PATH instead, or place the task model at this path."
        )

    h, w = selfie_bgr.shape[:2]
    selfie_rgb = cv2.cvtColor(selfie_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=selfie_rgb)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise RuntimeError("No face detected in selfie.")

    face_landmarks = result.face_landmarks[0]
    pts = normalized_landmarks_to_pixels(face_landmarks, w, h)

    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def geometry_from_face_mask(mask):
    """
    Derive reference geometry from a binary face mask.
    This is intentionally simple and deterministic.

    Returns:
        {
            "bbox": [x1, y1, x2, y2],
            "anchors": {...}
        }
    """
    contour = largest_contour(mask)
    if contour is None:
        raise RuntimeError("No face contour found in selfie face mask.")

    x1, y1, x2, y2 = contour_bbox(contour)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0

    # Derived anchors from face bbox proportions
    anchors = {
        "forehead_center": [cx, y1],
        "nose_center": [cx, y1 + h * 0.55],
        "chin_bottom": [cx, y2],
        "left_temple": [x1, y1 + h * 0.28],
        "right_temple": [x2, y1 + h * 0.28],
    }

    return {
        "bbox": [x1, y1, x2, y2],
        "anchors": anchors
    }


def validate_template_meta(meta):
    if "face_opening_bbox" not in meta:
        raise ValueError("template_mask_meta.json missing 'face_opening_bbox'")
    if "anchors" not in meta or "nose_center" not in meta["anchors"]:
        raise ValueError("template_mask_meta.json missing anchors.nose_center")


def compute_nonuniform_transform(template_meta, selfie_geom):
    """
    Option B:
    scale_x and scale_y independently, then translate so the template
    nose anchor lands on the selfie nose anchor.
    """
    tx1, ty1, tx2, ty2 = template_meta["face_opening_bbox"]
    sx1, sy1, sx2, sy2 = selfie_geom["bbox"]

    template_face_w = tx2 - tx1
    template_face_h = ty2 - ty1
    selfie_face_w = sx2 - sx1
    selfie_face_h = sy2 - sy1

    if template_face_w <= 0 or template_face_h <= 0:
        raise ValueError("Invalid template face_opening_bbox dimensions.")

    scale_x = (selfie_face_w / template_face_w) * FACE_WIDTH_MULTIPLIER
    scale_y = (selfie_face_h / template_face_h) * FACE_HEIGHT_MULTIPLIER

    template_anchor = template_meta["anchors"]["nose_center"]
    selfie_anchor = selfie_geom["anchors"]["nose_center"]

    tx_anchor_x, tx_anchor_y = template_anchor
    sx_anchor_x, sx_anchor_y = selfie_anchor

    translate_x = sx_anchor_x - (scale_x * tx_anchor_x)
    translate_y = sx_anchor_y - (scale_y * tx_anchor_y)

    M = np.array([
        [scale_x, 0.0, translate_x],
        [0.0, scale_y, translate_y]
    ], dtype=np.float32)

    return M, {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "translate_x": translate_x,
        "translate_y": translate_y
    }


def warp_template_mask(template_mask, output_w, output_h, M):
    warped = cv2.warpAffine(
        template_mask,
        M,
        (output_w, output_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    _, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)
    return warped


def transform_point(pt, M):
    x, y = pt
    tx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ty = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return [float(tx), float(ty)]


def make_debug_overlay(selfie_bgr, selfie_face_mask, transformed_mask, template_meta, selfie_geom, M):
    debug = selfie_bgr.copy()

    # Face mask contour in green
    contour = largest_contour(selfie_face_mask)
    if contour is not None:
        cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)

    # Transformed template mask contour in cyan
    contour2 = largest_contour(transformed_mask)
    if contour2 is not None:
        cv2.drawContours(debug, [contour2], -1, (255, 255, 0), 2)

    # Selfie bbox in yellow
    sx1, sy1, sx2, sy2 = [int(v) for v in selfie_geom["bbox"]]
    cv2.rectangle(debug, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)

    # Transformed template face bbox in magenta
    tx1, ty1, tx2, ty2 = template_meta["face_opening_bbox"]
    p1 = transform_point((tx1, ty1), M)
    p2 = transform_point((tx2, ty2), M)
    cv2.rectangle(
        debug,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        (255, 0, 255),
        2
    )

    # Anchor points
    for name, pt in template_meta["anchors"].items():
        tp = transform_point(pt, M)
        cv2.circle(debug, (int(tp[0]), int(tp[1])), 5, (0, 0, 255), -1)
        cv2.putText(debug, f"T:{name}", (int(tp[0]) + 6, int(tp[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    for name, pt in selfie_geom["anchors"].items():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(debug, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(debug, f"S:{name}", (x + 6, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    # Light red overlay fill for final mask
    overlay = debug.copy()
    overlay[transformed_mask > 0] = (0, 0, 255)
    debug = cv2.addWeighted(overlay, 0.18, debug, 0.82, 0)

    return debug


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(SELFIE_IMAGE_PATH):
        raise FileNotFoundError(f"Missing selfie image: {SELFIE_IMAGE_PATH}")

    if not os.path.exists(TEMPLATE_MASK_PATH):
        raise FileNotFoundError(f"Missing template mask: {TEMPLATE_MASK_PATH}")

    if not os.path.exists(TEMPLATE_META_PATH):
        raise FileNotFoundError(f"Missing template metadata: {TEMPLATE_META_PATH}")

    selfie_bgr = cv2.imread(SELFIE_IMAGE_PATH)
    if selfie_bgr is None:
        raise RuntimeError("Could not read selfie image.")

    selfie_h, selfie_w = selfie_bgr.shape[:2]

    template_mask = load_binary_mask(TEMPLATE_MASK_PATH)

    with open(TEMPLATE_META_PATH, "r", encoding="utf-8") as f:
        template_meta = json.load(f)

    validate_template_meta(template_meta)

    # Sanity check template size against metadata
    t_h, t_w = template_mask.shape[:2]
    if "image_size" in template_meta:
        meta_w, meta_h = template_meta["image_size"]
        if (meta_w != t_w) or (meta_h != t_h):
            print(
                f"[WARN] template_meta image_size={template_meta['image_size']} "
                f"but actual template_mask size={[t_w, t_h]}"
            )

    # --------------------------------------------------------
    # Get selfie face mask
    # --------------------------------------------------------
    if SELFIE_FACE_MASK_PATH and os.path.exists(SELFIE_FACE_MASK_PATH):
        selfie_face_mask = load_binary_mask(SELFIE_FACE_MASK_PATH)
        if selfie_face_mask.shape[:2] != (selfie_h, selfie_w):
            raise ValueError("SELFIE_FACE_MASK_PATH mask dimensions do not match selfie image.")
    else:
        selfie_face_mask = build_face_mask_from_mediapipe(selfie_bgr, MEDIAPIPE_MODEL_PATH)

    save_binary_mask(OUTPUT_SELFIE_FACE_MASK_PATH, selfie_face_mask)

    # --------------------------------------------------------
    # Derive selfie face geometry
    # --------------------------------------------------------
    selfie_geom = geometry_from_face_mask(selfie_face_mask)

    # --------------------------------------------------------
    # Compute transform and warp template
    # --------------------------------------------------------
    M, transform_info = compute_nonuniform_transform(template_meta, selfie_geom)

    transformed_mask = warp_template_mask(template_mask, selfie_w, selfie_h, M)

    # Optional cleanup/tuning
    transformed_mask = morph_mask(
        transformed_mask,
        dilate_px=POST_DILATE_PX,
        erode_px=POST_ERODE_PX
    )
    transformed_mask = blur_binary_mask(transformed_mask, FEATHER_PX)

    save_binary_mask(OUTPUT_FINAL_MASK_PATH, transformed_mask)

    # --------------------------------------------------------
    # Debug overlay
    # --------------------------------------------------------
    debug = make_debug_overlay(
        selfie_bgr,
        selfie_face_mask,
        transformed_mask,
        template_meta,
        selfie_geom,
        M
    )
    cv2.imwrite(OUTPUT_DEBUG_OVERLAY_PATH, debug)

    print("Done.")
    print(f"Saved final mask:        {OUTPUT_FINAL_MASK_PATH}")
    print(f"Saved selfie face mask:  {OUTPUT_SELFIE_FACE_MASK_PATH}")
    print(f"Saved debug overlay:     {OUTPUT_DEBUG_OVERLAY_PATH}")
    print("Transform info:")
    for k, v in transform_info.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()