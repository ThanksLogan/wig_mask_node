import os

import cv2
import numpy as np
import torch

from .selfie_processing.hair_pipeline import (
    build_tryon_mask_from_wig_hair,
    isolate_hair_mask,
    overlay_dual_mask_preview,
    prepare_wig_selfie_tryon,
    remove_selfie_hair,
)
from .selfie_processing.pipeline import generate_wig_region_mask
from .selfie_processing.types import WigMaskConfig

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(NODE_DIR, "models", "face_landmarker.task")
DEFAULT_TEMPLATE_MASK_PATH = os.path.join(NODE_DIR, "images", "template_mask.png")
DEFAULT_TEMPLATE_META_PATH = os.path.join(NODE_DIR, "images", "template_mask_meta.json")
DEFAULT_SEGMENTER_MODEL_PATH = os.path.join(NODE_DIR, "models", "selfie_multiclass_256x256.tflite")


def tensor_to_bgr_u8(image_tensor):
    img = image_tensor.detach().cpu().numpy()
    rgb = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if rgb.ndim != 3:
        raise ValueError(f"Expected image tensor slice with shape [H,W,C], got {tuple(rgb.shape)}")
    if rgb.shape[-1] == 4:
        return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    if rgb.shape[-1] == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Expected image tensor slice with 3 or 4 channels, got {rgb.shape[-1]}")


def bgr_u8_to_comfy_image(image_bgr):
    if image_bgr.ndim != 3:
        raise ValueError(f"Expected image array with shape [H,W,C], got {tuple(image_bgr.shape)}")
    if image_bgr.shape[-1] == 4:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2RGBA)
    elif image_bgr.shape[-1] == 3:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Expected image array with 3 or 4 channels, got {image_bgr.shape[-1]}")
    image_f = image.astype(np.float32) / 255.0
    return torch.from_numpy(image_f)[None, ...]


def cutout_bgr_with_alpha(image_bgr, mask_u8, feather_ksize=0):
    alpha = mask_u8.astype(np.uint8)
    feather_ksize = int(feather_ksize)
    if feather_ksize > 1:
        if feather_ksize % 2 == 0:
            feather_ksize += 1
        alpha = cv2.GaussianBlur(alpha, (feather_ksize, feather_ksize), 0)

    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def mask_u8_to_comfy_mask(mask_u8):
    return torch.from_numpy(mask_u8.astype(np.float32) / 255.0)


def comfy_mask_to_u8(mask_tensor):
    arr = mask_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return np.where(arr > 0.5, 255, 0).astype(np.uint8)


def _shared_cfg(
    model_path,
    image_segmenter_model_path,
    template_mask_path=DEFAULT_TEMPLATE_MASK_PATH,
    template_meta_path=DEFAULT_TEMPLATE_META_PATH,
    template_face_cutout_inset_px=4,
    segmentation_threshold=0.15,
):
    return WigMaskConfig(
        face_landmarker_model_path=model_path or None,
        image_segmenter_model_path=image_segmenter_model_path or None,
        template_mask_path=template_mask_path or None,
        template_meta_path=template_meta_path or None,
        template_face_cutout_inset_px=int(template_face_cutout_inset_px),
        segmentation_threshold=float(segmentation_threshold),
    )


class WigMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["wig", "face"],),
                "template_face_cutout_inset_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
                "template_mask_path": ("STRING", {"default": DEFAULT_TEMPLATE_MASK_PATH}),
                "template_meta_path": ("STRING", {"default": DEFAULT_TEMPLATE_META_PATH}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        image,
        mode,
        template_face_cutout_inset_px,
        model_path=DEFAULT_MODEL_PATH,
        template_mask_path=DEFAULT_TEMPLATE_MASK_PATH,
        template_meta_path=DEFAULT_TEMPLATE_META_PATH,
    ):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(image.shape)}")

        masks = []
        previews = []
        for i in range(image.shape[0]):
            bgr = tensor_to_bgr_u8(image[i])
            cfg = WigMaskConfig(
                mask_mode=mode,
                face_landmarker_model_path=model_path or None,
                template_mask_path=template_mask_path or None,
                template_meta_path=template_meta_path or None,
                template_face_cutout_inset_px=int(template_face_cutout_inset_px),
            )
            mask_u8, debug = generate_wig_region_mask(bgr, cfg)
            masks.append(mask_u8_to_comfy_mask(mask_u8))
            preview_bgr = overlay_dual_mask_preview(bgr, mask_u8, getattr(debug, "base_mask", None))
            previews.append(bgr_u8_to_comfy_image(preview_bgr))

        return (torch.stack(masks, dim=0), torch.cat(previews, dim=0))


class WigHairIsolationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segmentation_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.99, "step": 0.01}),
                "hair_face_dilate_px": ("INT", {"default": 8, "min": 0, "max": 64}),
                "hair_post_dilate_px": ("INT", {"default": 2, "min": 0, "max": 64}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
                "image_segmenter_model_path": ("STRING", {"default": DEFAULT_SEGMENTER_MODEL_PATH}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = ("hair_mask", "face_mask", "isolated_hair", "preview")
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        image,
        segmentation_threshold,
        hair_face_dilate_px,
        hair_post_dilate_px,
        model_path=DEFAULT_MODEL_PATH,
        image_segmenter_model_path=DEFAULT_SEGMENTER_MODEL_PATH,
    ):
        hair_masks = []
        face_masks = []
        isolated_images = []
        previews = []

        for i in range(image.shape[0]):
            bgr = tensor_to_bgr_u8(image[i])
            cfg = _shared_cfg(model_path, image_segmenter_model_path, segmentation_threshold=segmentation_threshold)
            cfg.hair_face_dilate_px = int(hair_face_dilate_px)
            cfg.hair_post_dilate_px = int(hair_post_dilate_px)

            hair_mask_u8, isolated_bgr, debug = isolate_hair_mask(bgr, cfg)
            face_mask_u8 = np.where(debug.extra["face_mask"] > 0, 255, 0).astype(np.uint8)
            preview_bgr = overlay_dual_mask_preview(isolated_bgr, hair_mask_u8, face_mask_u8)
            isolated_rgba = cutout_bgr_with_alpha(bgr, hair_mask_u8, feather_ksize=cfg.hair_blur_ksize)

            hair_masks.append(mask_u8_to_comfy_mask(hair_mask_u8))
            face_masks.append(mask_u8_to_comfy_mask(face_mask_u8))
            isolated_images.append(bgr_u8_to_comfy_image(isolated_rgba))
            previews.append(bgr_u8_to_comfy_image(preview_bgr))

        return (
            torch.stack(hair_masks, dim=0),
            torch.stack(face_masks, dim=0),
            torch.cat(isolated_images, dim=0),
            torch.cat(previews, dim=0),
        )


class SelfieHairEraseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fill_mode": (["gray", "blur", "black"],),
                "segmentation_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
                "image_segmenter_model_path": ("STRING", {"default": DEFAULT_SEGMENTER_MODEL_PATH}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("hair_removed", "hair_mask", "preview")
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        image,
        fill_mode,
        segmentation_threshold,
        model_path=DEFAULT_MODEL_PATH,
        image_segmenter_model_path=DEFAULT_SEGMENTER_MODEL_PATH,
    ):
        outputs = []
        masks = []
        previews = []

        for i in range(image.shape[0]):
            bgr = tensor_to_bgr_u8(image[i])
            cfg = _shared_cfg(model_path, image_segmenter_model_path, segmentation_threshold=segmentation_threshold)
            erased_bgr, hair_mask_u8, _debug = remove_selfie_hair(bgr, cfg, fill_mode=fill_mode)
            preview_bgr = overlay_dual_mask_preview(erased_bgr, hair_mask_u8)

            outputs.append(bgr_u8_to_comfy_image(erased_bgr))
            masks.append(mask_u8_to_comfy_mask(hair_mask_u8))
            previews.append(bgr_u8_to_comfy_image(preview_bgr))

        return (torch.cat(outputs, dim=0), torch.stack(masks, dim=0), torch.cat(previews, dim=0))


class WigToSelfieMaskNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wig_hair_mask": ("MASK",),
                "wig_face_mask": ("MASK",),
                "selfie_image": ("IMAGE",),
                "template_face_cutout_inset_px": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        wig_hair_mask,
        wig_face_mask,
        selfie_image,
        template_face_cutout_inset_px,
        model_path=DEFAULT_MODEL_PATH,
    ):
        if selfie_image.ndim != 4:
            raise ValueError(f"Expected selfie_image IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(selfie_image.shape)}")

        if wig_hair_mask.ndim == 2:
            wig_hair_mask = wig_hair_mask.unsqueeze(0)
        if wig_face_mask.ndim == 2:
            wig_face_mask = wig_face_mask.unsqueeze(0)

        batch = selfie_image.shape[0]
        if wig_hair_mask.shape[0] not in (1, batch):
            raise ValueError("wig_hair_mask batch must be 1 or match selfie_image batch.")
        if wig_face_mask.shape[0] not in (1, batch):
            raise ValueError("wig_face_mask batch must be 1 or match selfie_image batch.")

        masks = []
        previews = []
        for i in range(batch):
            bgr = tensor_to_bgr_u8(selfie_image[i])
            hair_mask_u8 = comfy_mask_to_u8(wig_hair_mask[0 if wig_hair_mask.shape[0] == 1 else i])
            face_mask_u8 = comfy_mask_to_u8(wig_face_mask[0 if wig_face_mask.shape[0] == 1 else i])

            cfg = _shared_cfg(
                model_path,
                None,
                template_face_cutout_inset_px=template_face_cutout_inset_px,
            )
            mask_u8, debug = build_tryon_mask_from_wig_hair(hair_mask_u8, face_mask_u8, bgr, cfg)
            preview_bgr = overlay_dual_mask_preview(bgr, mask_u8, debug.extra.get("selfie_face_mask"))

            masks.append(mask_u8_to_comfy_mask(mask_u8))
            previews.append(bgr_u8_to_comfy_image(preview_bgr))

        return (torch.stack(masks, dim=0), torch.cat(previews, dim=0))


class WigSelfiePrepNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "wig_image": ("IMAGE",),
                "selfie_image": ("IMAGE",),
                "fill_mode": (["gray", "blur", "black"],),
                "segmentation_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.99, "step": 0.01}),
                "template_face_cutout_inset_px": ("INT", {"default": 4, "min": 0, "max": 64}),
                "wig_mask_expand_px": ("INT", {"default": 16, "min": 0, "max": 128}),
                "wig_face_overlap_expand_px": ("INT", {"default": 4, "min": 0, "max": 64}),
                "wig_top_trim_px": ("INT", {"default": 0, "min": 0, "max": 64}),
                "selfie_hair_blur_expand_px": ("INT", {"default": 0, "min": 0, "max": 64}),
                "selfie_hair_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "connect_floating_face_hair": ("BOOLEAN", {"default": True}),
                "floating_face_hair_max_expand_px": ("INT", {"default": 18, "min": 0, "max": 64}),
                "fill_face_islands": ("BOOLEAN", {"default": True}),
                "face_island_exclusion_pad_px": ("INT", {"default": 10, "min": 0, "max": 64}),
                "face_island_ring_px": ("INT", {"default": 14, "min": 1, "max": 64}),
                "face_island_max_area_ratio": ("FLOAT", {"default": 1.0, "min": 0.005, "max": 1.0, "step": 0.005}),
                "face_island_max_center_y_ratio": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 1.0, "step": 0.01}),
                "hair_face_dilate_px": ("INT", {"default": 8, "min": 0, "max": 64}),
                "hair_post_dilate_px": ("INT", {"default": 2, "min": 0, "max": 64}),
            },
            "optional": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH}),
                "image_segmenter_model_path": ("STRING", {"default": DEFAULT_SEGMENTER_MODEL_PATH}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "wig_face_hair_mask",
        "selfie_face_mask",
        "transfer_mask",
        "selfie_hair_removed",
        "wig_preview",
        "transfer_preview",
    )
    FUNCTION = "run"
    CATEGORY = "WigAI"

    def run(
        self,
        wig_image,
        selfie_image,
        fill_mode,
        segmentation_threshold,
        template_face_cutout_inset_px,
        wig_mask_expand_px,
        wig_face_overlap_expand_px,
        wig_top_trim_px,
        selfie_hair_blur_expand_px,
        selfie_hair_blur_strength,
        connect_floating_face_hair,
        floating_face_hair_max_expand_px,
        fill_face_islands,
        face_island_exclusion_pad_px,
        face_island_ring_px,
        face_island_max_area_ratio,
        face_island_max_center_y_ratio,
        hair_face_dilate_px,
        hair_post_dilate_px,
        model_path=DEFAULT_MODEL_PATH,
        image_segmenter_model_path=DEFAULT_SEGMENTER_MODEL_PATH,
    ):
        if wig_image.ndim != 4:
            raise ValueError(f"Expected wig_image IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(wig_image.shape)}")
        if selfie_image.ndim != 4:
            raise ValueError(f"Expected selfie_image IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(selfie_image.shape)}")

        batch = max(wig_image.shape[0], selfie_image.shape[0])
        if wig_image.shape[0] not in (1, batch):
            raise ValueError("wig_image batch must be 1 or match selfie_image batch.")
        if selfie_image.shape[0] not in (1, batch):
            raise ValueError("selfie_image batch must be 1 or match wig_image batch.")

        wig_masks = []
        selfie_face_masks = []
        transfer_masks = []
        selfie_outputs = []
        wig_previews = []
        transfer_previews = []

        for i in range(batch):
            wig_bgr = tensor_to_bgr_u8(wig_image[0 if wig_image.shape[0] == 1 else i])
            selfie_bgr = tensor_to_bgr_u8(selfie_image[0 if selfie_image.shape[0] == 1 else i])
            cfg = _shared_cfg(
                model_path,
                image_segmenter_model_path,
                template_face_cutout_inset_px=template_face_cutout_inset_px,
                segmentation_threshold=segmentation_threshold,
            )
            cfg.wig_mask_expand_px = int(wig_mask_expand_px)
            cfg.wig_face_overlap_expand_px = int(wig_face_overlap_expand_px)
            cfg.wig_top_trim_px = int(wig_top_trim_px)
            cfg.selfie_hair_blur_expand_px = int(selfie_hair_blur_expand_px)
            cfg.selfie_hair_blur_strength = float(selfie_hair_blur_strength)
            cfg.connect_floating_face_hair = bool(connect_floating_face_hair)
            cfg.floating_face_hair_max_expand_px = int(floating_face_hair_max_expand_px)
            cfg.fill_face_islands = bool(fill_face_islands)
            cfg.face_island_exclusion_pad_px = int(face_island_exclusion_pad_px)
            cfg.face_island_ring_px = int(face_island_ring_px)
            cfg.face_island_max_area_ratio = float(face_island_max_area_ratio)
            cfg.face_island_max_center_y_ratio = float(face_island_max_center_y_ratio)
            cfg.hair_face_dilate_px = int(hair_face_dilate_px)
            cfg.hair_post_dilate_px = int(hair_post_dilate_px)

            result = prepare_wig_selfie_tryon(wig_bgr, selfie_bgr, cfg, fill_mode=fill_mode)
            wig_face_hair_mask = result["wig_face_hair_mask"]
            selfie_face_mask = result["selfie_face_mask"]
            transfer_mask = result["final_mask"]
            selfie_hair_removed = result["selfie_hair_removed"]

            wig_preview_bgr = overlay_dual_mask_preview(wig_bgr, wig_face_hair_mask, result["wig_face_mask"])
            transfer_preview_bgr = overlay_dual_mask_preview(selfie_hair_removed, transfer_mask, selfie_face_mask)

            wig_masks.append(mask_u8_to_comfy_mask(wig_face_hair_mask))
            selfie_face_masks.append(mask_u8_to_comfy_mask(selfie_face_mask))
            transfer_masks.append(mask_u8_to_comfy_mask(transfer_mask))
            selfie_outputs.append(bgr_u8_to_comfy_image(selfie_hair_removed))
            wig_previews.append(bgr_u8_to_comfy_image(wig_preview_bgr))
            transfer_previews.append(bgr_u8_to_comfy_image(transfer_preview_bgr))

        return (
            torch.stack(wig_masks, dim=0),
            torch.stack(selfie_face_masks, dim=0),
            torch.stack(transfer_masks, dim=0),
            torch.cat(selfie_outputs, dim=0),
            torch.cat(wig_previews, dim=0),
            torch.cat(transfer_previews, dim=0),
        )


NODE_CLASS_MAPPINGS = {
    "WigMaskNode": WigMaskNode,
    "WigHairIsolationNode": WigHairIsolationNode,
    "SelfieHairEraseNode": SelfieHairEraseNode,
    "WigToSelfieMaskNode": WigToSelfieMaskNode,
    "WigSelfiePrepNode": WigSelfiePrepNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WigMaskNode": "Wig Mask (Template+MP)",
    "WigHairIsolationNode": "Wig Hair Isolation (MP)",
    "SelfieHairEraseNode": "Selfie Hair Erase (MP)",
    "WigToSelfieMaskNode": "Wig To Selfie Mask (MP)",
    "WigSelfiePrepNode": "Wig Selfie Prep (MP)",
}
