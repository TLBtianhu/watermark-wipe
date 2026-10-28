"""通用图像工具：兼容中文路径的读写、遮罩处理、合成。"""
from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps


def imread_unicode(path: str) -> np.ndarray:
    """读取图片为 RGB uint8，兼容中文路径，并按 EXIF 方向自动摆正。"""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取图片: {path} ({exc})") from exc
    if rgb.size == 0:
        raise ValueError(f"图片为空: {path}")
    return np.ascontiguousarray(rgb)


def imwrite_unicode(path: str, rgb: np.ndarray, quality: int = 95) -> None:
    """保存 RGB 图片，兼容中文路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        ext = ".jpg"
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
        ext = ".png"
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise ValueError(f"图片编码失败: {path}")
    buf.tofile(path)


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("图片为空")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[2] == 3:
        return image
    raise ValueError(f"不支持的图片通道数: {image.shape}")


def boxes_to_mask(shape: tuple[int, int], boxes: Iterable, expand: int = 6) -> np.ndarray:
    """把检测框转成二值遮罩；shape=(h, w)。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = int(max(0, np.floor(x1 - expand)))
        y1 = int(max(0, np.floor(y1 - expand)))
        x2 = int(min(w, np.ceil(x2 + expand)))
        y2 = int(min(h, np.ceil(y2 + expand)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def mask_area_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).sum()) / float(mask.size) if mask.size else 0.0


def feather_composite(
    original: np.ndarray, repaired: np.ndarray, mask: np.ndarray, feather: int = 2
) -> np.ndarray:
    """用羽化后的遮罩把修复结果贴回原图，减少接缝。"""
    if mask is None or mask.size == 0 or int(mask.max()) == 0:
        return original.copy()
    alpha = (mask > 0).astype(np.float32)
    if feather > 0:
        k = max(3, feather * 2 + 1)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = alpha[..., None]
    out = original.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 45, 45),
    alpha: float = 0.45,
) -> np.ndarray:
    """在图片上叠加半透明红色遮罩，用于预览。"""
    out = image.copy()
    if mask is None or mask.size == 0:
        return out
    m = mask > 0
    if not m.any():
        return out
    c = np.array(color, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + c * alpha).astype(np.uint8)
    return out


def mask_to_rgba(mask: np.ndarray, color: tuple[int, int, int] = (255, 45, 45)) -> np.ndarray:
    """把二值遮罩转成 RGBA 图层，供 MaskEditor 显示。"""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.where(mask > 0, 255, 0).astype(np.uint8)
    return rgba


def mask_from_editor(value, shape: tuple[int, int]) -> np.ndarray:
    """从 Gradio ImageEditor / Sketch 的返回值提取二值遮罩。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if value is None:
        return mask

    layers = None
    if isinstance(value, dict):
        layers = value.get("layers")
        if layers is None and "mask" in value:
            layers = [value["mask"]]
    elif hasattr(value, "layers"):
        layers = getattr(value, "layers", None)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        # gr.Image(tool="sketch") 的 (image, mask) 形式
        layers = [value[1]]
    elif isinstance(value, np.ndarray):
        layers = [value]

    if layers is None:
        return mask
    if isinstance(layers, np.ndarray):
        layers = [layers]

    for layer in layers:
        if layer is None:
            continue
        arr = np.asarray(layer)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[0] != h or arr.shape[1] != w:
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        if arr.shape[2] >= 4:
            alpha = arr[..., 3]
            mask = np.maximum(mask, np.where(alpha > 8, 255, 0).astype(np.uint8))
        else:
            gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY) if arr.shape[2] == 3 else arr[..., 0]
            mask = np.maximum(mask, np.where(gray > 8, 255, 0).astype(np.uint8))
    return mask


def draw_boxes(image: np.ndarray, boxes: Iterable, confs: Iterable | None = None) -> np.ndarray:
    out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)
        if confs is not None:
            label = f"{float(confs[i]):.2f}"
            cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2, cv2.LINE_AA)
    return out
