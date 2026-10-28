"""LaMa 图像修复（TorchScript big-lama.pt）。"""
from __future__ import annotations

import gc
import os
import time

import cv2
import numpy as np
import torch

from .utils import feather_composite


class LamaInpainter:
    """基于 big-lama.pt 的遮罩修复。

    - 输入/输出均为 RGB uint8，遮罩为 0/255 uint8。
    - 大图会先缩放到 max_side 再修复，然后贴回原图，避免显存不足。
    """

    def __init__(self, model_path: str, device: str | torch.device = "cuda", max_side: int = 2048):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到 LaMa 模型: {model_path}")
        self.device = torch.device(device)
        self.max_side = int(max_side)
        t0 = time.time()
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.model.to(self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.load_seconds = time.time() - t0

    @torch.inference_mode()
    def _forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        border = cv2.BORDER_REFLECT_101 if min(h, w) > 8 else cv2.BORDER_REPLICATE
        image_p = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, border)
        mask_p = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        img_t = torch.from_numpy(image_p).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy((mask_p > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        out = self.model(img_t, mask_t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out[0].permute(1, 2, 0).float().cpu().numpy()

        # big-lama 正常情况下输出 0~1；若输出为 -1~1 则自动换算。
        if float(out.min()) < -0.05:
            out = (out + 1.0) * 0.5
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
        return out[:h, :w]

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        feather: int = 2,
        crop_margin: int | None = None,
    ) -> np.ndarray:
        if image is None or mask is None or int(mask.max()) == 0:
            return image.copy()

        h, w = image.shape[:2]

        # 视频逐帧处理时，只裁出水印附近区域送进 LaMa，速度可提升数倍。
        if crop_margin and crop_margin > 0:
            ys, xs = np.where(mask > 127)
            if len(xs) > 0:
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                if (x1 - x0) * (y1 - y0) < 0.45 * h * w:
                    m = int(crop_margin)
                    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
                    cx1, cy1 = min(w, x1 + m), min(h, y1 + m)
                    # 保证最小裁剪尺寸，避免上下文太少
                    min_side = 256
                    if cx1 - cx0 < min_side:
                        grow = (min_side - (cx1 - cx0)) // 2 + 1
                        cx0, cx1 = max(0, cx0 - grow), min(w, cx1 + grow)
                    if cy1 - cy0 < min_side:
                        grow = (min_side - (cy1 - cy0)) // 2 + 1
                        cy0, cy1 = max(0, cy0 - grow), min(h, cy1 + grow)
                    crop_img = image[cy0:cy1, cx0:cx1]
                    crop_mask = mask[cy0:cy1, cx0:cx1]
                    repaired_crop = self._run_with_oom_retry(crop_img, crop_mask)
                    repaired = image.copy()
                    repaired[cy0:cy1, cx0:cx1] = repaired_crop
                    return feather_composite(image, repaired, mask, feather=feather)

        longest = max(h, w)
        scale = min(1.0, float(self.max_side) / float(longest)) if longest > 0 else 1.0

        if scale < 1.0:
            new_w, new_h = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
            small_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            small_img, small_mask = image, mask

        repaired_small = self._run_with_oom_retry(small_img, small_mask)
        repaired = cv2.resize(repaired_small, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return feather_composite(image, repaired, mask, feather=feather)

    def _run_with_oom_retry(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sides = [max(image.shape[:2]), 1536, 1024, 768]
        last_err = None
        for side in sides:
            h, w = image.shape[:2]
            if max(h, w) > side:
                s = side / float(max(h, w))
                img = cv2.resize(image, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
                m = cv2.resize(mask, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_NEAREST)
            else:
                img, m = image, mask
            try:
                return self._forward(img, m)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # type: ignore[attr-defined]
                msg = str(exc).lower()
                if "out of memory" not in msg and "cuda" not in msg:
                    raise
                last_err = exc
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        raise RuntimeError(f"LaMa 修复失败（显存不足）: {last_err}")
