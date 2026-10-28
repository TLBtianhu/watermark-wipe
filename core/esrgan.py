"""Real-ESRGAN 超分（x2 / x4），使用 spandrel 加载官方权重。"""
from __future__ import annotations

import gc
import os
import time

import cv2
import numpy as np
import torch

MAX_OUTPUT_SIDE = 8192


class RealESRGANUpscaler:
    def __init__(
        self,
        model_paths: dict[int, str],
        device: str | torch.device = "cuda",
        tile: int = 512,
        context: int = 16,
    ):
        self.model_paths = {int(k): v for k, v in model_paths.items()}
        self.device = torch.device(device)
        self.tile = int(tile)
        self.context = int(context)
        self._cache: dict[int, tuple[torch.nn.Module, int]] = {}

    def _load(self, scale: int) -> tuple[torch.nn.Module, int]:
        if scale in self._cache:
            return self._cache[scale]
        path = self.model_paths.get(scale)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"找不到 Real-ESRGAN x{scale} 模型: {path}")

        from spandrel import ModelLoader

        t0 = time.time()
        try:
            desc = ModelLoader().load_from_file(path)
        except TypeError:
            desc = ModelLoader(device=self.device).load_from_file(path)  # type: ignore[call-arg]

        model = getattr(desc, "model", desc)
        model_scale = int(getattr(desc, "scale", scale) or scale)
        model.eval().to(self.device)
        if self.device.type == "cuda":
            model.half()
        self._cache[scale] = (model, model_scale)
        print(f"[RealESRGAN] loaded x{scale} in {time.time() - t0:.2f}s (arch scale={model_scale})")
        return model, model_scale

    @torch.inference_mode()
    def _run_tile(self, model: torch.nn.Module, tile_rgb: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(tile_rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            tensor = tensor.half()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(tensor)
        else:
            out = model(tensor)
        out = out.float().clamp_(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return (out * 255.0).round().astype(np.uint8)

    def _pad_to_multiple(self, image: np.ndarray, multiple: int = 4) -> tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        if ph == 0 and pw == 0:
            return image, h, w
        border = cv2.BORDER_REFLECT_101 if min(h, w) > 4 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(image, 0, ph, 0, pw, border)
        return padded, h, w

    def _upscale_with_model(self, image: np.ndarray, scale: int) -> np.ndarray:
        model, real_scale = self._load(scale)
        h, w = image.shape[:2]
        padded, orig_h, orig_w = self._pad_to_multiple(image, 4)
        ph, pw = padded.shape[:2]

        out = np.zeros((ph * real_scale, pw * real_scale, 3), dtype=np.uint8)
        ts = self.tile
        ctx = self.context
        for y in range(0, ph, ts):
            y2 = min(ph, y + ts)
            for x in range(0, pw, ts):
                x2 = min(pw, x + ts)
                cy0, cy1 = max(0, y - ctx), min(ph, y2 + ctx)
                cx0, cx1 = max(0, x - ctx), min(pw, x2 + ctx)
                tile = padded[cy0:cy1, cx0:cx1]
                tile_p, _, _ = self._pad_to_multiple(tile, 4)
                tile_out = self._run_tile(model, tile_p)
                oy0, ox0 = (y - cy0) * real_scale, (x - cx0) * real_scale
                oy1, ox1 = oy0 + (y2 - y) * real_scale, ox0 + (x2 - x) * real_scale
                out[y * real_scale : y2 * real_scale, x * real_scale : x2 * real_scale] = tile_out[
                    oy0:oy1, ox0:ox1
                ]

        out = out[: orig_h * real_scale, : orig_w * real_scale]
        return out

    def _normalize_target(self, target_size: tuple[int, int]) -> tuple[int, int]:
        tw, th = int(target_size[0]), int(target_size[1])
        tw, th = max(64, tw), max(64, th)
        longest = max(tw, th)
        if longest > MAX_OUTPUT_SIDE:
            f = MAX_OUTPUT_SIDE / float(longest)
            tw, th = max(64, int(round(tw * f))), max(64, int(round(th * f)))
        return tw, th

    def _apply_target(self, out: np.ndarray, tw: int, th: int, fit: str) -> np.ndarray:
        h, w = out.shape[:2]
        if fit == "stretch":
            return cv2.resize(out, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        if fit == "cover":
            f = max(tw / float(w), th / float(h))
            rw, rh = max(tw, int(round(w * f))), max(th, int(round(h * f)))
            resized = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
            x = (rw - tw) // 2
            y = (rh - th) // 2
            return resized[y : y + th, x : x + tw]

        # contain / pad：先等比缩放到目标框内
        f = min(tw / float(w), th / float(h))
        rw, rh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        resized = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
        if fit != "pad":
            return resized
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x = (tw - rw) // 2
        y = (th - rh) // 2
        canvas[y : y + rh, x : x + rw] = resized
        return canvas

    def upscale(
        self,
        image: np.ndarray,
        outscale: float = 4.0,
        target_size: tuple[int, int] | None = None,
        fit: str = "contain",
    ) -> tuple[np.ndarray, float]:
        """返回 (放大后的图, 实际倍数)。

        target_size 非空时按目标分辨率输出，fit 控制尺寸适配方式。
        """
        h, w = image.shape[:2]

        if target_size is not None:
            tw, th = self._normalize_target(target_size)
            if fit in ("contain", "pad"):
                needed = min(tw / float(w), th / float(h))
            else:
                needed = max(tw / float(w), th / float(h))

            if needed <= 1.01:
                out = image.copy()
            else:
                model_scale = 2 if needed <= 2.0 else 4
                out = self._upscale_with_model(image, model_scale)

            out = self._apply_target(out, tw, th, fit)
            actual = float(out.shape[1]) / float(w)

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()
            return out, actual

        target = float(outscale)
        max_allowed = MAX_OUTPUT_SIDE / float(max(h, w))
        target = min(target, max_allowed)
        if target <= 1.01:
            return image.copy(), 1.0

        model_scale = 2 if target <= 2.0 else 4
        out = self._upscale_with_model(image, model_scale)
        actual = float(out.shape[1]) / float(w)
        if abs(actual - target) > 0.01:
            new_w, new_h = max(1, int(round(w * target))), max(1, int(round(h * target)))
            out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            actual = float(new_w) / float(w)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        else:
            gc.collect()
        return out, actual
