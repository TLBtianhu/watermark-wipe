"""去水印 + 超分流水线。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .detector import WatermarkDetector
from .esrgan import RealESRGANUpscaler
from .lama import LamaInpainter
from .utils import (
    draw_boxes,
    ensure_rgb_uint8,
    mask_area_ratio,
    overlay_mask,
)

from .paths import MODEL_DIR, OUTPUT_DIR, ROOT


class WatermarkNotFound(RuntimeError):
    pass


def pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class WatermarkSRPipeline:
    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        device: str | None = None,
        lama_max_side: int = 2048,
    ):
        self.model_dir = Path(model_dir)
        self.device = device or pick_device()
        self.lama_max_side = lama_max_side
        self._detector: WatermarkDetector | None = None
        self._inpainter: LamaInpainter | None = None
        self._upscaler: RealESRGANUpscaler | None = None

    @property
    def detector(self) -> WatermarkDetector:
        if self._detector is None:
            t0 = time.time()
            self._detector = WatermarkDetector(
                str(self.model_dir / "watermark_yolo11x.pt"), device=self.device
            )
            print(f"[pipeline] detector ready in {time.time() - t0:.2f}s")
        return self._detector

    @property
    def inpainter(self) -> LamaInpainter:
        if self._inpainter is None:
            t0 = time.time()
            self._inpainter = LamaInpainter(
                str(self.model_dir / "big-lama.pt"), device=self.device, max_side=self.lama_max_side
            )
            print(f"[pipeline] LaMa ready in {time.time() - t0:.2f}s")
        return self._inpainter

    @property
    def upscaler(self) -> RealESRGANUpscaler:
        if self._upscaler is None:
            self._upscaler = RealESRGANUpscaler(
                {
                    2: str(self.model_dir / "RealESRGAN_x2plus.pth"),
                    4: str(self.model_dir / "RealESRGAN_x4plus.pth"),
                },
                device=self.device,
            )
        return self._upscaler

    def detect(
        self, image: np.ndarray, conf: float = 0.25, expand: int = 6
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = ensure_rgb_uint8(image)
        return self.detector.detect(image, conf=conf, expand=expand)

    def detect_preview(
        self, image: np.ndarray, conf: float = 0.25, expand: int = 6
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        image = ensure_rgb_uint8(image)
        boxes, scores, mask = self.detector.detect(image, conf=conf, expand=expand)
        view = overlay_mask(image, mask)
        view = draw_boxes(view, boxes, scores)
        return boxes, scores, mask, view

    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        scale: float = 2,
        conf: float = 0.25,
        expand: int = 6,
        progress=None,
    ) -> tuple[np.ndarray, dict]:
        """去水印并超分。

        mask 为空时会自动检测水印；返回 (结果图, 信息字典)。
        """
        image = ensure_rgb_uint8(image)
        h, w = image.shape[:2]
        timings: dict[str, float] = {}
        info: dict = {"device": self.device, "input_size": [w, h]}

        t_all = time.perf_counter()
        if progress:
            progress(0.05, desc="检测水印…")
        t0 = time.perf_counter()

        auto_used = False
        boxes = np.zeros((0, 4), dtype=np.float32)
        if mask is None or int(np.asarray(mask).max()) == 0:
            boxes, _scores, mask = self.detector.detect(image, conf=conf, expand=expand)
            auto_used = True
        else:
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.shape[:2] != (h, w):
                import cv2

                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        timings["detect"] = time.perf_counter() - t0
        ratio = mask_area_ratio(mask)
        info.update(
            {
                "auto_detected": auto_used,
                "boxes": int(len(boxes)),
                "mask_ratio": ratio,
            }
        )
        if int(mask.max()) == 0:
            raise WatermarkNotFound("没有检测到水印。请降低检测阈值，或手动涂抹水印区域后重试。")
        if ratio > 0.75:
            raise WatermarkNotFound(
                f"检测到的遮罩覆盖了 {ratio * 100:.0f}% 画面，可能是误检。请调高检测阈值或手动涂抹。"
            )

        if progress:
            progress(0.25, desc="LaMa 修复中…")
        t0 = time.perf_counter()
        cleaned = self.inpainter.inpaint(image, mask)
        timings["inpaint"] = time.perf_counter() - t0

        actual_scale = 1.0
        if float(scale) > 1.01:
            if progress:
                progress(0.55, desc=f"Real-ESRGAN x{int(scale)} 超分中…")
            t0 = time.perf_counter()
            result, actual_scale = self.upscaler.upscale(cleaned, outscale=float(scale))
            timings["upscale"] = time.perf_counter() - t0
        else:
            result = cleaned

        timings["total"] = time.perf_counter() - t_all
        info.update(
            {
                "output_size": [int(result.shape[1]), int(result.shape[0])],
                "scale": round(float(actual_scale), 3),
                "timings": {k: round(v, 2) for k, v in timings.items()},
            }
        )
        if progress:
            progress(1.0, desc="完成")
        return result, info

    def process_file(
        self,
        input_path: str,
        output_path: str | None = None,
        scale: float = 2,
        conf: float = 0.25,
        expand: int = 6,
        mask_path: str | None = None,
    ) -> tuple[str, dict]:
        from .utils import imread_unicode, imwrite_unicode

        image = imread_unicode(input_path)
        mask = None
        if mask_path:
            import cv2

            mask = cv2.cvtColor(imread_unicode(mask_path), cv2.COLOR_RGB2GRAY)
        result, info = self.process(image, mask=mask, scale=scale, conf=conf, expand=expand)
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stem = Path(input_path).stem
            output_path = str(OUTPUT_DIR / f"{stem}_clean_sr.png")
        imwrite_unicode(output_path, result)
        info["output_path"] = output_path
        return output_path, info
