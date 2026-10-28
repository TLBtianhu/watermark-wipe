"""水印/Logo 检测：YOLO11x（corzent/yolo11x_watermark_detection）。"""
from __future__ import annotations

import os
import time

import numpy as np
from PIL import Image

from .utils import boxes_to_mask


class WatermarkDetector:
    def __init__(self, model_path: str, device: str = "cuda", imgsz: int = 1280):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到水印检测模型: {model_path}")
        from ultralytics import YOLO

        self.device = 0 if str(device).startswith("cuda") else "cpu"
        self.imgsz = int(imgsz)
        t0 = time.time()
        self.model = YOLO(model_path)
        self.names = getattr(self.model, "names", {0: "watermark"})
        self.load_seconds = time.time() - t0

    def detect(
        self, image: np.ndarray, conf: float = 0.25, expand: int = 6, imgsz: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (boxes Nx4, scores N, mask HxW)。"""
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        results = self.model.predict(
            source=pil,
            imgsz=int(imgsz or self.imgsz),
            conf=float(conf),
            iou=0.5,
            verbose=False,
            device=self.device,
        )
        if not results:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((h, w), dtype=np.uint8)

        r = results[0]
        boxes_t = getattr(r, "boxes", None)
        if boxes_t is None or len(boxes_t) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.float32), np.zeros((h, w), dtype=np.uint8)

        boxes = boxes_t.xyxy.detach().float().cpu().numpy()
        scores = boxes_t.conf.detach().float().cpu().numpy()
        # 过滤退化框
        keep = (boxes[:, 2] > boxes[:, 0] + 2) & (boxes[:, 3] > boxes[:, 1] + 2)
        boxes, scores = boxes[keep], scores[keep]
        mask = boxes_to_mask((h, w), boxes, expand=expand)
        return boxes.astype(np.float32), scores.astype(np.float32), mask
