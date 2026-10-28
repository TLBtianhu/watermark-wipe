"""输出尺寸解析：倍数 / 预设分辨率 / 自定义分辨率。"""
from __future__ import annotations

PRESETS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}

OUTPUT_CHOICES = [
    ("不放大（仅去水印）", "1"),
    ("2 倍", "2"),
    ("4 倍", "4"),
    ("720p（1280×720）", "720p"),
    ("1080p（1920×1080）", "1080p"),
    ("2K（2560×1440）", "2k"),
    ("4K（3840×2160）", "4k"),
    ("8K（7680×4320）", "8k"),
    ("自定义分辨率", "custom"),
]

FIT_CHOICES = [
    ("等比适配（推荐，输出可能略小）", "contain"),
    ("留黑边（严格目标尺寸）", "pad"),
    ("拉伸（严格目标尺寸）", "stretch"),
    ("裁剪填满（严格目标尺寸）", "cover"),
]


def resolve_output(
    choice: str, width: int | float | None = None, height: int | float | None = None
) -> tuple[float, tuple[int, int] | None]:
    """返回 (倍数, 目标尺寸)。目标尺寸非空时优先使用目标尺寸。"""
    choice = str(choice or "2")
    if choice in ("1", "2", "4"):
        return float(choice), None
    if choice in PRESETS:
        return 1.0, PRESETS[choice]
    if choice == "custom":
        try:
            w, h = int(width), int(height)
        except (TypeError, ValueError):
            raise ValueError("自定义分辨率需要填写整数宽高。") from None
        if not (64 <= w <= 8192 and 64 <= h <= 8192):
            raise ValueError("自定义分辨率宽高需在 64–8192 之间。")
        return 1.0, (w, h)
    raise ValueError(f"未知输出尺寸选项: {choice}")
