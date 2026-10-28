"""路径解析：兼容源码运行和 PyInstaller 冻结后的目录结构。"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """只读资源根目录（源码目录；冻结后是 PyInstaller 的 _internal / _MEIPASS）。"""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def writable_root() -> Path:
    """可写目录（源码目录；冻结后是 exe 所在目录）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = bundle_root()
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = writable_root() / "output"
