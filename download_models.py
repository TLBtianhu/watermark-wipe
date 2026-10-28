#!/usr/bin/env python3
"""下载 ClearMark 运行所需的模型权重。

权重不放进 Git 仓库（体积大，且部分模型有单独的许可条款）。
默认从官方 GitHub Release / HuggingFace 下载到 models/ 目录。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

MODELS = [
    {
        "name": "watermark_yolo11x.pt",
        "url": "https://huggingface.co/corzent/yolo11x_watermark_detection/resolve/main/best.pt",
        "size": 114_434_898,
        "label": "YOLO11x 水印检测",
    },
    {
        "name": "big-lama.pt",
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "size": 205_669_692,
        "label": "LaMa 图像修复",
    },
    {
        "name": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "size": 67_061_725,
        "label": "Real-ESRGAN 2× 超分",
    },
    {
        "name": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "size": 67_040_989,
        "label": "Real-ESRGAN 4× 超分",
    },
]


def _mb(n: int | float) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


def _print_progress(downloaded: int, total: int, name: str) -> None:
    if total <= 0:
        print(f"\r  {name}  {_mb(downloaded)}", end="", flush=True)
        return
    pct = min(100.0, downloaded * 100.0 / total)
    filled = int(pct / 2)
    bar = "#" * filled + "-" * (50 - filled)
    print(f"\r  {name} [{bar}] {pct:5.1f}%  {_mb(downloaded)} / {_mb(total)}", end="", flush=True)


def download(item: dict, dest: Path, force: bool = False) -> bool:
    expected = int(item["size"])
    if dest.exists() and not force:
        actual = dest.stat().st_size
        if actual == expected:
            print(f"[skip] {dest.name} 已存在（{_mb(actual)}）")
            return True
        print(f"[warn] {dest.name} 大小不对（{_mb(actual)} != {_mb(expected)}），重新下载")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(item["url"], headers={"User-Agent": "ClearMark/1.0"})
    print(f"[get ] {item['label']} -> {dest.name}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or expected)
            downloaded = 0
            last = 0.0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last > 0.15:
                    _print_progress(downloaded, total, dest.name)
                    last = now
            _print_progress(downloaded, total, dest.name)
            print()
        actual = part.stat().st_size
        if actual != expected:
            part.unlink(missing_ok=True)
            print(f"[fail] {dest.name} 大小校验失败：{_mb(actual)} != {_mb(expected)}")
            return False
        part.replace(dest)
        print(f"[ok  ] {dest.name}  {_mb(actual)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n[fail] {dest.name}: {exc}")
        part.unlink(missing_ok=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 ClearMark 所需模型")
    parser.add_argument("--proxy", help="代理地址，例如 http://127.0.0.1:7897")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    parser.add_argument("--only", choices=[m["name"] for m in MODELS], help="只下载某一个")
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        print(f"使用代理: {args.proxy}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MODELS if not args.only or m["name"] == args.only]
    ok = 0
    for item in todo:
        if download(item, MODEL_DIR / item["name"], force=args.force):
            ok += 1

    print(f"\n完成 {ok}/{len(todo)} 个模型，目录: {MODEL_DIR}")
    if ok != len(todo):
        print("有模型下载失败。国内网络可加 --proxy http://127.0.0.1:7897 后重试。")
        return 1
    print("现在可以双击 start.bat，或运行 python app.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
