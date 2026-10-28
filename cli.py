"""命令行版：图片 / 视频 / 文件夹去水印 + 超分。"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2

from core.pipeline import OUTPUT_DIR, WatermarkNotFound, WatermarkSRPipeline
from core.utils import imread_unicode, imwrite_unicode
from core.video import extract_first_frame, is_video, process_video

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_OUT_EXTS = {".mp4", ".mov", ".mkv"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="一键去水印 + 超分（命令行版，支持图片/视频）")
    p.add_argument("input", help="输入图片、视频，或包含图片的文件夹")
    p.add_argument("-o", "--output", help="输出文件或输出文件夹（默认 output/）")
    p.add_argument(
        "-s", "--scale", type=float, default=None, choices=[1, 2, 4],
        help="超分倍数：1/2/4（默认图片 2，视频 1）",
    )
    p.add_argument("-c", "--conf", type=float, default=0.25, help="水印检测置信度阈值，默认 0.25")
    p.add_argument("-e", "--expand", type=int, default=8, help="遮罩向外扩展像素，默认 8")
    p.add_argument("-m", "--mask", help="手动遮罩图片（白色=修复区域），提供后不再自动检测")
    p.add_argument("--mode", choices=["fixed", "per_frame"], default="fixed", help="视频水印模式：固定遮罩/逐帧检测")
    p.add_argument("--preview", type=float, default=0, help="视频只处理前 N 秒（0=完整）")
    p.add_argument("--device", choices=["cuda", "cpu"], help="运行设备，默认自动选择")
    return p


def resolve_image_output(output: str | None, src: Path) -> Path:
    if output is None:
        return OUTPUT_DIR / f"{src.stem}_clean_sr.png"
    p = Path(output)
    if p.suffix.lower() in IMAGE_EXTS:
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{src.stem}_clean_sr.png"


def resolve_video_output(output: str | None, src: Path) -> Path:
    if output is None:
        return OUTPUT_DIR / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    p = Path(output)
    if p.suffix.lower() in VIDEO_OUT_EXTS:
        return p
    if p.suffix:
        return p.with_suffix(".mp4")
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{src.stem}_clean_sr.mp4"


def _load_mask(path: str | None):
    if not path:
        return None
    return cv2.cvtColor(imread_unicode(path), cv2.COLOR_RGB2GRAY)


def _video_progress():
    last = [-10]

    def report(frac: float, desc: str = ""):
        pct = int(float(frac) * 100)
        if pct // 10 != last[0] // 10:
            last[0] = pct
            print(f"  {pct:3d}%  {desc}", flush=True)

    return report


def run_video(pipeline: WatermarkSRPipeline, src: Path, args, mask) -> int:
    frame = extract_first_frame(src)
    if mask is None:
        _b, _s, mask = pipeline.detect(frame, conf=args.conf, expand=args.expand)
        if int(mask.max()) == 0:
            print("首帧没有检测到水印。请降低 -c、用 -m 提供遮罩，或改用手动遮罩。")
            return 1

    out_path = resolve_video_output(args.output, src)
    print(f"处理视频: {src.name}  ({frame.shape[1]}x{frame.shape[0]}, 模式={args.mode})", flush=True)
    try:
        _, info = process_video(
            src,
            mask=mask,
            scale=float(args.scale if args.scale is not None else 1),
            conf=args.conf,
            expand=args.expand,
            mode=args.mode,
            preview_seconds=args.preview,
            output_path=out_path,
            pipeline=pipeline,
            progress=_video_progress(),
        )
    except WatermarkNotFound as exc:
        print(f"失败：{exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"失败：{exc}")
        return 1

    print(f"已保存: {out_path}")
    print(
        f"输出: {info['output_size'][0]}x{info['output_size'][1]}  "
        f"{info['frames']} 帧 / {info['duration']:g}s  "
        f"音频={'保留' if info['audio'] else '无'}  耗时={info['timings']['total']}s"
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.input)
    if not src.exists():
        print(f"输入不存在: {src}")
        return 2

    pipeline = WatermarkSRPipeline(device=args.device)
    mask = _load_mask(args.mask)

    if src.is_dir():
        out_dir = Path(args.output) if args.output else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted([p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not files:
            print("文件夹中没有找到图片（批量文件夹目前只处理图片）。")
            return 2
        ok = 0
        scale = float(args.scale if args.scale is not None else 2)
        for i, f in enumerate(files, 1):
            out_path = out_dir / f"{f.stem}_clean_sr.png"
            print(f"[{i}/{len(files)}] {f.name} ...", flush=True)
            try:
                image = imread_unicode(str(f))
                result, info = pipeline.process(
                    image, mask=None, scale=scale, conf=args.conf, expand=args.expand
                )
                imwrite_unicode(str(out_path), result)
                print(
                    f"    -> {out_path}  ({info['output_size'][0]}x{info['output_size'][1]}, "
                    f"{info['timings']['total']}s)"
                )
                ok += 1
            except WatermarkNotFound as exc:
                print(f"    跳过：{exc}")
            except Exception as exc:  # noqa: BLE001
                print(f"    失败：{exc}")
        print(f"\n完成 {ok}/{len(files)} 张。")
        return 0 if ok else 1

    if is_video(src):
        return run_video(pipeline, src, args, mask)

    scale = float(args.scale if args.scale is not None else 2)
    out_path = resolve_image_output(args.output, src)
    try:
        image = imread_unicode(str(src))
        result, info = pipeline.process(
            image, mask=mask, scale=scale, conf=args.conf, expand=args.expand
        )
    except WatermarkNotFound as exc:
        print(f"失败：{exc}")
        return 1

    imwrite_unicode(str(out_path), result)
    print(f"已保存: {out_path}")
    print(f"输出尺寸: {info['output_size'][0]}x{info['output_size'][1]}  耗时: {info['timings']['total']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
