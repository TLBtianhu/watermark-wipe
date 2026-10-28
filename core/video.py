"""视频去水印 + 超分：抽帧 → 逐帧 LaMa 修复 → 可选 Real-ESRGAN → 带音频重编码。"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pipeline import OUTPUT_DIR, WatermarkNotFound, WatermarkSRPipeline
from .utils import imread_unicode

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".m4v"}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def get_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("找不到 ffmpeg，请先安装 imageio-ffmpeg") from None
        return exe


def _run(cmd: list[str], desc: str) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError(f"{desc}失败：\n" + "\n".join(tail))
    return proc


@contextmanager
def _ascii_video(path: str | Path):
    """ffmpeg / OpenCV 在 Windows 下对中文路径支持不稳定，必要时复制到 ASCII 临时目录。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"视频不存在: {p}")
    if str(p).isascii():
        yield p
        return
    td = Path(tempfile.mkdtemp(prefix="wmascii_"))
    try:
        dst = td / ("input" + p.suffix.lower())
        shutil.copy2(p, dst)
        yield dst
    finally:
        shutil.rmtree(td, ignore_errors=True)


def extract_first_frame(path: str | Path) -> np.ndarray:
    ff = get_ffmpeg()
    with _ascii_video(path) as src, tempfile.TemporaryDirectory(prefix="wmfirst_") as td:
        out = Path(td) / "first.png"
        _run(
            [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", "0", "-i", str(src), "-frames:v", "1", str(out)],
            "提取首帧",
        )
        if not out.exists():
            raise RuntimeError("提取视频首帧失败")
        return imread_unicode(str(out))


def probe_video(path: str | Path) -> dict:
    with _ascii_video(path) as src:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError("无法打开视频，请确认格式是否受支持")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    first = extract_first_frame(path)
    if fps <= 1:
        fps = 25.0
    return {
        "fps": fps,
        "frames": count,
        "duration": (count / fps) if count > 0 else 0.0,
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
    }


def _extract_audio(src: Path, dst: Path, duration: float = 0.0) -> bool:
    ff = get_ffmpeg()
    cmd = [
        ff, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src), "-vn", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
    ]
    if duration and duration > 0:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 500


def process_video(
    input_path: str | Path,
    mask: np.ndarray | None = None,
    scale: float = 1,
    conf: float = 0.25,
    expand: int = 8,
    mode: str = "fixed",
    preview_seconds: float = 0,
    target_size: tuple[int, int] | None = None,
    fit: str = "contain",
    output_path: str | Path | None = None,
    pipeline: WatermarkSRPipeline | None = None,
    progress=None,
    crf: int = 18,
    preset: str = "fast",
    keep_frames: bool = False,
) -> tuple[str, dict]:
    """处理视频。

    mode="fixed"：整段视频使用同一块遮罩（推荐，速度快、稳定）。
    mode="per_frame"：每一帧重新检测水印（适合移动水印，但慢很多）。
    preview_seconds > 0：只处理前 N 秒，用于快速预览。
    """
    src_in = Path(input_path)
    if not src_in.is_file():
        raise FileNotFoundError(f"视频不存在: {src_in}")
    if mode not in ("fixed", "per_frame"):
        raise ValueError("mode 必须是 fixed 或 per_frame")

    pipeline = pipeline or WatermarkSRPipeline()
    ff = get_ffmpeg()
    t_all = time.time()
    timings: dict[str, float] = {}
    work = Path(tempfile.mkdtemp(prefix="wmvideo_"))

    try:
        ascii_src = work / ("input" + src_in.suffix.lower())
        shutil.copy2(src_in, ascii_src)

        meta = probe_video(ascii_src)
        fps = max(1.0, float(meta["fps"]))

        if progress:
            progress(0.02, desc="提取视频帧…")
        t0 = time.time()
        in_dir = work / "frames_in"
        out_dir = work / "frames_out"
        in_dir.mkdir()
        out_dir.mkdir()
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(ascii_src)]
        if preview_seconds and float(preview_seconds) > 0:
            cmd += ["-t", f"{float(preview_seconds):.3f}"]
        cmd += ["-vsync", "0", "-compression_level", "3", str(in_dir / "%08d.png")]
        _run(cmd, "视频抽帧")
        timings["extract"] = time.time() - t0

        frames = sorted(in_dir.glob("*.png"))
        if not frames:
            raise RuntimeError("没有抽取到视频帧")
        total = len(frames)

        audio_path = work / "audio.m4a"
        audio_ok = _extract_audio(ascii_src, audio_path, float(preview_seconds or 0))

        first_bgr = cv2.imread(str(frames[0]))
        if first_bgr is None:
            raise RuntimeError("读取首帧失败")
        h, w = first_bgr.shape[:2]

        if mask is not None:
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)

        fixed_mask = mask
        detected_any = fixed_mask is not None and int(fixed_mask.max()) > 0
        if mode == "fixed" and not detected_any:
            raise WatermarkNotFound("没有可用的水印遮罩，请先在第一帧上涂抹水印区域。")
        if mode == "per_frame" and not detected_any:
            first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
            _b, _s, m0 = pipeline.detector.detect(first_rgb, conf=conf, expand=expand)
            if int(m0.max()) == 0:
                raise WatermarkNotFound("逐帧检测在首帧没有发现水印，请改用手动遮罩。")
            detected_any = True

        if progress:
            progress(0.06, desc=f"共 {total} 帧，开始逐帧处理…")

        t0 = time.time()
        actual_scale = 1.0
        for i, frame_path in enumerate(frames):
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise RuntimeError(f"读取视频帧失败: {frame_path.name}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            frame_mask = fixed_mask
            if mode == "per_frame":
                _boxes, _scores, detected = pipeline.detector.detect(rgb, conf=conf, expand=expand)
                if int(detected.max()) > 0:
                    frame_mask = detected
                    detected_any = True

            if frame_mask is not None and int(frame_mask.max()) > 0:
                cleaned = pipeline.inpainter.inpaint(rgb, frame_mask, crop_margin=160)
            else:
                cleaned = rgb

            if target_size is not None:
                cleaned, actual_scale = pipeline.upscaler.upscale(
                    cleaned, target_size=target_size, fit=fit
                )
            elif float(scale) > 1.01:
                cleaned, actual_scale = pipeline.upscaler.upscale(cleaned, float(scale))

            cv2.imwrite(
                str(out_dir / f"{i + 1:08d}.png"),
                cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR),
            )

            if progress:
                done = i + 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                eta_text = f"{eta / 60:.1f} 分钟" if eta >= 60 else f"{eta:.0f} 秒"
                progress(
                    0.06 + 0.90 * done / total,
                    desc=f"逐帧处理 {done}/{total} · 剩余约 {eta_text}",
                )
        timings["process"] = time.time() - t0

        if not detected_any:
            raise WatermarkNotFound("整段视频都没有检测到水印。")

        if progress:
            progress(0.97, desc="编码视频（含音频）…")
        t0 = time.time()
        out_mp4 = work / "output.mp4"
        encode = [
            ff, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", f"{fps:.6f}", "-i", str(out_dir / "%08d.png"),
        ]
        if audio_ok:
            encode += ["-i", str(audio_path)]
        encode += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ]
        if audio_ok:
            encode += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        encode += ["-movflags", "+faststart", str(out_mp4)]
        _run(encode, "视频编码")
        timings["encode"] = time.time() - t0

        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = Path(output_path)
        if output_path.suffix.lower() not in (".mp4", ".mov", ".mkv"):
            output_path = output_path.with_suffix(".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_mp4, output_path)

        out_frame = cv2.imread(str(sorted(out_dir.glob("*.png"))[0]))
        out_h, out_w = out_frame.shape[:2] if out_frame is not None else (0, 0)
        timings["total"] = time.time() - t_all
        info = {
            "fps": round(fps, 3),
            "frames": total,
            "duration": round(total / fps, 3),
            "input_size": [w, h],
            "output_size": [out_w, out_h],
            "scale": round(float(actual_scale), 3),
            "target_size": list(target_size) if target_size is not None else None,
            "fit": fit if target_size is not None else None,
            "audio": bool(audio_ok),
            "mode": mode,
            "output_path": str(output_path),
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }
        if progress:
            progress(1.0, desc="完成")
        return str(output_path), info

    finally:
        if not keep_frames:
            shutil.rmtree(work, ignore_errors=True)
