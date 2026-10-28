"""一键去水印 + 超分 —— 本地 Web 界面。"""
from __future__ import annotations

import os
import socket
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np

from core.paths import ROOT
from core.pipeline import OUTPUT_DIR, WatermarkNotFound, WatermarkSRPipeline
from core.sizing import FIT_CHOICES, OUTPUT_CHOICES, resolve_output
from core.utils import imread_unicode, imwrite_unicode, mask_from_editor, mask_to_rgba, overlay_mask
from core.video import extract_first_frame, probe_video, process_video

FIT_LABEL = {value: label for label, value in FIT_CHOICES}
pipeline = WatermarkSRPipeline()

CSS = """
.gradio-container { max-width: 1500px !important; width: 100% !important; margin: 0 auto !important; }
#title { text-align: center; margin-bottom: 0.2rem; }
#subtitle { text-align: center; color: #9aa4b2; margin-top: 0; margin-bottom: 1rem; }
#run-btn, #batch-btn, #video-btn { min-height: 52px; font-size: 1.05rem; font-weight: 700; }
#status-box { min-height: 64px; }
#batch-status, #video-status { min-height: 80px; }
footer { display: none !important; }
"""
THEME = gr.themes.Soft(primary_hue="red")


def _detect_and_build_editor(image: np.ndarray, conf: float, expand: int):
    boxes, scores, mask, view = pipeline.detect_preview(image, conf=conf, expand=int(expand))
    value = {
        "background": image,
        "layers": [mask_to_rgba(mask)],
        "composite": view,
    }
    if len(boxes) == 0:
        status = (
            "⚠️ **未检测到水印**。请用右侧画笔涂抹水印区域（红色），"
            "或把「检测灵敏度」调低后点「重新检测」。"
        )
    else:
        score_text = "、".join(f"{float(s):.2f}" for s in scores[:8])
        ratio = float((mask > 0).mean())
        warn = " 检测区域较大，建议检查遮罩是否正确。" if ratio > 0.5 else ""
        status = (
            f"✅ **检测到 {len(boxes)} 处水印**（置信度 {score_text}）。"
            f"红色区域会被修复，可直接点「一键去水印 + 超分」；"
            f"如有遗漏可用画笔补充，画多了用橡皮擦掉。{warn}"
        )
    return value, status


def on_upload(image, conf, expand):
    if image is None:
        return None, "请上传图片。"
    image = np.asarray(image)
    t0 = time.time()
    try:
        value, status = _detect_and_build_editor(image, conf, expand)
        return value, status + f"\n\n<sub>检测耗时 {time.time() - t0:.2f}s</sub>"
    except Exception as exc:  # noqa: BLE001
        return {"background": image, "layers": [], "composite": image}, f"❌ 检测失败：{exc}"


def on_detect(image, conf, expand):
    if image is None:
        return None, "请先上传图片。"
    return on_upload(image, conf, expand)


def on_clear(image):
    if image is None:
        return None, "请先上传图片。"
    image = np.asarray(image)
    return {"background": image, "layers": [], "composite": image}, "遮罩已清空，请用画笔涂抹要修复的区域。"


def _save_result(result: np.ndarray, tag: str = "result") -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    imwrite_unicode(str(path), result)
    return path


def on_run(
    image,
    editor_value,
    size_choice,
    fit_mode,
    target_w,
    target_h,
    conf,
    expand,
    progress=gr.Progress(),
):
    if image is None:
        gr.Warning("请先上传图片")
        return None, None, "请先上传图片。"

    try:
        scale, target = resolve_output(size_choice, target_w, target_h)
    except ValueError as exc:
        gr.Warning(str(exc))
        return None, None, f"⚠️ {exc}"

    image = np.asarray(image)
    mask = mask_from_editor(editor_value, image.shape[:2])

    t0 = time.time()
    try:
        result, info = pipeline.process(
            image,
            mask=mask,
            scale=float(scale),
            conf=float(conf),
            expand=int(expand),
            target_size=target,
            fit=str(fit_mode),
            progress=progress,
        )
    except WatermarkNotFound as exc:
        gr.Warning(str(exc))
        return None, None, f"⚠️ {exc}"
    except Exception as exc:  # noqa: BLE001
        gr.Warning(f"处理失败：{exc}")
        return None, None, f"❌ 处理失败：{exc}"

    out_path = _save_result(result)
    timings = info.get("timings", {})
    w, h = info["output_size"]
    if target is not None:
        size_desc = f"{target[0]}×{target[1]}（{FIT_LABEL.get(str(fit_mode), fit_mode)}）"
    else:
        size_desc = f"{info['scale']:g}×"
    boxes_text = f"自动检测 {info['boxes']} 处" if info.get("auto_detected") else "遮罩区域"
    status = (
        f"✅ **完成** · 修复 {boxes_text} · 超分 {size_desc} · "
        f"输出 {w}×{h} · 总耗时 {time.time() - t0:.1f}s\n\n"
        f"<sub>检测 {timings.get('detect', 0)}s · 修复 {timings.get('inpaint', 0)}s · "
        f"超分 {timings.get('upscale', 0)}s · 已保存到 output/</sub>"
    )
    return result, str(out_path), status


def on_batch(
    files,
    size_choice,
    fit_mode,
    target_w,
    target_h,
    conf,
    expand,
    progress=gr.Progress(),
):
    if not files:
        gr.Warning("请先选择图片")
        return None, "请先选择要批量处理的图片。"

    try:
        scale, target = resolve_output(size_choice, target_w, target_h)
    except ValueError as exc:
        return None, f"⚠️ {exc}"

    paths: list[str] = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "path"):
            paths.append(getattr(f, "path"))
        elif isinstance(f, dict):
            paths.append(f.get("path") or f.get("name"))
        else:
            paths.append(str(f))
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        return None, "没有找到可读取的图片文件。"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"batch_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_clean_sr" if (float(scale) > 1 or target is not None) else "_clean"

    saved: list[Path] = []
    errors: list[str] = []
    total = len(paths)
    for i, path in enumerate(paths):
        progress(i / total, desc=f"处理中 {i + 1}/{total}")
        try:
            image = imread_unicode(path)
            result, _info = pipeline.process(
                image,
                mask=None,
                scale=float(scale),
                conf=float(conf),
                expand=int(expand),
                target_size=target,
                fit=str(fit_mode),
            )
            stem = Path(path).stem
            out_path = out_dir / f"{stem}{suffix}.png"
            if out_path.exists():
                out_path = out_dir / f"{stem}_{i + 1}{suffix}.png"
            imwrite_unicode(str(out_path), result)
            saved.append(out_path)
        except WatermarkNotFound:
            errors.append(f"{Path(path).name}：没有检测到水印")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{Path(path).name}：{exc}")

    if not saved:
        msg = "❌ 没有成功处理的图片。\n\n" + "\n".join(f"- {e}" for e in errors[:10])
        return None, msg

    progress(0.98, desc="打包 ZIP…")
    zip_path = OUTPUT_DIR / f"batch_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in saved:
            zf.write(p, arcname=p.name)

    if target is not None:
        size_desc = f"{target[0]}×{target[1]}（{FIT_LABEL.get(str(fit_mode), fit_mode)}）"
    else:
        size_desc = f"{float(scale):g}×"
    status = (
        f"✅ **批量完成 {len(saved)}/{total} 张** · 超分 {size_desc} · "
        f"结果已打包。\n\n"
        f"<sub>图片保存在 output/{out_dir.name}/，ZIP 在 output/ 下。</sub>"
    )
    if errors:
        status += "\n\n**跳过/失败：**\n" + "\n".join(f"- {e}" for e in errors[:10])
    return str(zip_path), status


def _video_meta_text(meta: dict) -> str:
    frames = int(meta.get("frames") or 0)
    frames_text = f"{frames} 帧" if frames > 0 else "帧数未知"
    return (
        f"**{meta['width']}×{meta['height']}** · {meta['fps']:.2f} fps · "
        f"{float(meta.get('duration') or 0):.1f} 秒 · {frames_text}"
    )


def on_video_upload(video_path, conf, expand):
    if not video_path:
        return None, "等待上传视频…", "请上传视频。"
    try:
        first = extract_first_frame(video_path)
        meta = probe_video(video_path)
        boxes, scores, mask, view = pipeline.detect_preview(
            first, conf=float(conf), expand=int(expand)
        )
        value = {
            "background": first,
            "layers": [mask_to_rgba(mask)],
            "composite": view,
        }
        info = _video_meta_text(meta)
        if len(boxes) == 0:
            status = (
                "⚠️ **首帧未检测到水印**。请在右侧用画笔涂抹水印区域，"
                "或降低检测灵敏度后点「重新检测首帧」。"
            )
        else:
            score_text = "、".join(f"{float(s):.2f}" for s in scores[:8])
            status = (
                f"✅ **首帧检测到 {len(boxes)} 处水印**（置信度 {score_text}）。"
                "红色区域会应用到整段视频；确认无误后可直接开始处理。"
            )
        if float(meta.get("duration") or 0) > 60:
            status += "\n\n⏱️ 视频较长，建议先把「只处理前 N 秒」设为 5–10 秒预览效果。"
        return value, info, status
    except Exception as exc:  # noqa: BLE001
        return None, "读取视频失败", f"❌ {exc}"


def on_video_run(
    video_path,
    editor_value,
    size_choice,
    fit_mode,
    target_w,
    target_h,
    mode,
    conf,
    expand,
    preview_seconds,
    progress=gr.Progress(),
):
    if not video_path:
        gr.Warning("请先上传视频")
        return None, None, "请先上传视频。"

    try:
        scale, target = resolve_output(size_choice, target_w, target_h)
    except ValueError as exc:
        gr.Warning(str(exc))
        return None, None, f"⚠️ {exc}"

    t0 = time.time()
    try:
        first = extract_first_frame(video_path)
        mask = mask_from_editor(editor_value, first.shape[:2])
        if int(mask.max()) == 0:
            _boxes, _scores, mask = pipeline.detect(first, conf=float(conf), expand=int(expand))
        if int(mask.max()) == 0:
            gr.Warning("没有可用的水印遮罩，请先用画笔涂抹")
            return None, None, "⚠️ 没有可用的水印遮罩。请用画笔涂抹水印区域，或降低检测灵敏度后重新检测。"

        out_path, info = process_video(
            video_path,
            mask=mask,
            scale=float(scale),
            conf=float(conf),
            expand=int(expand),
            mode=str(mode),
            preview_seconds=float(preview_seconds or 0),
            target_size=target,
            fit=str(fit_mode),
            pipeline=pipeline,
            progress=progress,
        )
    except WatermarkNotFound as exc:
        gr.Warning(str(exc))
        return None, None, f"⚠️ {exc}"
    except Exception as exc:  # noqa: BLE001
        gr.Warning(f"视频处理失败：{exc}")
        return None, None, f"❌ 视频处理失败：{exc}"

    timings = info.get("timings", {})
    w, h = info["output_size"]
    in_w, in_h = info["input_size"]
    if target is not None:
        size_desc = f"{target[0]}×{target[1]}（{FIT_LABEL.get(str(fit_mode), fit_mode)}）"
    else:
        size_desc = f"{info['scale']:g}×"
    status = (
        f"✅ **视频处理完成** · {info['frames']} 帧 / {info['duration']:g} 秒 · "
        f"{in_w}×{in_h} → {w}×{h} · 超分 {size_desc} · "
        f"{'音频保留' if info['audio'] else '无音频'}\n\n"
        f"<sub>抽帧 {timings.get('extract', 0)}s · 逐帧处理 {timings.get('process', 0)}s · "
        f"编码 {timings.get('encode', 0)}s · 总耗时 {time.time() - t0:.1f}s</sub>"
    )
    return out_path, out_path, status


def on_open_folder():
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(OUTPUT_DIR))  # type: ignore[attr-defined]
        return "📂 已打开输出文件夹。"
    except Exception as exc:  # noqa: BLE001
        return f"打开失败：{exc}"


def find_port(preferred: int = 7860) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="一键去水印 + 超分", fill_width=True) as demo:
        gr.Markdown("# 🪄 一键去水印 + 超分", elem_id="title")
        gr.Markdown(
            "上传图片或视频后自动识别水印 → LaMa 智能修复 → Real-ESRGAN 2×/4× 超分。"
            "图片可手动涂抹修正；视频保留原声，支持先处理前几秒预览。",
            elem_id="subtitle",
        )

        with gr.Tabs():
            with gr.Tab("单张处理"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image_in = gr.Image(
                            type="numpy",
                            label="1. 上传图片",
                            height=460,
                            sources=["upload", "clipboard"],
                        )
                        with gr.Row():
                            conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="检测灵敏度")
                            expand = gr.Slider(0, 40, value=8, step=1, label="遮罩外扩 px")
                        size_choice = gr.Dropdown(
                            choices=OUTPUT_CHOICES,
                            value="2",
                            label="2. 输出尺寸",
                            info="可选 1080p / 2K / 4K，或自定义分辨率",
                        )
                        with gr.Row():
                            target_w = gr.Number(value=1920, label="自定义宽", precision=0, minimum=64, maximum=8192)
                            target_h = gr.Number(value=1080, label="自定义高", precision=0, minimum=64, maximum=8192)
                        fit_mode = gr.Dropdown(
                            choices=FIT_CHOICES,
                            value="contain",
                            label="目标尺寸适配",
                        )
                        with gr.Row():
                            btn_detect = gr.Button("重新检测", variant="secondary")
                            btn_clear = gr.Button("清空遮罩", variant="secondary")
                        btn_run = gr.Button("🚀 一键去水印 + 超分", variant="primary", elem_id="run-btn")

                    with gr.Column(scale=5):
                        editor = gr.ImageEditor(
                            type="numpy",
                            label="水印遮罩（红色 = 会被修复；可画笔补充 / 橡皮擦修正）",
                            height=460,
                            canvas_size=(760, 540),
                            fixed_canvas=False,
                            brush=gr.Brush(colors=["#ff2d2d"], default_size=22, color_mode="fixed"),
                            eraser=gr.Eraser(default_size=30),
                            layers=True,
                            format="png",
                            sources=["upload", "clipboard"],
                        )
                        status = gr.Markdown("等待上传图片…", elem_id="status-box")

                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=6):
                        image_out = gr.Image(type="numpy", label="处理结果", height=560, interactive=False)
                    with gr.Column(scale=4):
                        file_out = gr.File(label="下载结果（PNG 无损）", interactive=False)
                        btn_open = gr.Button("📂 打开输出文件夹")
                        status_run = gr.Markdown("结果会保存到项目的 output/ 目录。")

                gr.Examples(
                    examples=[
                        [str(ROOT / "samples" / "example1.jpg")],
                        [str(ROOT / "samples" / "example2.jpg")],
                        [str(ROOT / "samples" / "example3.jpg")],
                    ],
                    inputs=image_in,
                    label="示例图片",
                )

            with gr.Tab("批量处理"):
                gr.Markdown(
                    "一次上传多张图片，按顺序去水印并超分，完成后打包成 ZIP 下载。"
                    "单张失败不会中断整批，失败原因会列在下方。"
                )
                batch_files = gr.Files(
                    label="上传图片（可多选，支持 JPG / PNG / WebP / BMP）",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    height=220,
                )
                with gr.Row():
                    batch_size_choice = gr.Dropdown(
                        choices=OUTPUT_CHOICES,
                        value="2",
                        label="输出尺寸",
                    )
                    batch_fit = gr.Dropdown(
                        choices=FIT_CHOICES,
                        value="contain",
                        label="目标尺寸适配",
                    )
                with gr.Row():
                    batch_w = gr.Number(value=1920, label="宽", precision=0, minimum=64, maximum=8192)
                    batch_h = gr.Number(value=1080, label="高", precision=0, minimum=64, maximum=8192)
                    batch_conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="检测灵敏度")
                    batch_expand = gr.Slider(0, 40, value=8, step=1, label="遮罩外扩 px")
                with gr.Row():
                    btn_batch = gr.Button("🚀 开始批量处理", variant="primary", elem_id="batch-btn")
                    btn_batch_open = gr.Button("📂 打开输出文件夹")
                batch_status = gr.Markdown("等待上传图片…", elem_id="batch-status")
                batch_zip = gr.File(label="下载结果 ZIP", interactive=False)

            with gr.Tab("视频处理"):
                gr.Markdown(
                    "上传视频 → 自动识别首帧水印 → 可手动修正遮罩 → 逐帧修复，可选 2×/4× 超分。"
                    "固定机位水印效果最好；移动水印请选「逐帧检测」。"
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        video_in = gr.Video(
                            label="1. 上传视频",
                            sources=["upload"],
                            height=330,
                            include_audio=True,
                        )
                        video_info = gr.Markdown("等待上传视频…")
                        video_size_choice = gr.Dropdown(
                            choices=OUTPUT_CHOICES,
                            value="1",
                            label="输出尺寸",
                            info="视频建议先用 1× 检查去水印效果；可选 1080p/2K/4K。",
                        )
                        with gr.Row():
                            video_w = gr.Number(value=1920, label="宽", precision=0, minimum=64, maximum=8192)
                            video_h = gr.Number(value=1080, label="高", precision=0, minimum=64, maximum=8192)
                        video_fit = gr.Dropdown(
                            choices=FIT_CHOICES,
                            value="contain",
                            label="目标尺寸适配",
                        )
                        video_mode = gr.Radio(
                            choices=[("固定遮罩（推荐）", "fixed"), ("逐帧检测（慢）", "per_frame")],
                            value="fixed",
                            label="水印模式",
                        )
                        with gr.Row():
                            video_conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label="检测灵敏度")
                            video_expand = gr.Slider(0, 40, value=8, step=1, label="遮罩外扩 px")
                        video_preview = gr.Slider(
                            0,
                            60,
                            value=0,
                            step=1,
                            label="只处理前 N 秒（0 = 完整视频）",
                            info="建议先用 5–10 秒预览，确认效果后再处理整段。",
                        )
                        with gr.Row():
                            btn_video_detect = gr.Button("重新检测首帧", variant="secondary")
                            btn_video_open = gr.Button("📂 打开输出文件夹")
                        btn_video = gr.Button("🚀 开始处理视频", variant="primary", elem_id="video-btn")

                    with gr.Column(scale=5):
                        video_editor = gr.ImageEditor(
                            type="numpy",
                            label="首帧水印遮罩（红色 = 整段视频都会被修复）",
                            height=460,
                            canvas_size=(760, 430),
                            fixed_canvas=False,
                            brush=gr.Brush(colors=["#ff2d2d"], default_size=22, color_mode="fixed"),
                            eraser=gr.Eraser(default_size=30),
                            layers=True,
                            format="png",
                            sources=["upload", "clipboard"],
                        )
                        video_status = gr.Markdown("等待上传视频…", elem_id="video-status")

                with gr.Row():
                    with gr.Column(scale=6):
                        video_out = gr.Video(label="结果视频", height=420, interactive=False)
                    with gr.Column(scale=4):
                        video_file = gr.File(label="下载视频（MP4）", interactive=False)
                        gr.Markdown(
                            "<sub>固定遮罩：整段视频共用首帧遮罩，速度快，适合台标/角标。"
                            "逐帧检测：每帧重新找水印，适合移动水印，但慢很多。</sub>"
                        )

        gr.Markdown(
            "<sub>模型：YOLO11x 水印检测 · LaMa 图像修复 · Real-ESRGAN 超分。"
            "请仅处理你拥有版权或已获授权的图片/视频。</sub>"
        )

        image_in.change(on_upload, inputs=[image_in, conf, expand], outputs=[editor, status])
        btn_detect.click(on_detect, inputs=[image_in, conf, expand], outputs=[editor, status])
        btn_clear.click(on_clear, inputs=[image_in], outputs=[editor, status])
        btn_run.click(
            on_run,
            inputs=[image_in, editor, size_choice, fit_mode, target_w, target_h, conf, expand],
            outputs=[image_out, file_out, status_run],
        )
        btn_open.click(on_open_folder, outputs=[status_run])
        btn_batch.click(
            on_batch,
            inputs=[batch_files, batch_size_choice, batch_fit, batch_w, batch_h, batch_conf, batch_expand],
            outputs=[batch_zip, batch_status],
        )
        btn_batch_open.click(on_open_folder, outputs=[batch_status])
        video_in.change(
            on_video_upload,
            inputs=[video_in, video_conf, video_expand],
            outputs=[video_editor, video_info, video_status],
        )
        btn_video_detect.click(
            on_video_upload,
            inputs=[video_in, video_conf, video_expand],
            outputs=[video_editor, video_info, video_status],
        )
        btn_video.click(
            on_video_run,
            inputs=[
                video_in,
                video_editor,
                video_size_choice,
                video_fit,
                video_w,
                video_h,
                video_mode,
                video_conf,
                video_expand,
                video_preview,
            ],
            outputs=[video_out, video_file, video_status],
        )
        btn_video_open.click(on_open_folder, outputs=[video_status])

    return demo


if __name__ == "__main__":
    port = find_port(int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    print(f"\n  一键去水印 + 超分  ->  http://127.0.0.1:{port}\n")
    ui = build_ui()
    ui.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=os.environ.get("NO_BROWSER") != "1",
        show_error=True,
        theme=THEME,
        css=CSS,
    )
