# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：Windows 便携版（onedir）。

用法：
    .\venv\Scripts\pyinstaller.exe --noconfirm --clean WatermarkWipe.spec

输出：
    dist\WatermarkWipe\WatermarkWipe.exe
"""
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []


def add(pkg: str) -> None:
    try:
        d, b, h = collect_all(pkg)
        datas.extend(d)
        binaries.extend(b)
        hiddenimports.extend(h)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] collect_all({pkg}) skipped: {exc}")


for pkg in [
    "gradio",
    "gradio_client",
    "safehttpx",
    "hf_gradio",
    "groovy",
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "ultralytics",
    "spandrel",
    "imageio_ffmpeg",
    "torchvision",
]:
    add(pkg)

# 模型和示例一起打进 _internal，离线可用
datas += [
    ("models", "models"),
    ("samples", "samples"),
]

hiddenimports += [
    "torch.jit",
    "torch.jit._recursive",
    "torch.serialization",
    "torchvision.ops",
    "torchvision.transforms",
    "multipart",
    "python_multipart",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "spandrel.architectures",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter.test",
        "matplotlib.tests",
        "numpy.tests",
        "pandas.tests",
        "torch.test",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WatermarkWipe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WatermarkWipe",
)
