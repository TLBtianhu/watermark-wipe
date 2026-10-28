@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM 如需使用本地代理，取消下面两行注释并按实际端口修改
REM set HTTP_PROXY=http://127.0.0.1:7897
REM set HTTPS_PROXY=http://127.0.0.1:7897

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11，并勾选 Add Python to PATH。
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo 创建虚拟环境...
    python -m venv venv
)

echo.
echo 升级 pip...
".\venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 安装 PyTorch 2.9.1 + CUDA 12.8（约 2.9GB，请耐心等待）...
".\venv\Scripts\python.exe" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

echo.
echo 安装其余依赖...
".\venv\Scripts\python.exe" -m pip install numpy opencv-python pillow gradio==6.31.0 ultralytics==8.4.191 spandrel==0.4.2 imageio-ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 安装完成。请先运行 python download_models.py 下载模型，再双击 start.bat 启动。
pause
