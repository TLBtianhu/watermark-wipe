@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set PYTHONUTF8=1
set GRADIO_ANALYTICS_ENABLED=False
set YOLO_OFFLINE=True

if not exist "venv\Scripts\python.exe" (
    echo [错误] 未找到 venv，请先双击 install.bat 安装依赖。
    pause
    exit /b 1
)

if not exist "models\big-lama.pt" goto :missing_models
if not exist "models\watermark_yolo11x.pt" goto :missing_models
if not exist "models\RealESRGAN_x2plus.pth" goto :missing_models
if not exist "models\RealESRGAN_x4plus.pth" goto :missing_models
goto :models_ok

:missing_models
echo [错误] 未找到模型权重，请先运行：
echo     python download_models.py
pause
exit /b 1

:models_ok

echo.
echo   一键去水印 + 超分
echo   ----------------------------------------
echo   浏览器会自动打开；如未打开，请访问：
echo   http://127.0.0.1:7860
echo.
echo   关闭此窗口即可退出程序。
echo.

".\venv\Scripts\python.exe" app.py

echo.
echo 程序已退出。
pause
