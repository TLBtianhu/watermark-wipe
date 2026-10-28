@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [错误] 请先运行 install.bat 安装依赖。
    pause
    exit /b 1
)

if not exist "models\big-lama.pt" (
    echo [错误] 请先运行 python download_models.py 下载模型。
    pause
    exit /b 1
)

echo.
echo 安装 PyInstaller...
".\venv\Scripts\python.exe" -m pip install pyinstaller pyinstaller-hooks-contrib -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo 开始打包（首次约 10-20 分钟）...
".\venv\Scripts\pyinstaller.exe" --noconfirm --clean WatermarkWipe.spec

echo.
echo 打包完成，输出目录: dist\WatermarkWipe\
echo 运行 dist\WatermarkWipe\WatermarkWipe.exe 测试。
pause
