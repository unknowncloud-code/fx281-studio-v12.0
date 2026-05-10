@echo off
echo ======================================
echo FX281 后端启动脚本 (CUDA GPU加速)
echo ======================================
echo.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo 正在创建虚拟环境...
    python -m venv venv
    echo 正在安装CUDA版PyTorch...
    venv\Scripts\pip.exe install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    echo 正在安装其他依赖...
    venv\Scripts\pip.exe install fastapi uvicorn python-multipart dashscope openai httpx pydantic funasr modelscope python-docx pydub numpy
)

echo.
echo 启动后端服务器 (GPU模式)...
echo 访问地址: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo.
venv\Scripts\python.exe main.py
pause
