@echo off
echo ======================================
echo FX281 Studio - Backend Launcher
echo ======================================
echo.
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    echo [2/3] Installing CUDA PyTorch...
    venv\Scripts\pip.exe install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    echo [3/3] Installing dependencies...
    venv\Scripts\pip.exe install fastapi uvicorn python-multipart openai httpx pydantic funasr modelscope python-docx pydub numpy
)

echo.
echo Starting backend server...
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Frontend: http://localhost:5173 (run "npm run dev" in project root)
echo.
venv\Scripts\python.exe main.py
pause
