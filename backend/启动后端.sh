#!/usr/bin/env bash
# ============================================
# FX281 Studio - Backend Launcher (macOS)
# ============================================
set -e

echo "======================================"
echo "FX281 Studio - Backend Launcher (macOS)"
echo "======================================"
echo ""

# 切到脚本所在目录
cd "$(dirname "$0")"

# 首次运行：创建 venv 并安装依赖
if [ ! -f "venv/bin/python" ]; then
    echo "[1/3] 创建虚拟环境..."
    python3 -m venv venv
    echo "[2/3] 安装 PyTorch（Apple Silicon 自带 MPS 加速）..."
    ./venv/bin/python -m pip install --upgrade pip
    ./venv/bin/pip install torch==2.6.0 torchaudio==2.6.0
    echo "[3/3] 安装其余依赖..."
    ./venv/bin/pip install -r requirements.txt
fi

echo ""
echo "Starting backend server..."
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Frontend: http://localhost:5173 (在项目根目录运行 \"npm run dev\")"
echo ""

./venv/bin/python main.py
