#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../api
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"                        # .../sentiment140
CLONE_DIR="$(dirname "$PROJECT_DIR")"                         # .../Parcial_1_Corte
VENV_DIR="$PROJECT_DIR/venv"
API_PORT=8000

echo "============================================"
echo " EC2-B Setup — Sentiment API"
echo "============================================"
echo "PROJECT_DIR : $PROJECT_DIR"

# 1. Sistema
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

# 2. Git pull
cd "$CLONE_DIR"
git pull

# 3. venv
cd "$PROJECT_DIR"
if [ ! -d "$VENV_DIR" ]; then python3 -m venv "$VENV_DIR"; fi
source "$VENV_DIR/bin/activate"

# 4. Dependencias
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. Reconstruir pipeline (borra el viejo para que siempre use el módulo nuevo)
echo "Reconstruyendo pipeline..."
rm -f "$PROJECT_DIR/models/best_model.pkl"
PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/build_pipeline.py"

echo ""
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "TU_IP")
echo "✅ Iniciando API en http://$PUBLIC_IP:$API_PORT/docs"
echo ""

# 6. Arrancar uvicorn
cd "$PROJECT_DIR"
PYTHONPATH="$SCRIPT_DIR" uvicorn api.main:app --host 0.0.0.0 --port $API_PORT --workers 1
