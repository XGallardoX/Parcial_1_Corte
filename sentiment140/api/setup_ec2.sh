#!/bin/bash
set -e

REPO_URL="https://github.com/XGallardoX/Parcial_1_Corte.git"
PROJECT_DIR="$HOME/sentiment140"
API_PORT=8000

echo "============================================"
echo " EC2-B Setup — Sentiment API"
echo "============================================"

# 1. Dependencias del sistema
sudo apt update -y
sudo apt install -y python3 python3-pip python3-venv git

# 2. Clonar repo
if [ -d "$PROJECT_DIR" ]; then
    echo "Repo ya existe, haciendo git pull..."
    cd "$PROJECT_DIR" && git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 3. Crear entorno virtual
python3 -m venv venv

# 4. Activar entorno
source venv/bin/activate

# 5. Actualizar pip
pip install --upgrade pip

# 6. Instalar dependencias
pip install -r api/requirements.txt

# 7. Construir pipeline si no existe
if [ ! -f "models/best_model.pkl" ]; then
    echo "Pipeline no encontrado, construyendo..."
    python3 build_pipeline.py
else
    echo "Pipeline ya existe ✔"
fi

echo ""
echo "✅ Setup completo. Iniciando API..."

uvicorn api.main:app --host 0.0.0.0 --port $API_PORT