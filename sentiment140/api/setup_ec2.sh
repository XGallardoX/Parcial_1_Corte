#!/bin/bash
set -e

REPO_URL="https://github.com/XGallardoX/Parcial_1_Corte.git"
CLONE_DIR="$HOME/NLP/Parcial_1_Corte"
PROJECT_DIR="$CLONE_DIR/sentiment140"
API_PORT=8000

echo "============================================"
echo " EC2-B Setup — Sentiment API"
echo "============================================"

# 1. Dependencias del sistema
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git

# 2. Clonar o actualizar repo
mkdir -p "$HOME/NLP"
if [ -d "$CLONE_DIR/.git" ]; then
    echo "Repo ya existe, haciendo git pull..."
    cd "$CLONE_DIR"
    git pull
else
    echo "Clonando repo..."
    git clone "$REPO_URL" "$CLONE_DIR"
fi

# 3. Ir al directorio del proyecto
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"

# 4. Crear venv si no existe
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 5. Activar venv
source venv/bin/activate

# 6. Actualizar pip
pip install --upgrade pip

# 7. Crear requirements.txt si no existe en el repo
if [ ! -f "api/requirements.txt" ]; then
    echo "requirements.txt no encontrado, creando..."
    cat > api/requirements.txt << 'REQS'
fastapi==0.115.0
uvicorn[standard]==0.30.6
scikit-learn==1.5.2
joblib==1.4.2
numpy==1.26.4
matplotlib==3.9.2
pydantic==2.9.2
REQS
fi

# 8. Instalar dependencias
pip install -r api/requirements.txt

# 9. Construir pipeline si no existe
if [ ! -f "models/best_model.pkl" ]; then
    echo "Pipeline no encontrado, construyendo..."
    python3 build_pipeline.py
else
    echo "Pipeline ya existe: models/best_model.pkl ✔"
fi

echo ""
echo "✅ Setup completo. Iniciando API en puerto $API_PORT..."
echo "   URL: http://$(curl -s ifconfig.me):$API_PORT/docs"
echo ""

# 10. Arrancar uvicorn desde PROJECT_DIR (no desde api/)
uvicorn api.main:app --host 0.0.0.0 --port $API_PORT --workers 1
