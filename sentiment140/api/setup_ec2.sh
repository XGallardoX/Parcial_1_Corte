#!/bin/bash
# =============================================================================
# setup_ec2.sh — EC2-B: API FastAPI de Inferencia de Sentimiento
# Ejecutar UNA SOLA VEZ al aprovisionar la instancia.
# Uso: bash setup_ec2.sh
# =============================================================================

set -e   # Detener en cualquier error

REPO_URL="https://github.com/XGallardoX/Parcial_1_Corte.git"  
PROJECT_DIR="$HOME/sentiment140"
API_PORT=8000

echo "============================================"
echo " EC2-B Setup — Sentiment API"
echo "============================================"

# 1. Instalar dependencias del sistema
sudo apt update -y
sudo apt install -y python3 python3-pip git

# 2. Clonar el repositorio
if [ -d "$PROJECT_DIR" ]; then
    echo "Repo ya existe, haciendo git pull..."
    cd "$PROJECT_DIR" && git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 3. Instalar dependencias de Python
pip3 install -r api/requirements.txt

# 4. Construir el pipeline (si best_model.pkl no viene en el repo)
#    Solo necesario si data/encoded/*.pkl están en el repo pero no models/best_model.pkl
if [ ! -f "models/best_model.pkl" ]; then
    echo "Pipeline no encontrado, construyendo..."
    python3 build_pipeline.py
else
    echo "Pipeline ya existe: models/best_model.pkl ✔"
fi

# 5. Abrir puerto 8000 en el Security Group (desde AWS Console o awscli)
# aws ec2 authorize-security-group-ingress \
#     --group-id sg-XXXXXXXX \
#     --protocol tcp --port $API_PORT --cidr 0.0.0.0/0

echo ""
echo "✅ Setup completo. Iniciando API..."
uvicorn api.main:app --host 0.0.0.0 --port $API_PORT
