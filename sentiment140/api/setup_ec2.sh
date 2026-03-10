#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../sentiment140/api
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"                        # .../sentiment140
CLONE_DIR="$(dirname "$PROJECT_DIR")"                         # .../Parcial_1_Corte
VENV_DIR="$PROJECT_DIR/venv"
API_PORT=8000

echo "============================================"
echo " EC2-B Setup — Sentiment API"
echo "============================================"
echo "SCRIPT_DIR : $SCRIPT_DIR"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "CLONE_DIR  : $CLONE_DIR"

# 1. Dependencias del sistema
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

# 2. Git pull
echo ">>> git pull..."
cd "$CLONE_DIR"
git pull

# 3. Ir al directorio raíz del proyecto
cd "$PROJECT_DIR"
echo "Working dir: $(pwd)"

# 4. Crear venv solo si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando venv..."
    python3 -m venv "$VENV_DIR"
fi

# 5. Activar venv con ruta absoluta
source "$VENV_DIR/bin/activate"
echo "Python: $(which python3)"

# 6. Actualizar pip
pip install --upgrade pip

# 7. Instalar dependencias
pip install -r "$SCRIPT_DIR/requirements.txt"

# 8. Construir pipeline si no existe
if [ ! -f "$PROJECT_DIR/models/best_model.pkl" ]; then
    echo "Pipeline no encontrado, construyendo..."
    python3 "$PROJECT_DIR/build_pipeline.py"
else
    echo "Pipeline ya existe: models/best_model.pkl ✔"
fi

echo ""
echo "✅ Setup completo. Iniciando API en puerto $API_PORT..."
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "TU_IP_PUBLICA")
echo "   Swagger UI: http://$PUBLIC_IP:$API_PORT/docs"
echo ""

# 9. PYTHONPATH=api para que main.py encuentre schemas e inference
cd "$PROJECT_DIR"
PYTHONPATH="$SCRIPT_DIR" uvicorn api.main:app --host 0.0.0.0 --port $API_PORT --workers 1
