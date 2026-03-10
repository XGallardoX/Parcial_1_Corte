#!/bin/bash
# start_api.sh — Arrancar la API (ya instalada)
# Uso: bash start_api.sh
cd "$HOME/sentiment140"
git pull                              # actualizar desde git
uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
