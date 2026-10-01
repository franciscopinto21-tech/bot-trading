#!/bin/bash
# Script para ejecutar el bot en modo live
# Uso: ./scripts/run_live.sh BTC/USDT moderate

SYMBOL=${1:-BTC/USDT}
RISK=${2:-moderate}

echo "=========================================="
echo "  Iniciando Bot en MODO LIVE"
echo "=========================================="
echo "Símbolo: $SYMBOL"
echo "Riesgo: $RISK"
echo "⚠️  ASEGÚRATE DE TENER CONFIGURADO .env"
echo "=========================================="

cd "$(dirname "$0")/.."

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ejecutar bot
python -m src.main --mode live --symbol "$SYMBOL" --risk "$RISK"
