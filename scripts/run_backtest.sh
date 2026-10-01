#!/bin/bash
# Script para ejecutar backtesting
# Uso: ./scripts/run_backtest.sh BTC/USDT 365

SYMBOL=${1:-BTC/USDT}
DAYS=${2:-365}

echo "=========================================="
echo "  Iniciando Backtesting"
echo "=========================================="
echo "Símbolo: $SYMBOL"
echo "Días: $DAYS"
echo "=========================================="

cd "$(dirname "$0")/.."

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ejecutar backtesting
python -m src.main --mode backtest --symbol "$SYMBOL" --days "$DAYS"
