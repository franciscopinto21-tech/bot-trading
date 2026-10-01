#!/bin/bash
# Script de configuración inicial del Bot de Inversión Inteligente

echo "=========================================="
echo "  Configuración del Bot de Inversión"
echo "=========================================="

# Verificar Python
PYTHON_VERSION=$(python3 --version 2>&1)
if [ $? -ne 0 ]; then
    echo "❌ Python 3 no encontrado. Instálalo primero."
    exit 1
fi
echo "✅ $PYTHON_VERSION"

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
fi

# Activar entorno virtual
source venv/bin/activate

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Crear directorios
echo "📁 Creando directorios..."
mkdir -p logs data models

# Configurar .env
if [ ! -f ".env" ]; then
    echo "📝 Creando archivo .env desde plantilla..."
    cp env_example.txt .env
    echo "⚠️  EDITA .env CON TUS CREDENCIALES ANTES DE EJECUTAR"
fi

echo ""
echo "=========================================="
echo "  ✅ CONFIGURACIÓN COMPLETADA"
echo "=========================================="
echo ""
echo "Pasos siguientes:"
echo "1. Edita el archivo .env con tus credenciales"
echo "2. Ejecuta: python -m src.main --mode backtest --symbol BTC/USDT"
echo "3. Para paper trading: python -m src.main --mode paper"
echo "=========================================="
