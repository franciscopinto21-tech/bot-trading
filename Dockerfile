# Dockerfile para el Bot de Inversión Inteligente
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Crear directorios necesarios
RUN mkdir -p logs data models

# Exponer puerto para dashboard
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "src.main", "--mode", "paper"]
