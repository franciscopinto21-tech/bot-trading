# 🤖 Bot de Inversión Inteligente

Bot automatizado de trading con análisis multicapa (técnico, sentimiento de noticias y fundamental) que envía alertas y ejecuta operaciones de forma automática.

## 🚀 Características

- 📊 **Análisis Técnico**: Indicadores, patrones de velas, soportes/resistencias
- 📰 **Análisis de Sentimiento**: NLP sobre noticias financieras (FinBERT)
- 📈 **Análisis Fundamental**: Datos macroeconómicos y fundamentales
- ⚠️ **Gestión de Riesgo**: Stop Loss dinámico, límites diarios, control de correlación
- 📱 **Notificaciones**: Telegram y WhatsApp con señales detalladas
- 🤖 **Ejecución Automática**: Conexión con brókers mediante API
- 🧠 **Aprendizaje Continuo**: Reentrenamiento periódico de modelos ML
- 📉 **Backtesting**: Walk-forward testing con métricas realistas

## 📋 Requisitos

- Python 3.11+
- Cuenta en bróker (Binance, Interactive Brokers, etc.)
- API Keys del bróker
- Token de Telegram Bot
- (Opcional) API Key de NewsAPI o similar

## 🔧 Instalación

```bash
# Clonar el repositorio
cd bot-de-inversion

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys y configuración
```

## 🏃 Ejecución

```bash
# Modo backtesting
python src/main.py --mode backtest --symbol BTC/USDT --days 365

# Modo paper trading (demo)
python src/main.py --mode paper --symbol BTC/USDT

# Modo live trading
python src/main.py --mode live --symbol BTC/USDT --risk moderate
```

## ⚙️ Configuración

Edita el archivo `.env` con tus credenciales:

```env
# Bróker
BROKER_TYPE=binance
BINANCE_API_KEY=tu_api_key
BINANCE_SECRET_KEY=tu_secret_key

# Notificaciones
TELEGRAM_BOT_TOKEN=tu_token
TELEGRAM_CHAT_ID=tu_chat_id
WHATSAPP_API_KEY=tu_whatsapp_key

# APIs de noticias
NEWS_API_KEY=tu_news_api_key

# Base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot

# Configuración de riesgo
MAX_RISK_PER_TRADE=2.0
MAX_DAILY_LOSS=3.0
STOP_LOSS_ATR_MULTIPLIER=2.0
TAKE_PROFIT_RATIO=1.5
```

## 📁 Estructura del proyecto

```
bot-de-inversion/
├── src/
│   ├── main.py              # Punto de entrada
│   ├── config.py            # Configuración centralizada
│   ├── broker/              # Conexión con brókers
│   ├── analyzer/            # Motores de análisis
│   ├── risk/                # Gestión de riesgo
│   ├── notifications/       # Sistema de notificaciones
│   ├── models/              # Modelos ML y señales
│   ├── backtesting/         # Motor de backtesting
│   └── database/            # Gestión de datos
├── tests/                   # Tests unitarios
├── data/                    # Datos históricos
├── requirements.txt         # Dependencias
└── .env                     # Variables de entorno
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/ -v

# Test específico
pytest tests/test_risk.py -v
```

## ⚠️ Disclaimer

**Este software es solo para fines educativos y de investigación.** Las inversiones conllevan riesgo de pérdida. No se garantiza rentabilidad alguna. Consulte a un asesor financiero profesional antes de invertir dinero real.

## 📄 Licencia

MIT License
