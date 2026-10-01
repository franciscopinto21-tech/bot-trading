"""
Módulo de configuración centralizada del Bot de Inversión.
Carga variables de entorno y provee configuración tipada.
"""

import os
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


class Mode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class BrokerType(Enum):
    BINANCE = "binance"
    INTERACTIVE_BROKERS = "ib"
    METATRADER5 = "mt5"


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Timeframe(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


@dataclass
class BrokerConfig:
    type: BrokerType = BrokerType.BINANCE
    binance_api_key: str = ""
    binance_secret_key: str = ""
    binance_testnet: bool = True
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""


@dataclass
class NotificationConfig:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    whatsapp_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 2.0
    max_daily_loss: float = 3.0
    max_correlation: float = 0.7
    stop_loss_atr_multiplier: float = 2.0
    take_profit_ratio: float = 1.5
    circuit_breaker_news: bool = True


@dataclass
class MLConfig:
    model_type: str = "xgboost"
    retrain_interval_hours: int = 24
    sentiment_enabled: bool = True
    sentiment_confidence_min: float = 0.6


@dataclass
class TradingConfig:
    mode: Mode = Mode.PAPER
    default_timeframe: str = "1h"
    max_positions: int = 5
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])


@dataclass
class DatabaseConfig:
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "trading_bot"
    user: str = "postgres"
    password: str = ""


@dataclass
class DashboardConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class AppConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Carga la configuración desde variables de entorno."""
        return cls(
            broker=BrokerConfig(
                type=BrokerType(os.getenv("BROKER_TYPE", "binance")),
                binance_api_key=os.getenv("BINANCE_API_KEY", ""),
                binance_secret_key=os.getenv("BINANCE_SECRET_KEY", ""),
                binance_testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
                ib_host=os.getenv("IB_HOST", "127.0.0.1"),
                ib_port=int(os.getenv("IB_PORT", "7497")),
                ib_client_id=int(os.getenv("IB_CLIENT_ID", "1")),
                mt5_login=int(os.getenv("MT5_LOGIN", "0")),
                mt5_password=os.getenv("MT5_PASSWORD", ""),
                mt5_server=os.getenv("MT5_SERVER", ""),
            ),
            notifications=NotificationConfig(
                telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
                whatsapp_enabled=os.getenv("WHATSAPP_ENABLED", "false").lower() == "true",
                twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
                twilio_whatsapp_number=os.getenv("TWILIO_WHATSAPP_NUMBER", ""),
            ),
            risk=RiskConfig(
                max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "2.0")),
                max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "3.0")),
                max_correlation=float(os.getenv("MAX_CORRELATION", "0.7")),
                stop_loss_atr_multiplier=float(os.getenv("STOP_LOSS_ATR_MULTIPLIER", "2.0")),
                take_profit_ratio=float(os.getenv("TAKE_PROFIT_RATIO", "1.5")),
                circuit_breaker_news=os.getenv("CIRCUIT_BREAKER_NEWS", "true").lower() == "true",
            ),
            ml=MLConfig(
                model_type=os.getenv("ML_MODEL_TYPE", "xgboost"),
                retrain_interval_hours=int(os.getenv("RETRAIN_INTERVAL_HOURS", "24")),
                sentiment_enabled=os.getenv("SENTIMENT_ENABLED", "true").lower() == "true",
                sentiment_confidence_min=float(os.getenv("SENTIMENT_CONFIDENCE_MIN", "0.6")),
            ),
            trading=TradingConfig(
                mode=Mode(os.getenv("MODE", "paper")),
                default_timeframe=os.getenv("DEFAULT_TIMEFRAME", "1h"),
                max_positions=int(os.getenv("MAX_POSITIONS", "5")),
                symbols=[s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")],
            ),
            database=DatabaseConfig(
                type=os.getenv("DB_TYPE", "sqlite"),
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                name=os.getenv("DB_NAME", "trading_bot"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
            ),
            dashboard=DashboardConfig(
                enabled=os.getenv("DASHBOARD_ENABLED", "false").lower() == "true",
                host=os.getenv("DASHBOARD_HOST", "0.0.0.0"),
                port=int(os.getenv("DASHBOARD_PORT", "8000")),
            ),
        )


# Singleton de configuración global
config = AppConfig.from_env()
