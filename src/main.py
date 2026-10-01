"""
Bot de Inversion Inteligente - Punto de entrada principal.
"""

import argparse
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
    level="INFO",
)
logger.add(
    "logs/trading_bot_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from src.config import AppConfig, Mode
from src.analyzer.technical import TechnicalAnalyzer
from src.analyzer.sentiment import SentimentAnalyzer
from src.analyzer.fundamental import FundamentalAnalyzer
from src.models.signal_model import SignalEngine
from src.risk.risk_manager import RiskManager
from src.notifications.telegram_notifier import TelegramNotifier
from src.notifications.whatsapp_notifier import WhatsAppNotifier
from src.backtesting.backtester import Backtester
from src.database.db_manager import DatabaseManager


class TradingBot:
    """
    Bot de Inversion Inteligente - Clase principal.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env()
        self.running = False
        self._init_modules()

    def _init_modules(self):
        logger.info("Inicializando modulos del Bot de Inversion...")
        self.db = DatabaseManager()
        self.broker = None
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer(
            news_api_key="",
            confidence_threshold=self.config.ml.sentiment_confidence_min,
        )
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.signal_engine = SignalEngine(
            min_confidence=self.config.ml.sentiment_confidence_min,
        )
        self.risk_manager = RiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=self.config.risk.max_risk_per_trade,
            max_daily_loss=self.config.risk.max_daily_loss,
            max_correlation=self.config.risk.max_correlation,
            circuit_breaker_news=self.config.risk.circuit_breaker_news,
        )
        self.telegram = TelegramNotifier(
            bot_token=self.config.notifications.telegram_bot_token,
            chat_id=self.config.notifications.telegram_chat_id,
        )
        self.whatsapp = WhatsAppNotifier(
            account_sid=self.config.notifications.twilio_account_sid,
            auth_token=self.config.notifications.twilio_auth_token,
            from_number=self.config.notifications.twilio_whatsapp_number,
        )
        self.backtester = Backtester()
        logger.info("Todos los modulos inicializados correctamente")

    def _generate_synthetic_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        periods = days * 24
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")
        t = np.linspace(0, 6 * np.pi, periods)
        base_price = {"BTC/USDT": 50000, "ETH/USDT": 3000, "SOL/USDT": 100}.get(symbol, 100)
        trend = base_price + base_price * 0.3 * np.sin(t) + base_price * 0.05 * np.linspace(0, 1, periods)
        noise = np.random.randn(periods) * base_price * 0.02
        close = np.maximum(trend + noise, base_price * 0.5)
        df = pd.DataFrame({
            "open": close * (1 - np.abs(np.random.randn(periods)) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(periods)) * 0.01),
            "low": close * (1 - np.abs(np.random.randn(periods)) * 0.01),
            "close": close,
            "volume": np.random.rand(periods) * base_price * 10 + base_price,
        }, index=dates)
        return df

    def run_backtest_sync(self, symbol: str, days: int = 365):
        logger.info(f"Iniciando backtesting para {symbol} ({days} dias)")
        df = self._generate_synthetic_data(symbol, days)
        logger.info(f"Datos generados: {len(df)} velas horarias")
        result = self.backtester.run(df, symbol=symbol, use_walk_forward=False)
        self.backtester.print_summary(result)
        metrics = {
            "date": datetime.now().date(),
            "symbol": symbol,
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "max_drawdown": result.max_drawdown_percent,
            "profit_factor": result.profit_factor,
            "sharpe_ratio": result.sharpe_ratio,
            "capital": self.backtester.initial_capital + result.total_pnl,
        }
        self.db.save_performance_metrics(metrics)
        return result

    def get_status(self) -> dict:
        risk_stats = self.risk_manager.get_statistics()
        return {
            "mode": self.config.trading.mode.value,
            "running": self.running,
            "symbols": self.config.trading.symbols,
            "timeframe": self.config.trading.default_timeframe,
            "risk_stats": risk_stats,
            "open_positions": len(self.risk_manager.open_positions),
            "total_pnl": risk_stats.get("total_pnl", 0),
            "win_rate": risk_stats.get("win_rate", 0),
        }


def main():
    parser = argparse.ArgumentParser(description="Bot de Inversion Inteligente")
    parser.add_argument("--mode", choices=["live", "paper", "backtest"], default="backtest", help="Modo de operacion")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Simbolo a operar")
    parser.add_argument("--days", type=int, default=365, help="Dias de datos para backtesting")
    parser.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate", help="Nivel de riesgo")

    args = parser.parse_args()
    config = AppConfig.from_env()
    config.trading.mode = Mode(args.mode)
    bot = TradingBot(config)

    if args.mode == "backtest":
        bot.run_backtest_sync(args.symbol, args.days)
    else:
        print(f"Modo {args.mode} requiere conexion a broker (Binance API keys)")
        print()
        print("Para probar sin broker, usa:")
        print(f"  python -m src.main --mode backtest --symbol BTC/USDT --days 365")
        print()
        print("Para ejecutar tests unitarios:")
        print(f"  python -m pytest tests/ -v")


if __name__ == "__main__":
    main()
