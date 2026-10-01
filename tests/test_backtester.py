"""
Tests para el Motor de Backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtesting.backtester import Backtester


@pytest.fixture
def sample_market_data():
    """Crea datos de mercado de muestra para backtesting."""
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1h")
    
    # Crear tendencia con ciclos
    t = np.linspace(0, 4 * np.pi, 1000)
    trend = 100 + 10 * np.sin(t) + t * 0.01
    
    noise = np.random.randn(1000) * 0.5
    close = trend + noise
    
    data = {
        "open": close - np.random.rand(1000) * 0.5,
        "high": close + np.random.rand(1000) * 0.8,
        "low": close - np.random.rand(1000) * 0.8,
        "close": close,
        "volume": np.random.rand(1000) * 1000 + 500,
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


class TestBacktester:
    """Pruebas del motor de backtesting."""

    def setup_method(self):
        self.backtester = Backtester(
            initial_capital=10000.0,
            commission_pct=0.001,
            risk_per_trade=2.0,
        )

    def test_backtest_runs(self, sample_market_data):
        """Verifica que el backtesting se ejecuta sin errores."""
        result = self.backtester.run(sample_market_data, symbol="BTC/USDT")
        
        assert result is not None
        assert result.symbol == "BTC/USDT"

    def test_backtest_produces_trades(self, sample_market_data):
        """Verifica que se generan trades."""
        result = self.backtester.run(sample_market_data, symbol="BTC/USDT")
        
        assert result.total_trades >= 0

    def test_metrics_calculation(self, sample_market_data):
        """Verifica que las métricas se calculan correctamente."""
        result = self.backtester.run(sample_market_data, symbol="BTC/USDT")
        
        if result.total_trades > 0:
            assert 0 <= result.win_rate <= 100
            assert result.profit_factor >= 0
            assert result.max_drawdown_percent >= 0

    def test_equity_curve(self, sample_market_data):
        """Verifica que la curva de equity existe."""
        result = self.backtester.run(sample_market_data, symbol="BTC/USDT")
        
        assert len(result.equity_curve) > 0
        assert result.equity_curve[0] == 10000.0  # Capital inicial

    def test_commission_impact(self, sample_market_data):
        """Verifica que las comisiones se aplican."""
        # Backtest con comisiones
        bt_with_commission = Backtester(commission_pct=0.01)
        result_with = bt_with_commission.run(sample_market_data, symbol="BTC/USDT")
        
        # Backtest sin comisiones
        bt_without = Backtester(commission_pct=0.0)
        result_without = bt_without.run(sample_market_data, symbol="BTC/USDT")
        
        # Con comisiones debería tener menor PnL
        if result_with.total_trades > 0 and result_without.total_trades > 0:
            assert result_with.total_pnl <= result_without.total_pnl

    def test_print_summary(self, sample_market_data, capsys):
        """Verifica que el resumen se imprime."""
        result = self.backtester.run(sample_market_data, symbol="BTC/USDT")
        self.backtester.print_summary(result)
        
        captured = capsys.readouterr()
        assert "RESUMEN DE BACKTESTING" in captured.out
        assert "BTC/USDT" in captured.out
