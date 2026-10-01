"""
Tests para el Analizador Técnico.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.analyzer.technical import TechnicalAnalyzer


@pytest.fixture
def sample_ohlcv_data():
    """Crea datos OHLCV de muestra."""
    dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
    
    # Tendencia alcista con algo de ruido
    trend = np.linspace(100, 110, 200)
    noise = np.random.randn(200) * 0.5
    close = trend + noise
    
    data = {
        "open": close - np.random.rand(200) * 0.3,
        "high": close + np.random.rand(200) * 0.5,
        "low": close - np.random.rand(200) * 0.5,
        "close": close,
        "volume": np.random.rand(200) * 1000 + 500,
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


class TestTechnicalAnalyzer:
    """Pruebas del analizador técnico."""

    def setup_method(self):
        self.analyzer = TechnicalAnalyzer()

    def test_analyze_returns_signal(self, sample_ohlcv_data):
        """Verifica que el análisis retorna un TechnicalSignal."""
        signal = self.analyzer.analyze(sample_ohlcv_data, "BTC/USDT")
        
        assert signal is not None
        assert signal.symbol == "BTC/USDT"
        assert signal.direction in [-1, 0, 1]
        assert 0 <= signal.strength <= 1
        assert signal.trend in ["bullish", "bearish", "sideways"]

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Verifica que el RSI está en rango válido."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        rsi = signal.indicators.get("rsi_14", 50)
        
        assert 0 <= rsi <= 100

    def test_macd_calculation(self, sample_ohlcv_data):
        """Verifica que MACD tiene los componentes esperados."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        
        assert "macd" in signal.indicators
        assert "macd_signal" in signal.indicators
        assert "macd_hist" in signal.indicators

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Verifica que las Bollinger Bands son consistentes."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        
        bb_upper = signal.indicators.get("bb_upper", 0)
        bb_lower = signal.indicators.get("bb_lower", 0)
        bb_middle = signal.indicators.get("bb_middle", 0)
        
        assert bb_upper >= bb_middle >= bb_lower

    def test_atr_positive(self, sample_ohlcv_data):
        """Verifica que ATR es positivo."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        atr = signal.indicators.get("atr", 0)
        
        assert atr >= 0

    def test_support_resistance(self, sample_ohlcv_data):
        """Verifica que soporte es menor que resistencia."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        
        assert signal.support <= signal.resistance

    def test_patterns_detection(self, sample_ohlcv_data):
        """Verifica que se detectan patrones."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        
        assert isinstance(signal.patterns, list)

    def test_volume_ratio(self, sample_ohlcv_data):
        """Verifica ratio de volumen."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        vol_ratio = signal.indicators.get("volume_ratio", 1)
        
        assert vol_ratio > 0

    def test_ema_order(self, sample_ohlcv_data):
        """Verifica que EMAs están en orden correcto."""
        signal = self.analyzer.analyze(sample_ohlcv_data)
        
        ema_9 = signal.indicators.get("ema_9", 0)
        ema_21 = signal.indicators.get("ema_21", 0)
        ema_50 = signal.indicators.get("ema_50", 0)
        
        # En tendencia alcista, EMA9 > EMA21 > EMA50
        if signal.trend == "bullish":
            assert ema_9 >= ema_21
