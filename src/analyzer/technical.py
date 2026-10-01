"""
Analizador Técnico - Calcula indicadores técnicos y patrones de velas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    symbol: str
    timestamp: pd.Timestamp
    direction: int  # 1 = alcista, -1 = bajista, 0 = neutral
    strength: float  # 0 a 1
    indicators: Dict[str, float]
    patterns: List[str]
    support: float
    resistance: float
    trend: str  # bullish, bearish, sideways


class TechnicalAnalyzer:
    """
    Analiza datos OHLCV para generar señales técnicas.
    """

    def __init__(self):
        self.last_signals: Dict[str, TechnicalSignal] = {}

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> TechnicalSignal:
        """
        Analiza un DataFrame OHLCV y genera una señal técnica completa.
        """
        df = df.copy()

        # Calcular todos los indicadores
        indicators = self._calculate_indicators(df)

        # Detectar patrones de velas
        patterns = self._detect_candlestick_patterns(df)

        # Calcular soportes y resistencias
        support, resistance = self._find_support_resistance(df)

        # Determinar tendencia general
        trend = self._determine_trend(indicators)

        # Generar señal consolidada
        direction, strength = self._consolidate_signal(indicators, patterns, trend)

        signal = TechnicalSignal(
            symbol=symbol,
            timestamp=df.index[-1],
            direction=direction,
            strength=strength,
            indicators=indicators,
            patterns=patterns,
            support=support,
            resistance=resistance,
            trend=trend,
        )

        self.last_signals[symbol] = signal
        return signal

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula todos los indicadores técnicos."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        open_price = df["open"].values

        indicators = {}

        # ---- MEDIAS MÓVILES ----
        ema_9 = self._ema(close, 9)
        ema_21 = self._ema(close, 21)
        ema_50 = self._ema(close, 50)
        ema_200 = self._ema(close, 200)

        indicators["ema_9"] = ema_9[-1] if len(ema_9) > 0 else close[-1]
        indicators["ema_21"] = ema_21[-1] if len(ema_21) > 0 else close[-1]
        indicators["ema_50"] = ema_50[-1] if len(ema_50) > 0 else close[-1]
        indicators["ema_200"] = ema_200[-1] if len(ema_200) > 0 else close[-1]

        # Cruce de medias
        if len(ema_9) >= 2 and len(ema_21) >= 2:
            prev_cross = ema_9[-2] - ema_21[-2]
            curr_cross = ema_9[-1] - ema_21[-1]
            indicators["ema_cross"] = 1 if curr_cross > 0 and prev_cross <= 0 else (
                -1 if curr_cross < 0 and prev_cross >= 0 else 0
            )
        else:
            indicators["ema_cross"] = 0

        # ---- RSI ----
        indicators["rsi_14"] = self._rsi(close, 14)

        # ---- MACD ----
        macd_line, signal_line, macd_hist = self._macd(close)
        indicators["macd"] = macd_line[-1] if len(macd_line) > 0 else 0
        indicators["macd_signal"] = signal_line[-1] if len(signal_line) > 0 else 0
        indicators["macd_hist"] = macd_hist[-1] if len(macd_hist) > 0 else 0

        if len(macd_hist) >= 2:
            indicators["macd_cross"] = (
                1 if macd_hist[-1] > 0 and macd_hist[-2] <= 0 else
                -1 if macd_hist[-1] < 0 and macd_hist[-2] >= 0 else
                0
            )
        else:
            indicators["macd_cross"] = 0

        # ---- BOLLINGER BANDS ----
        bb_middle, bb_upper, bb_lower = self._bollinger_bands(close)
        indicators["bb_middle"] = bb_middle[-1]
        indicators["bb_upper"] = bb_upper[-1]
        indicators["bb_lower"] = bb_lower[-1]
        indicators["bb_width"] = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]

        # Posición del precio en las BB
        bb_range = bb_upper[-1] - bb_lower[-1]
        if bb_range > 0:
            indicators["bb_position"] = (close[-1] - bb_lower[-1]) / bb_range
        else:
            indicators["bb_position"] = 0.5

        # ---- ATR (Average True Range) ----
        atr = self._atr(high, low, close, 14)
        indicators["atr"] = atr[-1] if len(atr) > 0 else 0
        indicators["atr_pct"] = (atr[-1] / close[-1] * 100) if len(atr) > 0 and close[-1] > 0 else 0

        # ---- VOLUMEN ----
        vol_sma = pd.Series(volume).rolling(20).mean().values
        indicators["volume_ratio"] = (volume[-1] / vol_sma[-1]) if len(vol_sma) > 0 and vol_sma[-1] > 0 else 1

        # ---- ESTOCÁSTICO ----
        indicators["stoch_k"], indicators["stoch_d"] = self._stochastic(high, low, close)

        # ---- OBV (On Balance Volume) ----
        indicators["obv"] = self._obv(close, volume)

        # ---- ICHIMOKU (simplificado) ----
        tenkan, kijun, senkou_a, senkou_b = self._ichimoku(high, low, close)
        indicators["tenkan"] = tenkan[-1] if len(tenkan) > 0 else 0
        indicators["kijun"] = kijun[-1] if len(kijun) > 0 else 0
        indicators["senkou_a"] = senkou_a[-1] if len(senkou_a) > 0 else 0
        indicators["senkou_b"] = senkou_b[-1] if len(senkou_b) > 0 else 0

        # ---- WILLIAMS %R ----
        indicators["williams_r"] = self._williams_r(high, low, close)

        # ---- ADX (Average Directional Index) ----
        adx_val, di_plus, di_minus = self._adx(high, low, close)
        indicators["adx"] = adx_val[-1] if len(adx_val) > 0 else 0
        indicators["di_plus"] = di_plus[-1] if len(di_plus) > 0 else 0
        indicators["di_minus"] = di_minus[-1] if len(di_minus) > 0 else 0

        return indicators

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detecta patrones de velas."""
        patterns = []
        close = df["close"].values
        open_price = df["open"].values
        high = df["high"].values
        low = df["low"].values

        if len(df) < 3:
            return patterns

        # Últimas 3 velas
        c, o, h, l = close[-1], open_price[-1], high[-1], low[-1]
        c_1, o_1, h_1, l_1 = close[-2], open_price[-2], high[-2], low[-2]
        c_2, o_2 = close[-3], open_price[-3]

        body = abs(c - o)
        body_1 = abs(c_1 - o_1)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l

        # Doji
        if body / (h - l) < 0.1 and (h - l) > 0:
            patterns.append("doji")

        # Martillo
        if lower_shadow > 2 * body and upper_shadow < body and c > o:
            patterns.append("hammer")

        # Martillo invertido
        if upper_shadow > 2 * body and lower_shadow < body and c < o:
            patterns.append("shooting_star")

        # Engulfing alcista
        if c > o_1 and c_1 < o_1 and c > o and body > body_1:
            patterns.append("bullish_engulfing")

        # Engulfing bajista
        if c < o_1 and c_1 > o_1 and c < o and body > body_1:
            patterns.append("bearish_engulfing")

        # Tres soldados blancos
        if (c > o and c_1 > o_1 and c_2 > o_2 and
                c > c_1 > c_2 and o > o_1 > o_2):
            patterns.append("three_white_soldiers")

        # Tres cuervos negros
        if (c < o and c_1 < o_1 and c_2 < o_2 and
                c < c_1 < c_2 and o < o_1 < o_2):
            patterns.append("three_black_crows")

        return patterns

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calcula EMA."""
        if len(data) < period:
            return np.array([])
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[:period] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = data[i] * alpha + ema[i - 1] * (1 - alpha)
        return ema

    def _rsi(self, data: np.ndarray, period: int = 14) -> float:
        """Calcula RSI."""
        if len(data) < period + 1:
            return 50.0
        deltas = np.diff(data)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _macd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula MACD."""
        ema_12 = self._ema(data, 12)
        ema_26 = self._ema(data, 26)

        if len(ema_12) == 0 or len(ema_26) == 0:
            return np.array([]), np.array([]), np.array([])

        min_len = min(len(ema_12), len(ema_26))
        macd_line = ema_12[-min_len:] - ema_26[-min_len:]
        signal_line = self._ema(macd_line, 9)
        macd_hist = macd_line - signal_line

        return macd_line, signal_line, macd_hist

    def _bollinger_bands(
        self, data: np.ndarray, period: int = 20, std_mult: float = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula Bollinger Bands."""
        if len(data) < period:
            return np.array([]), np.array([]), np.array([])

        middle = pd.Series(data).rolling(period).mean().values
        std = pd.Series(data).rolling(period).std().values

        upper = middle + std_mult * std
        lower = middle - std_mult * std

        return middle, upper, lower

    def _atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Calcula ATR."""
        if len(high) < 2:
            return np.array([])

        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr

    def _stochastic(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Tuple[float, float]:
        """Calcula Estocástico."""
        if len(close) < period:
            return 50.0, 50.0

        lowest_low = pd.Series(low).rolling(period).min().values
        highest_high = pd.Series(high).rolling(period).max().values

        range_val = highest_high - lowest_low
        k = np.where(range_val > 0, (close - lowest_low) / range_val * 100, 50)

        d = pd.Series(k).rolling(3).mean().values

        return float(k[-1]), float(d[-1])

    def _obv(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calcula OBV."""
        if len(close) < 2:
            return 0.0

        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                obv += volume[i]
            elif close[i] < close[i - 1]:
                obv -= volume[i]

        return float(obv)

    def _ichimoku(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcula Ichimoku simplificado."""
        if len(high) < 52:
            return np.array([]), np.array([]), np.array([]), np.array([])

        tenkan = (pd.Series(high).rolling(9).max() + pd.Series(low).rolling(9).min()) / 2
        kijun = (pd.Series(high).rolling(26).max() + pd.Series(low).rolling(26).min()) / 2
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (pd.Series(high).rolling(52).max() + pd.Series(low).rolling(52).min()) / 2

        return (
            tenkan.values,
            kijun.values,
            senkou_a.values,
            senkou_b.values,
        )

    def _williams_r(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Calcula Williams %R."""
        if len(close) < period:
            return -50.0

        highest_high = pd.Series(high).rolling(period).max().values[-1]
        lowest_low = pd.Series(low).rolling(period).min().values[-1]

        range_val = highest_high - lowest_low
        if range_val == 0:
            return -50.0

        return float((highest_high - close[-1]) / range_val * -100)

    def _adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calcula ADX."""
        if len(high) < period + 1:
            return np.array([]), np.array([]), np.array([])

        # True Range
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Directional Movements
        up_move = np.zeros(len(high))
        down_move = np.zeros(len(high))
        for i in range(1, len(high)):
            up_move[i] = high[i] - high[i - 1]
            down_move[i] = low[i - 1] - low[i]
            if up_move[i] < 0 and down_move[i] < 0:
                up_move[i] = 0
                down_move[i] = 0
            elif up_move[i] > down_move[i]:
                down_move[i] = 0
            else:
                up_move[i] = 0

        # Suavizar
        tr_smooth = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        up_smooth = pd.Series(up_move).ewm(span=period, adjust=False).mean().values
        down_smooth = pd.Series(down_move).ewm(span=period, adjust=False).mean().values

        di_plus = np.where(tr_smooth > 0, up_smooth / tr_smooth * 100, 0)
        di_minus = np.where(tr_smooth > 0, down_smooth / tr_smooth * 100, 0)

        # DX
        dx = np.where(
            di_plus + di_minus > 0,
            abs(di_plus - di_minus) / (di_plus + di_minus) * 100,
            0,
        )

        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values

        return adx, di_plus, di_minus

    def _find_support_resistance(
        self, df: pd.DataFrame, window: int = 20
    ) -> Tuple[float, float]:
        """Encuentra niveles de soporte y resistencia."""
        if len(df) < window:
            return float(df["low"].min()), float(df["high"].max())

        # Usar puntos pivote
        recent = df.tail(window)
        resistance = float(recent["high"].max())
        support = float(recent["low"].min())

        return support, resistance

    def _determine_trend(self, indicators: Dict[str, float]) -> str:
        """Determina la tendencia general basada en indicadores."""
        bullish_signals = 0
        bearish_signals = 0

        # EMA
        if indicators.get("ema_cross", 0) == 1:
            bullish_signals += 1
        elif indicators.get("ema_cross", 0) == -1:
            bearish_signals += 1

        # Precio vs EMAs
        close = indicators.get("ema_9", 0)
        if close > indicators.get("ema_50", close):
            bullish_signals += 1
        else:
            bearish_signals += 1

        # RSI
        rsi = indicators.get("rsi_14", 50)
        if rsi > 50:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # MACD
        if indicators.get("macd_cross", 0) == 1:
            bullish_signals += 2
        elif indicators.get("macd_cross", 0) == -1:
            bearish_signals += 2

        # ADX
        adx = indicators.get("adx", 0)
        if adx > 25:
            if indicators.get("di_plus", 0) > indicators.get("di_minus", 0):
                bullish_signals += 2
            else:
                bearish_signals += 2

        # Bollinger
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos > 0.8:
            bearish_signals += 1  # Sobrecomprado
        elif bb_pos < 0.2:
            bullish_signals += 1  # Sobreventa

        if bullish_signals > bearish_signals:
            return "bullish"
        elif bearish_signals > bullish_signals:
            return "bearish"
        return "sideways"

    def _consolidate_signal(
        self,
        indicators: Dict[str, float],
        patterns: List[str],
        trend: str,
    ) -> Tuple[int, float]:
        """
        Consolida todas las señales en una dirección y fuerza.
        Returns: (direction: 1, -1, 0, strength: 0-1)
        """
        score = 0.0
        total_weight = 0.0

        # --- RSI ---
        rsi = indicators.get("rsi_14", 50)
        if rsi < 30:  # Sobreventa
            score += 2.0
        elif rsi < 40:
            score += 1.0
        elif rsi > 70:  # Sobrecompra
            score -= 2.0
        elif rsi > 60:
            score -= 1.0
        total_weight += 2.0

        # --- MACD ---
        macd_cross = indicators.get("macd_cross", 0)
        score += macd_cross * 2.0
        total_weight += 2.0

        # --- EMA Cross ---
        ema_cross = indicators.get("ema_cross", 0)
        score += ema_cross * 1.5
        total_weight += 1.5

        # --- Bollinger Position ---
        bb_pos = indicators.get("bb_position", 0.5)
        if bb_pos < 0.2:
            score += 1.5
        elif bb_pos > 0.8:
            score -= 1.5
        total_weight += 1.5

        # --- Patrones de velas ---
        bullish_patterns = ["hammer", "bullish_engulfing", "three_white_soldiers"]
        bearish_patterns = ["shooting_star", "bearish_engulfing", "three_black_crows"]

        for p in patterns:
            if p in bullish_patterns:
                score += 1.5
            elif p in bearish_patterns:
                score -= 1.5
        total_weight += 1.5

        # --- Volumen ---
        vol_ratio = indicators.get("volume_ratio", 1.0)
        if vol_ratio > 1.5 and score > 0:
            score += 0.5  # Confirmación alcista
        elif vol_ratio > 1.5 and score < 0:
            score -= 0.5  # Confirmación bajista
        total_weight += 0.5

        # --- Tendencia ---
        if trend == "bullish":
            score += 1.0
        elif trend == "bearish":
            score -= 1.0
        total_weight += 1.0

        if total_weight == 0:
            return 0, 0.0

        # Normalizar score a [-1, 1]
        normalized_score = score / total_weight

        # Determinar dirección
        if normalized_score > 0.15:
            direction = 1
        elif normalized_score < -0.15:
            direction = -1
        else:
            direction = 0

        # Fuerza de la señal (0 a 1)
        strength = min(abs(normalized_score), 1.0)

        return direction, strength
