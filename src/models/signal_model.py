"""
Motor de Señales - Fusiona análisis técnico, sentimiento y fundamental.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: str  # LONG, SHORT, NEUTRAL
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: float  # 0 a 1
    reason: str
    market_type: str
    timestamp: datetime
    signal_id: str = ""
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    fundamental_score: float = 0.0
    risk_reward: float = 0.0
    risk_percentage: float = 0.0


class SignalEngine:
    """
    Motor que fusiona las señales de las tres capas de análisis
    y decide si generar una señal de trading.
    """

    def __init__(
        self,
        technical_weight: float = 0.40,
        sentiment_weight: float = 0.30,
        fundamental_weight: float = 0.30,
        min_confidence: float = 0.65,
        min_layers_agreement: int = 2,
    ):
        self.weights = {
            "technical": technical_weight,
            "sentiment": sentiment_weight,
            "fundamental": fundamental_weight,
        }
        self.min_confidence = min_confidence
        self.min_layers_agreement = min_layers_agreement
        self.last_signals: Dict[str, TradeSignal] = {}
        self.signals_history: List[TradeSignal] = []

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        technical_result: Dict,
        sentiment_result: Optional[Dict] = None,
        fundamental_result: Optional[Dict] = None,
        risk_assessment: Optional[Dict] = None,
    ) -> Optional[TradeSignal]:
        """
        Evalúa todas las capas y genera una señal si es viable.
        """
        # 1. Extraer scores de cada capa
        tech_score = self._extract_technical_score(technical_result)
        sent_score = self._extract_sentiment_score(sentiment_result) if sentiment_result else 0.0
        fund_score = self._extract_fundamental_score(fundamental_result) if fundamental_result else 0.0

        # 2. Determinar dirección de cada capa
        tech_direction, tech_strength = self._get_direction_and_strength(tech_score)
        sent_direction, sent_strength = self._get_direction_and_strength(sent_score)
        fund_direction, fund_strength = self._get_direction_and_strength(fund_score)

        # 3. Verificar acuerdo entre capas
        directions = [tech_direction, sent_direction, fund_direction]
        layers_available = sum(1 for d in directions if d != 0)
        
        if layers_available < self.min_layers_agreement:
            logger.info(f"Capas insuficientes para {symbol}: {layers_available} disponibles")
            return None

        # Contar acuerdo
        longs = sum(1 for d in directions if d == 1)
        shorts = sum(1 for d in directions if d == -1)

        if longs == 0 and shorts == 0:
            logger.info(f"Sin consenso claro para {symbol}")
            return None

        # 4. Determinar dirección final
        if longs >= self.min_layers_agreement:
            final_direction = "LONG"
            direction_int = 1
        elif shorts >= self.min_layers_agreement:
            final_direction = "SHORT"
            direction_int = -1
        else:
            logger.info(f"Sin acuerdo suficiente para {symbol}")
            return None

        # 5. Calcular score compuesto
        composite_score = self._calculate_composite_score(
            tech_score, sent_score, fund_score, direction_int
        )

        # 6. Calcular confianza
        confidence = self._calculate_confidence(
            composite_score,
            tech_strength,
            sent_strength,
            fund_strength,
            longs if final_direction == "LONG" else shorts,
            layers_available,
        )

        # 7. Verificar umbral mínimo de confianza
        if confidence < self.min_confidence:
            logger.info(f"Confianza insuficiente para {symbol}: {confidence:.2%}")
            return None

        # 8. Obtener SL/TP y tamaño del risk assessment si está disponible
        stop_loss = 0.0
        take_profit = 0.0
        position_size = 0.0
        risk_reward = 0.0
        risk_percentage = 0.0

        if risk_assessment:
            stop_loss = risk_assessment.get("suggested_stop_loss", 0)
            take_profit = risk_assessment.get("suggested_take_profit", 0)
            position_size = risk_assessment.get("suggested_position_size", 0)
            risk_reward = risk_assessment.get("risk_reward_ratio", 0)
            risk_percentage = risk_assessment.get("risk_percentage", 0)
        else:
            # Calcular SL/TP básicos si no hay risk manager
            atr_price = current_price * 0.02  # 2% asumiendo ATR
            if direction_int == 1:
                stop_loss = current_price - atr_price * 2
                take_profit = current_price + atr_price * 3
            else:
                stop_loss = current_price + atr_price * 2
                take_profit = current_price - atr_price * 3
            position_size = 0.01  # Tamaño mínimo

        # 9. Generar razón legible
        reason = self._generate_reason(
            final_direction,
            tech_score,
            sent_score,
            fund_score,
            technical_result,
            sentiment_result,
        )

        # 10. Crear la señal
        signal = TradeSignal(
            symbol=symbol,
            direction=final_direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            confidence=confidence,
            reason=reason,
            market_type=self._detect_market_type(symbol),
            timestamp=datetime.now(),
            signal_id=f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{symbol.replace('/', '')}",
            technical_score=tech_score,
            sentiment_score=sent_score,
            fundamental_score=fund_score,
            risk_reward=risk_reward,
            risk_percentage=risk_percentage,
        )

        self.last_signals[symbol] = signal
        self.signals_history.append(signal)

        logger.info(
            f"Señal generada: {final_direction} {symbol} "
            f"| Confianza: {confidence:.2%} "
            f"| Técnico: {tech_score:.2f} "
            f"| Sentimiento: {sent_score:.2f} "
            f"| Fundamental: {fund_score:.2f}"
        )

        return signal

    def _extract_technical_score(self, technical_result: Dict) -> float:
        """Extrae score normalizado del análisis técnico."""
        if isinstance(technical_result, dict):
            direction = technical_result.get("direction", 0)
            strength = technical_result.get("strength", 0)
            return direction * strength
        return 0.0

    def _extract_sentiment_score(self, sentiment_result: Dict) -> float:
        """Extrae score normalizado del análisis de sentimiento."""
        if isinstance(sentiment_result, dict):
            return sentiment_result.get("score", 0.0)
        return 0.0

    def _extract_fundamental_score(self, fundamental_result: Dict) -> float:
        """Extrae score normalizado del análisis fundamental."""
        if isinstance(fundamental_result, dict):
            return fundamental_result.get("score", 0.0)
        return 0.0

    def _get_direction_and_strength(self, score: float) -> Tuple[int, float]:
        """Obtiene dirección y fuerza de un score."""
        if score > 0.15:
            return 1, min(abs(score), 1.0)
        elif score < -0.15:
            return -1, min(abs(score), 1.0)
        return 0, 0.0

    def _calculate_composite_score(
        self,
        tech_score: float,
        sent_score: float,
        fund_score: float,
        direction: int,
    ) -> float:
        """Calcula score compuesto ponderado."""
        composite = (
            tech_score * self.weights["technical"]
            + sent_score * self.weights["sentiment"]
            + fund_score * self.weights["fundamental"]
        )
        return composite * direction

    def _calculate_confidence(
        self,
        composite_score: float,
        tech_strength: float,
        sent_strength: float,
        fund_strength: float,
        layers_agreeing: int,
        total_layers: int,
    ) -> float:
        """Calcula nivel de confianza de la señal."""
        # Factor 1: Fuerza del score compuesto (0-0.5)
        score_factor = min(abs(composite_score) * 2, 0.5)

        # Factor 2: Acuerdo entre capas (0-0.3)
        if total_layers > 0:
            agreement_factor = (layers_agreeing / total_layers) * 0.3
        else:
            agreement_factor = 0.0

        # Factor 3: Fuerza individual promedio (0-0.2)
        avg_strength = (tech_strength + sent_strength + fund_strength) / 3
        strength_factor = avg_strength * 0.2

        # Sumar factores
        confidence = score_factor + agreement_factor + strength_factor

        return min(confidence, 1.0)

    def _generate_reason(
        self,
        direction: str,
        tech_score: float,
        sent_score: float,
        fund_score: float,
        technical_result: Dict,
        sentiment_result: Optional[Dict] = None,
    ) -> str:
        """Genera una razón legible para la señal."""
        reasons = []

        # Razón técnica
        if isinstance(technical_result, dict):
            trend = technical_result.get("trend", "unknown")
            patterns = technical_result.get("patterns", [])
            rsi = technical_result.get("indicators", {}).get("rsi_14", 50)

            if tech_score > 0.3:
                reasons.append(f"Tendencia {trend} fuerte (RSI: {rsi:.0f})")
            elif tech_score > 0.1:
                reasons.append(f"Tendencia {trend}")

            if patterns:
                pattern_names = [p.replace("_", " ") for p in patterns[:2]]
                reasons.append(f"Patrón: {', '.join(pattern_names)}")

        # Razón de sentimiento
        if sentiment_result:
            sentiment_class = sentiment_result.get("classification", "neutral")
            sent_conf = sentiment_result.get("confidence", 0)
            if sent_conf > 0.5:
                reasons.append(f"Sentimiento {sentiment_class} en noticias")
                entities = sentiment_result.get("key_entities", [])
                if entities:
                    reasons.append(f"Entidades relevantes: {', '.join(entities[:3])}")

        # Razón fundamental
        if fund_score > 0.3:
            reasons.append("Fundamentos macro sólidos")
        elif fund_score < -0.3:
            reasons.append("Señales macro de precaución")

        if not reasons:
            reasons.append("Análisis multicapa completado")

        return " | ".join(reasons)

    def _detect_market_type(self, symbol: str) -> str:
        """Detecta el tipo de mercado basado en el símbolo."""
        symbol_upper = symbol.upper()
        
        if any(c in symbol_upper for c in ["BTC", "ETH", "SOL", "XRP"]):
            return "Crypto"
        elif any(c in symbol_upper for c in ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]):
            return "Forex"
        elif any(c in symbol_upper for c in ["SPY", "QQQ", "DIA", "IWM"]):
            return "ETF"
        elif symbol_upper.endswith("USDT") or symbol_upper.endswith("USD"):
            return "Crypto"
        else:
            return "Stock/Other"

    def get_signal_summary(self, symbol: str) -> Optional[Dict]:
        """Obtiene resumen de la última señal para un símbolo."""
        signal = self.last_signals.get(symbol)
        if signal:
            return asdict(signal)
        return None

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Obtiene las señales más recientes."""
        recent = self.signals_history[-limit:]
        return [asdict(s) for s in recent]

    def get_performance_stats(self) -> Dict:
        """Obtiene estadísticas de rendimiento de las señales."""
        if not self.signals_history:
            return {"total_signals": 0}

        total = len(self.signals_history)
        longs = sum(1 for s in self.signals_history if s.direction == "LONG")
        shorts = sum(1 for s in self.signals_history if s.direction == "SHORT")

        avg_confidence = np.mean([s.confidence for s in self.signals_history])
        high_confidence = sum(1 for s in self.signals_history if s.confidence >= 0.8)

        return {
            "total_signals": total,
            "long_signals": longs,
            "short_signals": shorts,
            "long_pct": (longs / total * 100) if total > 0 else 0,
            "avg_confidence": avg_confidence,
            "high_confidence_signals": high_confidence,
            "high_confidence_pct": (high_confidence / total * 100) if total > 0 else 0,
        }
