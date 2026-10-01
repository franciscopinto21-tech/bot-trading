"""
Analizador Fundamental - Evalúa datos macroeconómicos y fundamentales.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import logging

logger = logging.getLogger(__name__)


@dataclass
class FundamentalResult:
    symbol: str
    timestamp: datetime
    score: float  # -1 a +1
    metrics: Dict[str, float]
    macro_indicators: Dict[str, float]
    sector_health: str
    overall_classification: str  # strong_buy, buy, hold, sell, strong_sell


class FundamentalAnalyzer:
    """
    Analiza fundamentos macroeconómicos y datos fundamentales de activos.
    """

    # Rangos "saludables" para indicadores macro
    HEALTHY_RANGES = {
        "gdp_growth": (1.5, 4.0),  # %
        "inflation": (1.5, 3.0),  # %
        "unemployment": (3.0, 6.0),  # %
        "interest_rate": (1.0, 5.0),  # %
        "consumer_confidence": (90.0, 120.0),
        "manufacturing_pmi": (45.0, 60.0),
        "services_pmi": (45.0, 60.0),
    }

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.alpha_vantage_key = alpha_vantage_key
        self.last_results: Dict[str, FundamentalResult] = {}
        self._macro_cache: Dict = {}
        self._last_macro_update: Optional[datetime] = None

    def analyze(
        self,
        symbol: str = "",
        sector: str = "",
        macro_data: Optional[Dict] = None,
    ) -> FundamentalResult:
        """
        Analiza fundamentos de un activo.
        """
        # Obtener o usar datos macro proporcionados
        if macro_data is None:
            macro_data = self._get_macro_data()

        # Obtener datos del sector
        sector_data = self._get_sector_data(sector) if sector else {}

        # Calcular score fundamental
        metrics = {}
        score = 0.0
        total_weight = 0.0

        # 1. Evaluar indicadores macroeconómicos
        macro_score, macro_metrics = self._evaluate_macro_indicators(macro_data)
        metrics.update(macro_metrics)
        score += macro_score * 3.0
        total_weight += 3.0

        # 2. Evaluar salud del sector
        if sector_data:
            sector_score = self._evaluate_sector(sector_data)
            metrics["sector_score"] = sector_score
            score += sector_score * 2.0
            total_weight += 2.0

        # 3. Evaluar condiciones generales de mercado
        market_score = self._evaluate_market_conditions(macro_data)
        metrics["market_conditions"] = market_score
        score += market_score * 1.5
        total_weight += 1.5

        # 4. Riesgo geopolítico (proxy simplificado)
        geo_score = self._evaluate_geopolitical_risk()
        metrics["geopolitical_risk"] = geo_score
        score += geo_score * 0.5
        total_weight += 0.5

        # Score final normalizado
        if total_weight > 0:
            final_score = score / total_weight
        else:
            final_score = 0.0

        final_score = max(-1.0, min(1.0, final_score))

        # Clasificación
        if final_score > 0.5:
            classification = "strong_buy"
        elif final_score > 0.15:
            classification = "buy"
        elif final_score > -0.15:
            classification = "hold"
        elif final_score > -0.5:
            classification = "sell"
        else:
            classification = "strong_sell"

        # Salud del sector
        sector_health = (
            "strong" if metrics.get("sector_score", 0) > 0.3
            else "weak" if metrics.get("sector_score", 0) < -0.3
            else "neutral"
        )

        result = FundamentalResult(
            symbol=symbol,
            timestamp=datetime.now(),
            score=final_score,
            metrics=metrics,
            macro_indicators=macro_data,
            sector_health=sector_health,
            overall_classification=classification,
        )

        self.last_results[symbol] = result
        return result

    def _get_macro_data(self) -> Dict:
        """Obtiene datos macroeconómicos actuales."""
        # Intentar desde API si está configurada
        if self.alpha_vantage_key:
            try:
                return self._fetch_macro_from_api()
            except Exception as e:
                logger.warning(f"Error fetching macro data: {e}")

        # Cache o datos por defecto
        if self._macro_cache and self._last_macro_update:
            if datetime.now() - self._last_macro_update < timedelta(hours=24):
                return self._macro_cache

        # Datos simulados realistas para desarrollo
        macro_data = {
            "gdp_growth": 2.5,  # %
            "inflation": 2.8,  # %
            "unemployment": 3.7,  # %
            "interest_rate": 4.5,  # %
            "consumer_confidence": 104.5,
            "manufacturing_pmi": 52.3,
            "services_pmi": 54.1,
            "housing_starts": 1.45,  # millones
            "retail_sales_growth": 2.1,  # %
            "industrial_production": 1.8,  # %
        }

        self._macro_cache = macro_data
        self._last_macro_update = datetime.now()
        return macro_data

    def _fetch_macro_from_api(self) -> Dict:
        """Obtiene datos macro de Alpha Vantage."""
        # Implementar según disponibilidad de endpoints
        # Por ahora retorna datos simulados
        return self._get_macro_data()

    def _evaluate_macro_indicators(self, macro_data: Dict) -> Tuple[float, Dict]:
        """
        Evalúa indicadores macroeconómicos.
        Returns: (score, dict of individual scores)
        """
        scores = {}
        total = 0.0
        count = 0

        for indicator, value in macro_data.items():
            if indicator in self.HEALTHY_RANGES:
                low, high = self.HEALTHY_RANGES[indicator]
                if low <= value <= high:
                    indicator_score = 1.0  # Saludable
                elif value < low:
                    # Por debajo del rango saludable
                    deviation = (low - value) / low
                    indicator_score = max(-1.0, -deviation)
                else:
                    # Por encima del rango saludable
                    deviation = (value - high) / high
                    indicator_score = max(-1.0, -deviation)

                scores[f"macro_{indicator}"] = indicator_score
                total += indicator_score
                count += 1

        avg_score = total / count if count > 0 else 0.0
        return avg_score, scores

    def _get_sector_data(self, sector: str) -> Dict:
        """Obtiene datos de un sector específico."""
        sector_data = {
            "technology": {"growth": 8.5, "valuation": 25.0, "sentiment": 0.6},
            "finance": {"growth": 3.2, "valuation": 12.0, "sentiment": 0.3},
            "healthcare": {"growth": 6.1, "valuation": 18.0, "sentiment": 0.5},
            "energy": {"growth": -1.5, "valuation": 8.0, "sentiment": -0.2},
            "consumer": {"growth": 2.8, "valuation": 15.0, "sentiment": 0.2},
        }
        return sector_data.get(sector.lower(), {})

    def _evaluate_sector(self, sector_data: Dict) -> float:
        """Evalúa la salud de un sector."""
        score = 0.0
        total_weight = 0.0

        # Crecimiento
        growth = sector_data.get("growth", 0)
        if growth > 5:
            score += 1.0
        elif growth > 2:
            score += 0.5
        elif growth > 0:
            score += 0.2
        elif growth > -2:
            score -= 0.3
        else:
            score -= 1.0
        total_weight += 1.0

        # Valuación (P/E ratio implícito)
        valuation = sector_data.get("valuation", 15)
        if 10 <= valuation <= 20:
            score += 0.5  # Rango razonable
        elif valuation > 30:
            score -= 0.5  # Sobrevalorado
        elif valuation < 8:
            score -= 0.3  # Puede indicar problemas
        total_weight += 0.5

        # Sentimiento del sector
        sentiment = sector_data.get("sentiment", 0)
        score += sentiment * 0.8
        total_weight += 0.8

        return score / total_weight if total_weight > 0 else 0.0

    def _evaluate_market_conditions(self, macro_data: Dict) -> float:
        """Evalúa condiciones generales del mercado."""
        score = 0.0

        # Combinación de indicadores
        gdp = macro_data.get("gdp_growth", 2.0)
        inflation = macro_data.get("inflation", 2.0)
        unemployment = macro_data.get("unemployment", 4.0)

        # Crecimiento económico + baja inflación + bajo desempleo = Goldilocks
        if gdp >= 2.0 and inflation <= 3.0 and unemployment <= 5.0:
            score = 0.8
        elif gdp >= 1.0 and inflation <= 4.0 and unemployment <= 6.0:
            score = 0.3
        elif gdp < 0 or inflation > 6.0 or unemployment > 8.0:
            score = -0.8
        else:
            score = 0.0

        return score

    def _evaluate_geopolitical_risk(self) -> float:
        """
        Evalúa riesgo geopolítico (simplificado).
        Idealmente se conectaría a API de riesgo geopolítico.
        """
        # Por defecto, riesgo neutral
        return 0.0

    def analyze_portfolio_fundamentals(
        self, symbols: List[str], sectors: List[str]
    ) -> Dict[str, FundamentalResult]:
        """Analiza fundamentos de un portafolio completo."""
        results = {}
        macro_data = self._get_macro_data()

        for symbol, sector in zip(symbols, sectors):
            result = self.analyze(symbol=symbol, sector=sector, macro_data=macro_data)
            results[symbol] = result

        return results
