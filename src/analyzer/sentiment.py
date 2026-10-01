"""
Analizador de Sentimiento - Procesa noticias financieras usando NLP.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import requests
import logging

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    symbol: str
    timestamp: datetime
    score: float  # -1 (muy negativo) a +1 (muy positivo)
    confidence: float  # 0 a 1
    sources_analyzed: int
    headlines: List[str]
    key_entities: List[str]
    classification: str  # positive, negative, neutral


class SentimentAnalyzer:
    """
    Analiza el sentimiento de noticias financieras.
    Usa un enfoque basado en reglas + diccionario financiero cuando no hay APIs.
    """

    # Palabras positivas financieras
    POSITIVE_WORDS = {
        # Resultados
        "beat", "exceed", "surge", "soar", "boom", "rally", "breakthrough",
        "record", "growth", "profit", "revenue", "earnings", "dividend",
        "upgrade", "outperform", "bullish", "confidence", "expansion",
        "innovation", "partnership", "acquisition", "merger", "ipo",
        "positive", "strong", "robust", "momentum", "optimistic",
        "recovery", "rebound", "upside", "breakout", "accumulate",
        "overweight", "buy", "outperform", "market-beating",
    }

    # Palabras negativas financieras
    NEGATIVE_WORDS = {
        # Problemas
        "decline", "drop", "fall", "crash", "plunge", "slump", "loss",
        "debt", "bankruptcy", "fraud", "scandal", "investigation",
        "downgrade", "underperform", "bearish", "uncertainty", "recession",
        "inflation", "rate_hike", "layoff", "firing", "lawsuit",
        "negative", "weak", "declining", "pessimistic", "volatile",
        "sell-off", "correction", "bear_market", "default", "insolvent",
        "sell", "underweight", "reduce", "caution", "warning",
    }

    # Palabras de intensidad
    STRONG_INTENSIFIERS = {
        "massive", "extreme", "severe", "dramatic", "historic", "unprecedented",
        "catastrophic", "enormous", "tremendous", "colossal", "record-breaking",
    }

    MODERATE_INTENSIFIERS = {
        "significant", "substantial", "considerable", "notable", "marked",
        "moderate", "slight", "mild", "minor",
    }

    # Negaciones
    NEGATIONS = {
        "not", "no", "never", "neither", "nor", "none", "nobody",
        "nothing", "nowhere", "hardly", "barely", "scarcely",
        "doesn't", "don't", "didn't", "won't", "wouldn't",
        "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    }

    # Sectores y entidades comunes en finanzas
    SECTOR_KEYWORDS = {
        "tech": ["apple", "microsoft", "google", "amazon", "meta", "nvidia", "tesla"],
        "finance": ["jpmorgan", "goldman", "bank", "citigroup", "wells_fargo"],
        "crypto": ["bitcoin", "ethereum", "binance", "coinbase", "crypto"],
        "energy": ["exxon", "shell", "chevron", "bp", "renewable", "oil"],
    }

    def __init__(
        self,
        news_api_key: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        self.news_api_key = news_api_key
        self.finnhub_api_key = finnhub_api_key
        self.confidence_threshold = confidence_threshold
        self.last_results: Dict[str, SentimentResult] = {}

    def analyze_headline(self, headline: str) -> Tuple[float, float]:
        """
        Analiza el sentimiento de un titular usando enfoque basado en reglas.
        Returns: (score: -1 a 1, confidence: 0 a 1)
        """
        headline_lower = headline.lower()
        words = re.findall(r'\w+', headline_lower)

        score = 0.0
        positive_count = 0
        negative_count = 0
        intensity_multiplier = 1.0

        # Buscar patrones de negación en las últimas palabras
        negation_active = False
        negation_distance = 0

        for i, word in enumerate(words):
            if word in self.NEGATIONS:
                negation_active = True
                negation_distance = 0
                continue

            if negation_active:
                negation_distance += 1
                if negation_distance > 3:  # La negación solo afecta 3 palabras
                    negation_active = False

            # Determinar intensidad
            intensity = 1.0
            if word in self.STRONG_INTENSIFIERS:
                intensity = 2.0
            elif word in self.MODERATE_INTENSIFIERS:
                intensity = 1.5

            # Evaluar palabra
            if word in self.POSITIVE_WORDS:
                if negation_active:
                    score -= 0.5 * intensity
                    negative_count += 1
                else:
                    score += 0.5 * intensity
                    positive_count += 1
            elif word in self.NEGATIVE_WORDS:
                if negation_active:
                    score += 0.5 * intensity
                    positive_count += 1
                else:
                    score -= 0.5 * intensity
                    negative_count += 1

        # Calcular score normalizado
        total_matches = positive_count + negative_count
        if total_matches > 0:
            score = (positive_count - negative_count) / total_matches
            confidence = min(total_matches / 10.0, 1.0)  # Más palabras = más confianza
        else:
            score = 0.0
            confidence = 0.1

        # Buscar signos de exclamación o mayúsculas (intensidad)
        if "!" in headline:
            intensity_multiplier = 1.3
        if headline.isupper() and len(headline) > 20:
            intensity_multiplier = 1.2

        score = max(-1.0, min(1.0, score * intensity_multiplier))

        return score, confidence

    def extract_entities(self, headline: str) -> List[str]:
        """Extrae entidades relevantes del titular."""
        entities = []
        headline_lower = headline.lower()

        # Buscar símbolos de ticker ($AAPL)
        tickers = re.findall(r'\$([A-Z]{1,5})', headline)
        entities.extend(tickers)

        # Buscar empresas conocidas
        for sector, companies in self.SECTOR_KEYWORDS.items():
            for company in companies:
                if company in headline_lower:
                    entities.append(company)

        # Buscar menciones de criptos
        crypto_patterns = r'\b(BTC|ETH|SOL|XRP|ADA|DOT|LINK|MATIC|AVAX)\b'
        crypto_matches = re.findall(crypto_patterns, headline)
        entities.extend(crypto_matches)

        return list(set(entities))

    def analyze(self, symbol: str = "", headlines: Optional[List[str]] = None) -> SentimentResult:
        """
        Analiza el sentimiento general de un activo.
        Si no se proporcionan titulares, intenta obtenerlos de APIs.
        """
        if headlines is None:
            headlines = self._fetch_news(symbol)

        if not headlines:
            return SentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                score=0.0,
                confidence=0.0,
                sources_analyzed=0,
                headlines=[],
                key_entities=[],
                classification="neutral",
            )

        scores = []
        confidences = []
        all_entities = []
        analyzed_headlines = []

        for headline in headlines:
            score, confidence = self.analyze_headline(headline)
            entities = self.extract_entities(headline)

            scores.append(score)
            confidences.append(confidence)
            all_entities.extend(entities)
            analyzed_headlines.append(headline)

        # Score compuesto (promedio ponderado por confianza)
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
            avg_confidence = total_confidence / len(confidences)
        else:
            weighted_score = 0.0
            avg_confidence = 0.0

        # Clasificación
        if weighted_score > 0.2:
            classification = "positive"
        elif weighted_score < -0.2:
            classification = "negative"
        else:
            classification = "neutral"

        result = SentimentResult(
            symbol=symbol,
            timestamp=datetime.now(),
            score=weighted_score,
            confidence=min(avg_confidence, 1.0),
            sources_analyzed=len(headlines),
            headlines=analyzed_headlines[:5],  # Guardar solo las 5 más relevantes
            key_entities=list(set(all_entities)),
            classification=classification,
        )

        self.last_results[symbol] = result
        return result

    def _fetch_news(self, symbol: str = "") -> List[str]:
        """
        Obtiene noticias financieras de APIs disponibles.
        """
        headlines = []

        # Intentar con NewsAPI
        if self.news_api_key:
            try:
                news_headlines = self._fetch_newsapi(symbol)
                headlines.extend(news_headlines)
            except Exception as e:
                logger.warning(f"Error fetching from NewsAPI: {e}")

        # Intentar con Finnhub
        if self.finnhub_api_key:
            try:
                finnhub_headlines = self._fetch_finnhub(symbol)
                headlines.extend(finnhub_headlines)
            except Exception as e:
                logger.warning(f"Error fetching from Finnhub: {e}")

        # Si no hay APIs, usar datos simulados para desarrollo
        if not headlines:
            headlines = self._get_default_headlines(symbol)

        return headlines

    def _fetch_newsapi(self, symbol: str) -> List[str]:
        """Obtiene noticias de NewsAPI."""
        query = symbol.replace("/", " ").split(" ")[0] if symbol else "financial markets"
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": self.news_api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [article["title"] for article in data.get("articles", []) if article.get("title")]
        return []

    def _fetch_finnhub(self, symbol: str) -> List[str]:
        """Obtiene noticias de Finnhub."""
        if not symbol:
            return []
        base_symbol = symbol.split("/")[0]
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": base_symbol,
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "to": datetime.now().strftime("%Y-%m-%d"),
            "token": self.finnhub_api_key,
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [article["headline"] for article in data if article.get("headline")][:10]
        return []

    def _get_default_headlines(self, symbol: str) -> List[str]:
        """Titulares por defecto para desarrollo/testing."""
        default_headlines = [
            "Markets rally on positive economic data",
            "Federal Reserve maintains current interest rates",
            "Tech stocks surge amid AI optimism",
            "Global markets show mixed results today",
            "Inflation concerns weigh on investor sentiment",
            "Corporate earnings beat expectations this quarter",
            "New regulations impact banking sector outlook",
            "Cryptocurrency market shows strong momentum",
            "Trade agreements boost international markets",
            "Consumer confidence reaches new highs",
        ]

        if symbol:
            base = symbol.split("/")[0]
            specific = [
                f"{base} shows strong growth potential",
                f"{base} faces regulatory challenges",
                f"Analysts upgrade {base} rating",
                f"{base} reports record quarterly results",
                f"Market sentiment turns bullish on {base}",
            ]
            return specific + default_headlines[:5]

        return default_headlines

    def get_sector_sentiment(self, sector: str) -> float:
        """Obtiene sentimiento agregado de un sector."""
        if sector in self.SECTOR_KEYWORDS:
            symbols = self.SECTOR_KEYWORDS[sector]
            scores = []
            for symbol in symbols:
                if symbol in self.last_results:
                    scores.append(self.last_results[symbol].score)
            if scores:
                return np.mean(scores)
        return 0.0

    def get_market_overview(self) -> Dict:
        """Obtiene una vista general del mercado."""
        overview = {}
        for sector in self.SECTOR_KEYWORDS:
            sentiment = self.get_sector_sentiment(sector)
            overview[sector] = {
                "sentiment": sentiment,
                "classification": (
                    "positive" if sentiment > 0.2
                    else "negative" if sentiment < -0.2
                    else "neutral"
                ),
                "strength": abs(sentiment),
            }
        return overview
