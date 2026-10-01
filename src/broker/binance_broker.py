"""
Implementación del bróker Binance usando CCXT.
Soporta testnet y modo real.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ccxt.async_support as ccxt
import logging
from .base import BaseBroker, OrderResult, AccountInfo

logger = logging.getLogger(__name__)


class BinanceBroker(BaseBroker):
    """
    Conexión con Binance (soporta testnet y producción).
    """

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
    }

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        testnet: bool = True,
    ):
        super().__init__("Binance")
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self._exchange = None

    async def connect(self) -> bool:
        """Conecta con Binance."""
        try:
            exchange_config = {
                "apiKey": self.api_key,
                "secret": self.secret_key,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                    "adjustForTimeDifference": True,
                },
            }

            if self.testnet:
                exchange_config.update({
                    "urls": {
                        "api": {
                            "public": "https://testnet.binance.vision/api/v3",
                            "private": "https://testnet.binance.vision/api/v3",
                        }
                    }
                })
                logger.info("Conectando a Binance TESTNET")

            self._exchange = ccxt.binance(exchange_config)
            await self._exchange.load_markets()
            self._connected = True
            logger.info(f"Conectado a Binance {'TESTNET' if self.testnet else 'PRODUCCIÓN'}")
            return True

        except Exception as e:
            logger.error(f"Error conectando a Binance: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Desconecta de Binance."""
        if self._exchange:
            await self._exchange.close()
            self._connected = False
            logger.info("Desconectado de Binance")

    async def get_account_info(self) -> AccountInfo:
        """Obtiene información de la cuenta."""
        try:
            balance = await self._exchange.fetch_balance()
            total_usdt = balance.get("USDT", {}).get("total", 0)
            free_usdt = balance.get("USDT", {}).get("free", 0)
            used_usdt = balance.get("USDT", {}).get("used", 0)

            return AccountInfo(
                total_equity=float(total_usdt),
                free_balance=float(free_usdt),
                locked_balance=float(used_usdt),
                currency="USDT",
            )
        except Exception as e:
            logger.error(f"Error obteniendo info de cuenta: {e}")
            raise

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        """
        Obtiene datos OHLCV históricos.
        
        Args:
            symbol: Símbolo del par (ej: BTC/USDT)
            timeframe: Timeframe (1m, 5m, 1h, 4h, 1d, etc.)
            limit: Número de velas a obtener
            
        Returns:
            DataFrame con columnas: timestamp, open, high, low, close, volume
        """
        try:
            tf = self.TIMEFRAME_MAP.get(timeframe, "1h")
            ohlcv = await self._exchange.fetch_ohlcv(symbol, tf, limit=limit)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Convertir a float
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            logger.info(f"Obtenidos {len(df)} datos OHLCV para {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error obteniendo OHLCV para {symbol}: {e}")
            raise

    async def get_ticker(self, symbol: str) -> Dict:
        """Obtiene ticker actual."""
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "bid": float(ticker.get("bid", 0)),
                "ask": float(ticker.get("ask", 0)),
                "last": float(ticker.get("last", 0)),
                "high": float(ticker.get("high", 0)),
                "low": float(ticker.get("low", 0)),
                "volume": float(ticker.get("baseVolume", 0)),
                "change": float(ticker.get("percentage", 0)),
                "timestamp": datetime.fromtimestamp(ticker.get("timestamp", 0) / 1000),
            }
        except Exception as e:
            logger.error(f"Error obteniendo ticker para {symbol}: {e}")
            raise

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> OrderResult:
        """Crea una orden en Binance."""
        try:
            # Ajustar cantidad a los lotes permitidos
            market = self._exchange.market(symbol)
            quantity = self._exchange.amount_to_precision(symbol, quantity)
            price_precision = market.get("precision", {}).get("price", 8)

            order_params = {}

            # Si es orden stop-loss
            if stop_loss and order_type == "market":
                # Creamos orden de mercado primero
                order_type = "market"

            if order_type == "limit" and price:
                price = self._exchange.price_to_precision(symbol, price)

            # Crear la orden principal
            order = await self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side.lower(),
                amount=float(quantity),
                price=price,
                params=order_params,
            )

            result = OrderResult(
                order_id=str(order.get("id", "")),
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                price=float(order.get("price", 0)),
                status=order.get("status", "unknown"),
                timestamp=datetime.fromtimestamp(order.get("timestamp", 0) / 1000),
                filled_quantity=float(order.get("filled", 0)),
                average_price=float(order.get("average", 0)),
                commission=float(order.get("fee", {}).get("cost", 0)),
            )

            # Si tiene stop loss, crear orden de stop loss
            if stop_loss and result.status == "closed":
                stop_side = "sell" if side.lower() == "buy" else "buy"
                try:
                    sl_order = await self._exchange.create_order(
                        symbol=symbol,
                        type="stop_market",
                        side=stop_side,
                        amount=float(quantity),
                        price=float(stop_loss),
                        params={"stopPrice": float(stop_loss)},
                    )
                    logger.info(f"Stop loss creado: {sl_order.get('id')}")
                except Exception as e:
                    logger.warning(f"No se pudo crear stop loss: {e}")

            # Si tiene take profit
            if take_profit and result.status == "closed":
                tp_side = "sell" if side.lower() == "buy" else "buy"
                try:
                    tp_order = await self._exchange.create_order(
                        symbol=symbol,
                        type="limit",
                        side=tp_side,
                        amount=float(quantity),
                        price=float(take_profit),
                    )
                    logger.info(f"Take profit creado: {tp_order.get('id')}")
                except Exception as e:
                    logger.warning(f"No se pudo crear take profit: {e}")

            logger.info(
                f"Orden {side} {quantity} {symbol} - "
                f"ID: {result.order_id} - Estado: {result.status}"
            )
            return result

        except Exception as e:
            logger.error(f"Error creando orden en Binance: {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price or 0,
                status="rejected",
                timestamp=datetime.now(),
                error_message=str(e),
            )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancela una orden."""
        try:
            await self._exchange.cancel_order(order_id, symbol)
            logger.info(f"Orden {order_id} cancelada para {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error cancelando orden {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """Obtiene estado de una orden."""
        try:
            order = await self._exchange.fetch_order(order_id, symbol)
            return OrderResult(
                order_id=str(order.get("id", "")),
                symbol=symbol,
                side=order.get("side", ""),
                quantity=float(order.get("amount", 0)),
                price=float(order.get("price", 0)),
                status=order.get("status", "unknown"),
                timestamp=datetime.fromtimestamp(order.get("timestamp", 0) / 1000),
                filled_quantity=float(order.get("filled", 0)),
                average_price=float(order.get("average", 0)),
                commission=float(order.get("fee", {}).get("cost", 0)),
            )
        except Exception as e:
            logger.error(f"Error obteniendo estado orden {order_id}: {e}")
            raise

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Obtiene órdenes abiertas."""
        try:
            orders = await self._exchange.fetch_open_orders(symbol)
            results = []
            for order in orders:
                results.append(OrderResult(
                    order_id=str(order.get("id", "")),
                    symbol=order.get("symbol", ""),
                    side=order.get("side", ""),
                    quantity=float(order.get("amount", 0)),
                    price=float(order.get("price", 0)),
                    status=order.get("status", "unknown"),
                    timestamp=datetime.fromtimestamp(order.get("timestamp", 0) / 1000),
                    filled_quantity=float(order.get("filled", 0)),
                ))
            return results
        except Exception as e:
            logger.error(f"Error obteniendo órdenes abiertas: {e}")
            raise

    async def get_positions(self) -> List[Dict]:
        """Obtiene posiciones abiertas (spot no tiene posiciones, pero lo simulamos)."""
        try:
            balance = await self._exchange.fetch_balance()
            positions = []
            for currency, data in balance.get("total", {}).items():
                if currency != "USDT" and data > 0:
                    positions.append({
                        "symbol": f"{currency}/USDT",
                        "quantity": float(data),
                        "currency": currency,
                    })
            return positions
        except Exception as e:
            logger.error(f"Error obteniendo posiciones: {e}")
            raise

    async def get_available_symbols(self) -> List[str]:
        """Obtiene lista de símbolos disponibles."""
        try:
            if not self._exchange:
                return []
            return [s for s in self._exchange.symbols if "/" in s and "UP" not in s and "DOWN" not in s]
        except Exception as e:
            logger.error(f"Error obteniendo símbolos: {e}")
            return []
