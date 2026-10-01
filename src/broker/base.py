"""
Base class for broker integrations.
Define la interfaz común para todos los brókers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str  # buy or sell
    quantity: float
    price: float
    status: str  # filled, partially_filled, canceled, rejected
    timestamp: datetime
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    error_message: str = ""


@dataclass
class AccountInfo:
    total_equity: float
    free_balance: float
    locked_balance: float
    currency: str
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0


class BaseBroker(ABC):
    """Interfaz base para todos los brókers."""

    def __init__(self, name: str):
        self.name = name
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Conecta con el bróker."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Desconecta del bróker."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Obtiene información de la cuenta."""
        pass

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, timeframe: str = "1h", limit: int = 500
    ) -> pd.DataFrame:
        """Obtiene datos OHLCV históricos."""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Obtiene ticker actual de un símbolo."""
        pass

    @abstractmethod
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
        """Crea una orden en el bróker."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancela una orden existente."""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> OrderResult:
        """Obtiene el estado de una orden."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """Obtiene todas las órdenes abiertas."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Obtiene posiciones abiertas."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    def validate_symbol(self, symbol: str) -> bool:
        """Valida que el símbolo sea válido para este bróker."""
        return True
