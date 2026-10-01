"""
Módulo de Gestión de Riesgo - CORAZÓN DEL SISTEMA.
Todas las operaciones pasan por aquí antes de ejecutarse.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    entry_price: float
    current_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    direction: int  # 1 long, -1 short
    entry_time: datetime
    position_id: str = ""
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = "open"  # open, closed, stopped, took_profit


@dataclass
class RiskAssessment:
    approved: bool
    reason: str = ""
    suggested_position_size: float = 0.0
    suggested_stop_loss: float = 0.0
    suggested_take_profit: float = 0.0
    risk_amount: float = 0.0
    risk_percentage: float = 0.0
    expected_profit: float = 0.0
    risk_reward_ratio: float = 0.0


class RiskManager:
    """
    Gestor de riesgo central. Valida cada operación antes de ejecutarla.
    Implementa todas las reglas de la guía.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_risk_per_trade: float = 2.0,
        max_daily_loss: float = 3.0,
        max_correlation: float = 0.7,
        stop_loss_atr_multiplier: float = 2.0,
        take_profit_ratio: float = 1.5,
        max_positions: int = 5,
        circuit_breaker_news: bool = True,
    ):
        # Capital
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Límites
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_correlation = max_correlation
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_ratio = take_profit_ratio
        self.max_positions = max_positions
        self.circuit_breaker_news = circuit_breaker_news

        # Estado
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.last_reset_day = datetime.now().date()
        self.trade_history: List[Dict] = []
        self.high_impact_news_active = False
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None

        # Estadísticas
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0.0
        self.win_streak = 0
        self.loss_streak = 0

    def _reset_daily_counters(self):
        """Resetea contadores diarios si es un nuevo día."""
        today = datetime.now().date()
        if today != self.last_reset_day:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_losses = 0
            self.last_reset_day = today
            logger.info("Contadores diarios reseteados")

    def assess_trade(
        self,
        symbol: str,
        entry_price: float,
        direction: int,
        atr: Optional[float] = None,
        sentiment_score: float = 0.0,
        market_conditions: Optional[Dict] = None,
    ) -> RiskAssessment:
        """
        Evalúa una operación propuesta contra todas las reglas de riesgo.
        Esta es la función principal que se llama antes de cada trade.
        """
        self._reset_daily_counters()

        # 1. Verificar circuit breaker
        if self.circuit_breaker_active:
            if self.circuit_breaker_until and datetime.now() < self.circuit_breaker_until:
                return RiskAssessment(
                    approved=False,
                    reason=f"Circuit breaker activo hasta {self.circuit_breaker_until}",
                )
            else:
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None

        # 2. Verificar noticias de alto impacto
        if self.circuit_breaker_news and self.high_impact_news_active:
            return RiskAssessment(
                approved=False,
                reason="Noticia de alto impacto detectada, operaciones pausadas",
            )

        # 3. Verificar límite de posiciones
        if len(self.open_positions) >= self.max_positions:
            return RiskAssessment(
                approved=False,
                reason=f"Máximo de {self.max_positions} posiciones alcanzado",
            )

        # 4. Verificar correlación
        if not self._check_correlation(symbol, direction):
            return RiskAssessment(
                approved=False,
                reason=f"Alta correlación detectada con posiciones existentes",
            )

        # 5. Verificar límite diario de pérdidas
        max_loss_amount = self.current_capital * (self.max_daily_loss / 100)
        if self.daily_pnl <= -max_loss_amount:
            self._activate_circuit_breaker(hours=24)
            return RiskAssessment(
                approved=False,
                reason=f"Límite diario de pérdida alcanzado ({self.daily_pnl:.2f})",
            )

        # 6. Verificar drawdown
        if not self._check_drawdown():
            return RiskAssessment(
                approved=False,
                reason=f"Drawdown máximo excedido",
            )

        # 7. Verificar racha de pérdidas
        if self.consecutive_losses >= 5:
            self._activate_circuit_breaker(hours=12)
            return RiskAssessment(
                approved=False,
                reason=f"5 pérdidas consecutivas, circuit breaker activado",
            )

        # 8. Calcular SL y TP basados en ATR
        if atr and atr > 0:
            suggested_sl, suggested_tp = self._calculate_sl_tp(entry_price, atr, direction)
        else:
            # Si no hay ATR, usar porcentaje fijo
            suggested_sl = entry_price * (1 - 0.02 * direction)
            suggested_tp = entry_price * (1 + 0.03 * direction)

        # 9. Calcular tamaño de posición
        risk_per_trade = self._calculate_risk_percentage(sentiment_score)
        suggested_size = self._calculate_position_size(
            self.current_capital, entry_price, suggested_sl, risk_per_trade
        )

        # 10. Calcular métricas de riesgo
        risk_amount = abs(entry_price - suggested_sl) * suggested_size
        risk_percentage = (risk_amount / self.current_capital) * 100
        potential_profit = abs(suggested_tp - entry_price) * suggested_size
        risk_reward = potential_profit / risk_amount if risk_amount > 0 else 0

        # 11. Verificar riesgo máximo por operación
        if risk_percentage > self.max_risk_per_trade:
            # Ajustar tamaño de posición
            suggested_size = self._calculate_position_size(
                self.current_capital, entry_price, suggested_sl, self.max_risk_per_trade
            )
            risk_amount = abs(entry_price - suggested_sl) * suggested_size
            risk_percentage = (risk_amount / self.current_capital) * 100

        # 12. Verificar risk/reward mínimo
        if risk_reward < 1.0:
            return RiskAssessment(
                approved=False,
                reason=f"Risk/Reward muy bajo: {risk_reward:.2f}. Mínimo: 1.0",
            )

        assessment = RiskAssessment(
            approved=True,
            reason="Operación aprobada",
            suggested_position_size=suggested_size,
            suggested_stop_loss=suggested_sl,
            suggested_take_profit=suggested_tp,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            expected_profit=potential_profit,
            risk_reward_ratio=risk_reward,
        )

        logger.info(
            f"Trade aprobado: {symbol} | "
            f"Tamaño: {suggested_size:.4f} | "
            f"Riesgo: ${risk_amount:.2f} ({risk_percentage:.2f}%) | "
            f"R/R: {risk_reward:.2f}"
        )

        return assessment

    def open_position(self, position: Position):
        """Registra una nueva posición abierta."""
        position.position_id = f"POS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.trade_history)}"
        self.open_positions.append(position)
        self.daily_trades += 1
        logger.info(f"Posición abierta: {position.symbol} | ID: {position.position_id}")

    def update_positions(self, current_prices: Dict[str, float]):
        """
        Actualiza precios actuales de todas las posiciones y verifica SL/TP.
        """
        positions_to_close = []

        for pos in self.open_positions:
            if pos.symbol in current_prices:
                pos.current_price = current_prices[pos.symbol]
                pos.unrealized_pnl = self._calculate_pnl(pos)

                # Verificar Stop Loss
                if pos.direction == 1 and pos.current_price <= pos.stop_loss:
                    pos.realized_pnl = pos.unrealized_pnl
                    pos.status = "stopped"
                    positions_to_close.append(pos)
                    logger.info(f"Stop Loss activado: {pos.symbol} @ {pos.current_price}")

                elif pos.direction == -1 and pos.current_price >= pos.stop_loss:
                    pos.realized_pnl = pos.unrealized_pnl
                    pos.status = "stopped"
                    positions_to_close.append(pos)
                    logger.info(f"Stop Loss activado: {pos.symbol} @ {pos.current_price}")

                # Verificar Take Profit
                elif pos.direction == 1 and pos.current_price >= pos.take_profit:
                    pos.realized_pnl = pos.unrealized_pnl
                    pos.status = "took_profit"
                    positions_to_close.append(pos)
                    logger.info(f"Take Profit activado: {pos.symbol} @ {pos.current_price}")

                elif pos.direction == -1 and pos.current_price <= pos.take_profit:
                    pos.realized_pnl = pos.unrealized_pnl
                    pos.status = "took_profit"
                    positions_to_close.append(pos)
                    logger.info(f"Take Profit activado: {pos.symbol} @ {pos.current_price}")

        # Cerrar posiciones
        for pos in positions_to_close:
            self.close_position(pos)

    def close_position(self, position: Position):
        """Cierra una posición y actualiza estadísticas."""
        if position in self.open_positions:
            self.open_positions.remove(position)
            self.closed_positions.append(position)

            # Calcular PnL si no se ha calculado aún
            if position.realized_pnl == 0.0:
                position.realized_pnl = self._calculate_pnl(position)

            # Actualizar capital
            self.current_capital += position.realized_pnl
            self.daily_pnl += position.realized_pnl
            self.total_pnl += position.realized_pnl

            # Actualizar pico de capital
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital

            # Estadísticas de wins/losses
            if position.realized_pnl > 0:
                self.total_wins += 1
                self.win_streak += 1
                self.loss_streak = 0
                self.consecutive_losses = 0
            else:
                self.total_losses += 1
                self.loss_streak += 1
                self.win_streak = 0
                self.consecutive_losses += 1
                self.daily_losses += 1

            # Registrar en historial
            self.trade_history.append({
                "position_id": position.position_id,
                "symbol": position.symbol,
                "direction": "long" if position.direction == 1 else "short",
                "entry_price": position.entry_price,
                "exit_price": position.current_price,
                "quantity": position.quantity,
                "pnl": position.realized_pnl,
                "status": position.status,
                "entry_time": position.entry_time,
                "exit_time": datetime.now(),
                "holding_period": str(datetime.now() - position.entry_time),
            })

            logger.info(
                f"Posición cerrada: {position.symbol} | "
                f"PnL: ${position.realized_pnl:.2f} | "
                f"Status: {position.status}"
            )

    def _check_correlation(self, symbol: str, direction: int) -> bool:
        """Verifica correlación con posiciones existentes."""
        if len(self.open_positions) < 2:
            return True

        # Pares de alta correlación conocidos
        correlated_pairs = {
            ("BTC/USDT", "ETH/USDT"): 0.85,
            ("BTC/USDT", "SOL/USDT"): 0.75,
            ("ETH/USDT", "SOL/USDT"): 0.80,
            ("EUR/USD", "GBP/USD"): 0.80,
            ("EUR/USD", "USD/CHF"): -0.75,
            ("GBP/USD", "USD/CHF"): -0.70,
            ("SPY", "QQQ"): 0.85,
            ("SPY", "DIA"): 0.90,
        }

        for pos in self.open_positions:
            pair = tuple(sorted([symbol, pos.symbol]))
            if pair in correlated_pairs:
                correlation = abs(correlated_pairs[pair])
                if correlation > self.max_correlation and direction == pos.direction:
                    logger.warning(
                        f"Correlación alta: {symbol}-{pos.symbol} "
                        f"({correlation:.2f}) en misma dirección"
                    )
                    return False
        return True

    def _check_drawdown(self) -> bool:
        """Verifica que el drawdown no exceda el 20%."""
        if self.peak_capital <= 0:
            return True

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        if drawdown > 20:
            logger.warning(f"Drawdown crítico: {drawdown:.2f}%")
            return False
        return True

    def _calculate_sl_tp(self, entry_price: float, atr: float, direction: int) -> Tuple[float, float]:
        """Calcula Stop Loss y Take Profit basados en ATR."""
        atr_distance = atr * self.stop_loss_atr_multiplier

        if direction == 1:  # Long
            stop_loss = entry_price - atr_distance
            take_profit = entry_price + (atr_distance * self.take_profit_ratio)
        else:  # Short
            stop_loss = entry_price + atr_distance
            take_profit = entry_price - (atr_distance * self.take_profit_ratio)

        return stop_loss, take_profit

    def _calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float,
    ) -> float:
        """
        Calcula tamaño de posición usando Kelly Criterion modificado.
        """
        # Diferencia entre entrada y stop loss
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0.0

        risk_per_unit_pct = risk_per_unit / entry_price
        capital_at_risk = capital * (risk_percent / 100)

        # Tamaño base
        base_size = capital_at_risk / (entry_price * risk_per_unit_pct)

        # Aplicar Kelly fraccional (25% de Kelly para ser conservadores)
        win_rate = self._get_historical_win_rate()
        avg_win_loss = self._get_avg_win_loss_ratio()

        if win_rate > 0 and avg_win_loss > 0:
            kelly = win_rate - ((1 - win_rate) / avg_win_loss)
            kelly = max(0, min(kelly * 0.25, 0.25))
        else:
            kelly = risk_percent / 100

        # Tamaño final ajustado por Kelly
        adjusted_capital_at_risk = capital * kelly
        final_size = adjusted_capital_at_risk / (entry_price * risk_per_unit_pct)

        return max(0, final_size)

    def _calculate_risk_percentage(self, sentiment_score: float) -> float:
        """Ajusta el riesgo según las condiciones del mercado."""
        base_risk = min(self.max_risk_per_trade, 2.0)

        # Reducir riesgo si el sentimiento es negativo
        if sentiment_score < -0.5:
            base_risk *= 0.5
        elif sentiment_score < -0.2:
            base_risk *= 0.75

        # Reducir riesgo si hay pérdidas recientes
        if self.consecutive_losses >= 3:
            base_risk *= 0.5

        return base_risk

    def _calculate_pnl(self, position: Position) -> float:
        """Calcula PnL no realizado de una posición."""
        if position.direction == 1:  # Long
            return (position.current_price - position.entry_price) * position.quantity
        else:  # Short
            return (position.entry_price - position.current_price) * position.quantity

    def _get_historical_win_rate(self) -> float:
        """Obtiene tasa de aciertos histórica."""
        total = self.total_wins + self.total_losses
        if total == 0:
            return 0.55  # Default conservador
        return self.total_wins / total

    def _get_avg_win_loss_ratio(self) -> float:
        """Obtiene ratio ganancia/pérdida promedio."""
        if not self.trade_history:
            return 2.0  # Default conservador

        wins = [t["pnl"] for t in self.trade_history if t["pnl"] > 0]
        losses = [t["pnl"] for t in self.trade_history if t["pnl"] < 0]

        avg_win = np.mean(wins) if wins else 1
        avg_loss = abs(np.mean(losses)) if losses else 1

        return avg_win / avg_loss if avg_loss > 0 else 2.0

    def _activate_circuit_breaker(self, hours: int = 24):
        """Activa el circuit breaker."""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now() + timedelta(hours=hours)
        logger.warning(f"Circuit breaker activado por {hours} horas")

    def deactivate_circuit_breaker(self):
        """Desactiva el circuit breaker manualmente."""
        self.circuit_breaker_active = False
        self.circuit_breaker_until = None
        logger.info("Circuit breaker desactivado manualmente")

    def set_high_impact_news(self, active: bool):
        """Marca si hay noticias de alto impacto."""
        self.high_impact_news_active = active
        if active:
            logger.warning("Noticia de alto impacto detectada - operaciones pausadas")

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del gestor de riesgo."""
        total_trades = self.total_wins + self.total_losses
        win_rate = (self.total_wins / total_trades * 100) if total_trades > 0 else 0
        drawdown = ((self.peak_capital - self.current_capital) / self.peak_capital * 100) if self.peak_capital > 0 else 0

        return {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": win_rate,
            "win_streak": self.win_streak,
            "loss_streak": self.loss_streak,
            "open_positions": len(self.open_positions),
            "drawdown_percentage": drawdown,
            "circuit_breaker_active": self.circuit_breaker_active,
            "consecutive_losses": self.consecutive_losses,
        }

    def get_total_exposure(self) -> float:
        """Calcula la exposición total actual."""
        exposure = 0.0
        for pos in self.open_positions:
            exposure += pos.quantity * pos.entry_price
        return exposure

    def get_exposure_percentage(self) -> float:
        """Obtiene el porcentaje de capital expuesto."""
        exposure = self.get_total_exposure()
        return (exposure / self.current_capital * 100) if self.current_capital > 0 else 0
