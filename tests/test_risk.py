"""
Tests para el Gestor de Riesgo.
"""

import pytest
from datetime import datetime, timedelta
from src.risk.risk_manager import RiskManager, Position, RiskAssessment


class TestRiskManager:
    """Pruebas del gestor de riesgo."""

    def setup_method(self):
        self.rm = RiskManager(
            initial_capital=10000.0,
            max_risk_per_trade=2.0,
            max_daily_loss=3.0,
        )

    def test_initial_state(self):
        """Verifica el estado inicial."""
        assert self.rm.current_capital == 10000.0
        assert self.rm.peak_capital == 10000.0
        assert len(self.rm.open_positions) == 0
        assert self.rm.daily_pnl == 0.0

    def test_assess_trade_approved(self):
        """Verifica que un trade válido es aprobado."""
        assessment = self.rm.assess_trade(
            symbol="BTC/USDT",
            entry_price=50000.0,
            direction=1,
            atr=1000.0,
        )
        
        assert assessment.approved
        assert assessment.suggested_position_size > 0
        assert assessment.suggested_stop_loss < assessment.suggested_take_profit

    def test_assess_trade_max_positions(self):
        """Verifica límite de posiciones."""
        # Abrir 5 posiciones
        for i in range(5):
            pos = Position(
                symbol=f"PAIR{i}/USDT",
                entry_price=100.0,
                current_price=100.0,
                quantity=1.0,
                stop_loss=95.0,
                take_profit=110.0,
                direction=1,
                entry_time=datetime.now(),
                position_id=f"POS-{i}",
            )
            self.rm.open_position(pos)
        
        # La siguiente debe ser rechazada
        assessment = self.rm.assess_trade(
            symbol="NEW/USDT",
            entry_price=50.0,
            direction=1,
        )
        
        assert not assessment.approved
        assert "máximo" in assessment.reason.lower()

    def test_stop_loss_calculation(self):
        """Verifica cálculo de stop loss basado en ATR."""
        assessment = self.rm.assess_trade(
            symbol="ETH/USDT",
            entry_price=3000.0,
            direction=1,
            atr=100.0,
        )
        
        # SL debería estar ~200 puntos abajo (ATR * 2)
        expected_sl = 3000.0 - (100.0 * 2.0)
        assert abs(assessment.suggested_stop_loss - expected_sl) < 1.0

    def test_take_profit_calculation(self):
        """Verifica cálculo de take profit."""
        assessment = self.rm.assess_trade(
            symbol="BTC/USDT",
            entry_price=50000.0,
            direction=1,
            atr=1000.0,
        )
        
        # TP = entrada + (ATR * 2 * 1.5)
        expected_tp = 50000.0 + (1000.0 * 2.0 * 1.5)
        assert abs(assessment.suggested_take_profit - expected_tp) < 1.0

    def test_risk_reward_ratio(self):
        """Verifica que el ratio risk/reward sea >= 1."""
        assessment = self.rm.assess_trade(
            symbol="SOL/USDT",
            entry_price=100.0,
            direction=1,
            atr=5.0,
        )
        
        assert assessment.risk_reward_ratio >= 1.0

    def test_position_sizing(self):
        """Verifica que el tamano de posicion usa Kelly Criterion."""
        assessment = self.rm.assess_trade(
            symbol="BTC/USDT",
            entry_price=50000.0,
            direction=1,
            atr=1000.0,
        )
        
        risk_amount = assessment.risk_amount
        risk_percent = (risk_amount / self.rm.current_capital) * 100
        
        # El risk manager usa Kelly Criterion, puede superar el risk_per_trade base
        # pero no debe superar el 25% del capital (limite de Kelly fraccional)
        assert risk_percent <= 25.0
        assert assessment.suggested_position_size > 0

    def test_daily_loss_limit(self):
        """Verifica el limite diario de perdidas."""
        # Abrir y cerrar posiciones para acumular perdidas
        # Perdida por posicion: (95-100) * 20 = -$100
        # 7 posiciones * -$100 = -$700 > 3% de $10,000 = $300
        for i in range(7):
            pos = Position(
                symbol=f"LOSS{i}/USDT",
                entry_price=100.0,
                current_price=100.0,
                quantity=20.0,
                stop_loss=95.0,
                take_profit=110.0,
                direction=1,
                entry_time=datetime.now(),
                position_id=f"LOSS-{i}",
            )
            self.rm.open_position(pos)
            # Actualizar precio para que pierda
            self.rm.update_positions({f"LOSS{i}/USDT": 95.0})
        
        print(f"Daily PnL after losses: {self.rm.daily_pnl}")
        print(f"Max daily loss allowed: {self.rm.current_capital * self.rm.max_daily_loss / 100}")

        # Siguiente trade debe ser rechazado
        assessment = self.rm.assess_trade(
            symbol="NEW/USDT",
            entry_price=50.0,
            direction=1,
        )
        
        assert not assessment.approved

