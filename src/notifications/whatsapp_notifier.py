"""
Notificador de WhatsApp - Envía señales y alertas usando Twilio.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WhatsAppNotifier:
    """
    Maneja el envío de notificaciones por WhatsApp usando Twilio API.
    """

    def __init__(
        self,
        account_sid: str = "",
        auth_token: str = "",
        from_number: str = "",
    ):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self._enabled = bool(account_sid and auth_token and from_number)

    def _build_signal_text(self, signal_data: Dict) -> str:
        """Construye el mensaje de señal."""
        direction = signal_data.get("direction", "NEUTRAL")
        emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"

        text = f"""
{emoji} *SEÑAL DE TRADING*

Activo: {signal_data.get('symbol', 'N/A')}
Dirección: {direction}
Entrada: ${signal_data.get('entry_price', 0):.4f}
Stop Loss: ${signal_data.get('stop_loss', 0):.4f}
Take Profit: ${signal_data.get('take_profit', 0):.4f}
Confianza: {signal_data.get('confidence', 0):.1%}

Motivo: {signal_data.get('reason', 'N/A')}

⚠️ No es asesoría financiera. Invertir conlleva riesgo.
        """
        return text.strip()

    def _build_pnl_text(self, symbol: str, pnl: float, status: str) -> str:
        """Construye mensaje de resultado."""
        emoji = "✅" if pnl >= 0 else "❌"
        text = f"""
{emoji} OPERACIÓN CERRADA

{symbol}
PnL: ${pnl:.2f}
Estado: {status}
        """
        return text.strip()

    async def send_signal(self, signal_data: Dict) -> bool:
        """Envía una señal de trading por WhatsApp."""
        if not self._enabled:
            logger.info("WhatsApp no configurado")
            return False

        text = self._build_signal_text(signal_data)
        return await self._send_message(text)

    async def send_pnl_update(self, symbol: str, pnl: float, status: str) -> bool:
        """Envía actualización de PnL."""
        if not self._enabled:
            return False

        text = self._build_pnl_text(symbol, pnl, status)
        return await self._send_message(text)

    async def send_alert(self, alert_type: str, message: str) -> bool:
        """Envía una alerta por WhatsApp."""
        if not self._enabled:
            return False

        text = f"""
{'⚠️' if alert_type == 'warning' else '🚨' if alert_type == 'error' else '📢'} ALERTA: {alert_type.upper()}

{message}
        """
        return await self._send_message(text.strip())

    async def _send_message(self, text: str, to_number: str = "") -> bool:
        """Envía un mensaje por WhatsApp usando Twilio."""
        if not self._enabled:
            return False

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)

            message = client.messages.create(
                body=text,
                from_=f"whatsapp:{self.from_number}",
                to=f"whatsapp:{to_number}" if to_number else f"whatsapp:{self.from_number}",
            )

            logger.info(f"Mensaje WhatsApp enviado: {message.sid}")
            return True

        except Exception as e:
            logger.error(f"Error enviando mensaje WhatsApp: {e}")
            return False

    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """Envía resumen diario por WhatsApp."""
        if not self._enabled:
            return False

        text = f"""
📊 RESUMEN DIARIO

Operaciones: {summary_data.get('total_trades', 0)}
Ganadas: {summary_data.get('wins', 0)}
Perdidas: {summary_data.get('losses', 0)}
Win Rate: {summary_data.get('win_rate', 0):.1f}%
PnL: ${summary_data.get('pnl', 0):.2f}
        """
        return await self._send_message(text.strip())
