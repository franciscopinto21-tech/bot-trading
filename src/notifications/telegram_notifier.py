"""
Notificador de Telegram - Envía señales y alertas del bot.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalMessage:
    symbol: str
    direction: str  # LONG o SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    confidence: float
    reason: str
    market_type: str
    timestamp: datetime


class TelegramNotifier:
    """
    Maneja el envío de notificaciones por Telegram.
    """

    def __init__(self, bot_token: str = "", chat_id: str = ""):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    def _build_signal_text(self, signal: SignalMessage) -> str:
        """Construye el mensaje formateado para una señal."""
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        
        text = f"""
{emoji} *SEÑAL DE {signal.direction}* {emoji}

*{signal.symbol}* | {signal.market_type}
⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━
*📊 Precios:*
┣ 💰 Entrada: `${signal.entry_price:.4f}`
┣ 🛑 Stop Loss: `${signal.stop_loss:.4f}`
┗ 🎯 Take Profit: `${signal.take_profit:.4f}`

*📐 Riesgo:*
┣ 📦 Tamaño: `{signal.position_size:.4f}`
┗ ⚡ Confianza: `{signal.confidence:.1%}`

*💡 Razón:*
{signal.reason}

━━━━━━━━━━━━━━━
⚠️ *No es asesoría financiera*
Las inversiones conllevan riesgo
        """
        return text.strip()

    def _build_pnl_text(
        self, symbol: str, pnl: float, pnl_percent: float, status: str
    ) -> str:
        """Construye mensaje de resultado de operación."""
        emoji = "✅" if pnl > 0 else "❌"
        status_emoji = {
            "took_profit": "🎯",
            "stopped": "🛑",
            "manual": "👤",
        }.get(status, "ℹ️")

        text = f"""
{emoji} *OPERACIÓN CERRADA* {status_emoji}

*{symbol}*
📊 Resultado: `${pnl:.2f}` ({pnl_percent:+.2f}%)
🔄 Estado: {status}

💰 PnL Total: `${pnl:.2f}`
        """
        return text.strip()

    def _build_alert_text(self, alert_type: str, message: str) -> str:
        """Construye mensaje de alerta."""
        icons = {
            "warning": "⚠️",
            "error": "🚨",
            "info": "ℹ️",
            "success": "✅",
            "risk": "🛑",
        }
        icon = icons.get(alert_type, "📢")

        text = f"""
{icon} *ALERTA: {alert_type.upper()}*

{message}

⏰ {datetime.now().strftime('%H:%M:%S')}
        """
        return text.strip()

    def _build_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        pnl: float,
        best_trade: Optional[Dict] = None,
        worst_trade: Optional[Dict] = None,
    ) -> str:
        """Construye resumen diario."""
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        text = f"""
📊 *RESUMEN DIARIO*
━━━━━━━━━━━━━━━

*📈 Estadísticas:*
┣ 🔄 Total Operaciones: {total_trades}
┣ ✅ Ganadas: {wins}
┣ ❌ Perdidas: {losses}
┗ 📊 Win Rate: {win_rate:.1f}%

*💰 Resultado:*
┣ 📈 PnL: `${pnl:.2f}`
┗ 📊 Balance: `{'✅ Positivo' if pnl >= 0 else '❌ Negativo'}`

*🏆 Mejor operación:*
{self._format_trade(best_trade) if best_trade else 'N/A'}

*📉 Peor operación:*
{self._format_trade(worst_trade) if worst_trade else 'N/A'}
        """
        return text.strip()

    def _format_trade(self, trade: Dict) -> str:
        """Formatea una operación individual."""
        return (
            f"┣ {trade.get('symbol', 'N/A')}: "
            f"`${trade.get('pnl', 0):.2f}` "
            f"({trade.get('status', 'N/A')})"
        )

    async def send_signal(self, signal: SignalMessage) -> bool:
        """Envía una señal de trading por Telegram."""
        if not self._enabled:
            logger.info(f"Telegram no configurado. Señal para {signal.symbol} no enviada.")
            return False

        text = self._build_signal_text(signal)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_pnl_update(
        self, symbol: str, pnl: float, pnl_percent: float, status: str
    ) -> bool:
        """Envía actualización de PnL."""
        if not self._enabled:
            return False

        text = self._build_pnl_text(symbol, pnl, pnl_percent, status)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_alert(self, alert_type: str, message: str) -> bool:
        """Envía una alerta."""
        if not self._enabled:
            return False

        text = self._build_alert_text(alert_type, message)
        return await self._send_message(text, parse_mode="Markdown")

    async def send_daily_summary(
        self,
        total_trades: int,
        wins: int,
        losses: int,
        pnl: float,
        best_trade: Optional[Dict] = None,
        worst_trade: Optional[Dict] = None,
    ) -> bool:
        """Envía resumen diario."""
        if not self._enabled:
            return False

        text = self._build_daily_summary(total_trades, wins, losses, pnl, best_trade, worst_trade)
        return await self._send_message(text, parse_mode="Markdown")

    async def _send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """Envía un mensaje a Telegram."""
        if not self._enabled:
            return False

        try:
            import requests

            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Mensaje Telegram enviado exitosamente")
                return True
            else:
                logger.error(f"Error enviando mensaje Telegram: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error enviando mensaje a Telegram: {e}")
            return False

    async def send_error(self, error_message: str) -> bool:
        """Envía una alerta de error crítico."""
        text = f"""
🚨 *ERROR CRÍTICO DEL SISTEMA*

{error_message}

⏰ {datetime.now().strftime('%H:%M:%S')}
🔧 Se requiere atención inmediata
        """
        return await self._send_message(text.strip(), parse_mode="Markdown")
