"""
Motor de Backtesting - Prueba estrategias con datos históricos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    stop_loss: float
    take_profit: float
    reason: str
    fees: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    profit_factor: float
    sharpe_ratio: float
    avg_win: float
    avg_loss: float
    avg_holding_period: timedelta
    trades: List[BacktestTrade]
    equity_curve: List[float]
    start_date: datetime
    end_date: datetime


class Backtester:
    """
    Motor de backtesting con walk-forward validation.
    Simula operaciones con comisiones, slippage y reglas de riesgo.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0005,  # 0.05%
        risk_per_trade: float = 2.0,  # %
        stop_loss_atr: float = 2.0,
        take_profit_ratio: float = 1.5,
        max_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.risk_per_trade = risk_per_trade
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_ratio = take_profit_ratio
        self.max_positions = max_positions

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        use_walk_forward: bool = True,
        training_months: int = 6,
        testing_months: int = 1,
    ) -> BacktestResult:
        """
        Ejecuta backtesting completo.
        
        Args:
            df: DataFrame OHLCV con índice de tiempo
            symbol: Símbolo del activo
            use_walk_forward: Si usar walk-forward testing
            training_months: Meses de entrenamiento por ventana
            testing_months: Meses de prueba por ventana
        """
        df = df.copy()
        
        # Calcular indicadores necesarios
        df = self._prepare_data(df)

        if use_walk_forward:
            return self._run_walk_forward(
                df, symbol, training_months, testing_months
            )
        else:
            return self._run_single(df, symbol)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos con indicadores necesarios."""
        df = df.copy()

        # ATR
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # EMA
        df["ema_9"] = df["close"].ewm(span=9).mean()
        df["ema_21"] = df["close"].ewm(span=21).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()

        # Bollinger
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + 2 * std_20
        df["bb_lower"] = sma_20 - 2 * std_20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Volumen ratio
        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)

        # Limpiar NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        return df

    def _run_walk_forward(
        self,
        df: pd.DataFrame,
        symbol: str,
        training_months: int,
        testing_months: int,
    ) -> BacktestResult:
        """Ejecuta walk-forward testing."""
        train_days = training_months * 30
        test_days = testing_months * 30

        all_trades = []
        equity_curves = []
        test_start = df.index[0] + pd.Timedelta(days=train_days)

        # Ventanas de testing
        test_windows = []
        current = test_start
        while current < df.index[-1]:
            window_end = current + pd.Timedelta(days=test_days)
            test_windows.append((current, min(window_end, df.index[-1])))
            current = window_end

        logger.info(f"Ejecutando walk-forward con {len(test_windows)} ventanas")

        progress_bar = tqdm(test_windows, desc="Walk-Forward Testing")
        for window_start, window_end in progress_bar:
            window_data = df.loc[window_start:window_end]
            trades, equity = self._simulate_trading(window_data, symbol)
            all_trades.extend(trades)
            equity_curves.extend(equity)
            progress_bar.set_postfix({"trades": len(trades)})

        return self._calculate_results(
            all_trades, equity_curves, df, symbol
        )

    def _run_single(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Ejecuta backtesting simple."""
        trades, equity = self._simulate_trading(df, symbol)
        return self._calculate_results(trades, equity, df, symbol)

    def _simulate_trading(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[List[BacktestTrade], List[float]]:
        """Simula trading sobre los datos."""
        trades = []
        equity = [self.initial_capital]
        capital = self.initial_capital
        open_positions: List[Dict] = []

        for i in range(50, len(df)):
            # Actualizar precios de posiciones abiertas
            current_price = df.iloc[i]["close"]
            self._update_positions(open_positions, current_price, df.iloc[i])

            # Verificar SL/TP
            closed = self._check_sl_tp(open_positions, df.iloc[i])
            for pos in closed:
                pnl = self._close_position(pos, df.iloc[i])
                capital += pnl
                
                trades.append(BacktestTrade(
                    entry_time=pos["entry_time"],
                    exit_time=df.index[i],
                    symbol=symbol,
                    direction=pos["direction"],
                    entry_price=pos["entry_price"],
                    exit_price=pos["exit_price"],
                    quantity=pos["quantity"],
                    pnl=pnl,
                    pnl_percent=(pnl / (pos["entry_price"] * pos["quantity"])) * 100 if pos["entry_price"] * pos["quantity"] > 0 else 0,
                    stop_loss=pos["stop_loss"],
                    take_profit=pos["take_profit"],
                    reason=pos["close_reason"],
                    fees=abs(pnl) * self.commission_pct,
                ))

            # Generar señal (estrategia simple)
            if len(open_positions) < self.max_positions:
                signal = self._generate_signal(df.iloc[i], df.iloc[i - 1])
                
                if signal["direction"] != 0:
                    # Calcular SL/TP
                    atr = df.iloc[i]["atr"]
                    entry_price = current_price * (1 + signal["direction"] * self.slippage_pct)
                    
                    if signal["direction"] == 1:  # Long
                        sl = entry_price - atr * self.stop_loss_atr
                        tp = entry_price + atr * self.stop_loss_atr * self.take_profit_ratio
                    else:  # Short
                        sl = entry_price + atr * self.stop_loss_atr
                        tp = entry_price - atr * self.stop_loss_atr * self.take_profit_ratio

                    # Calcular tamaño
                    risk_amount = capital * (self.risk_per_trade / 100)
                    risk_per_unit = abs(entry_price - sl)
                    quantity = risk_amount / (entry_price * (risk_per_unit / entry_price)) if risk_per_unit > 0 else 0

                    if quantity > 0 and capital > 0:
                        pos = {
                            "symbol": symbol,
                            "direction": "LONG" if signal["direction"] == 1 else "SHORT",
                            "entry_price": entry_price,
                            "current_price": entry_price,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "quantity": quantity,
                            "entry_time": df.index[i],
                            "close_reason": "",
                            "exit_price": 0,
                        }
                        open_positions.append(pos)

            # Registrar equity
            total_value = capital
            for pos in open_positions:
                pnl = (current_price - pos["entry_price"]) * pos["quantity"] if pos["direction"] == "LONG" else (pos["entry_price"] - current_price) * pos["quantity"]
                total_value += pnl
            equity.append(total_value)

        return trades, equity

    def _generate_signal(self, current_row: pd.Series, prev_row: pd.Series) -> Dict:
        """
        Estrategia de trading simple para backtesting.
        Señal basada en múltiples indicadores.
        """
        score = 0.0

        # EMA Cross
        if current_row["ema_9"] > current_row["ema_21"] and prev_row["ema_9"] <= prev_row["ema_21"]:
            score += 2.0
        elif current_row["ema_9"] < current_row["ema_21"] and prev_row["ema_9"] >= prev_row["ema_21"]:
            score -= 2.0

        # RSI
        if current_row["rsi"] < 30:
            score += 1.5
        elif current_row["rsi"] > 70:
            score -= 1.5

        # MACD
        if current_row["macd_hist"] > 0 and prev_row["macd_hist"] <= 0:
            score += 1.5
        elif current_row["macd_hist"] < 0 and prev_row["macd_hist"] >= 0:
            score -= 1.5

        # Bollinger Bands
        if current_row["bb_position"] < 0.2:
            score += 1.0
        elif current_row["bb_position"] > 0.8:
            score -= 1.0

        # Volumen
        if current_row["volume_ratio"] > 1.5:
            score *= 1.2

        # Decisión (umbrales ajustados para generar más señales)
        if score >= 2.0:
            return {"direction": 1}
        elif score <= -2.0:
            return {"direction": -1}
        return {"direction": 0}

    def _update_positions(self, positions: List[Dict], price: float, row: pd.Series):
        """Actualiza precios de posiciones abiertas."""
        for pos in positions:
            pos["current_price"] = price

    def _check_sl_tp(self, positions: List[Dict], row: pd.Series) -> List[Dict]:
        """Verifica si alguna posición debe cerrarse por SL/TP."""
        closed = []
        remaining = []

        for pos in positions:
            price = row["close"]
            if pos["direction"] == "LONG":
                if price <= pos["stop_loss"]:
                    pos["close_reason"] = "stop_loss"
                    pos["exit_price"] = pos["stop_loss"]
                    closed.append(pos)
                elif price >= pos["take_profit"]:
                    pos["close_reason"] = "take_profit"
                    pos["exit_price"] = pos["take_profit"]
                    closed.append(pos)
                else:
                    remaining.append(pos)
            else:  # SHORT
                if price >= pos["stop_loss"]:
                    pos["close_reason"] = "stop_loss"
                    pos["exit_price"] = pos["stop_loss"]
                    closed.append(pos)
                elif price <= pos["take_profit"]:
                    pos["close_reason"] = "take_profit"
                    pos["exit_price"] = pos["take_profit"]
                    closed.append(pos)
                else:
                    remaining.append(pos)

        positions[:] = remaining
        return closed

    def _close_position(self, pos: Dict, row: pd.Series) -> float:
        """Calcula PnL de cierre de posición."""
        if pos["direction"] == "LONG":
            pnl = (pos["exit_price"] - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - pos["exit_price"]) * pos["quantity"]
        
        # Restar comisiones
        fees = abs(pnl) * self.commission_pct
        return pnl - fees

    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity_curve: List[float],
        df: pd.DataFrame,
        symbol: str,
    ) -> BacktestResult:
        """Calcula todas las métricas del backtesting."""
        if not trades:
            return BacktestResult(
                symbol=symbol,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_holding_period=timedelta(),
                trades=[],
                equity_curve=[self.initial_capital],
                start_date=df.index[0],
                end_date=df.index[-1],
            )

        # Estadísticas básicas
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        win_rate = len(winning) / len(trades) * 100 if trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Drawdown
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = float(np.max(equity - np.minimum.accumulate(equity)))
        max_drawdown_percent = float(np.max(drawdown))

        # Sharpe Ratio (asumiendo risk-free rate = 2%)
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            excess_returns = returns - 0.02 / 365  # Diario
            sharpe_ratio = float(
                np.sqrt(365) * np.mean(excess_returns) / np.std(excess_returns)
            ) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0.0

        # Promedios
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0

        # Holding period promedio
        holding_periods = [(t.exit_time - t.entry_time) for t in trades]
        avg_holding = sum(holding_periods, timedelta()) / len(holding_periods) if holding_periods else timedelta()

        return BacktestResult(
            symbol=symbol,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_period=avg_holding,
            trades=trades,
            equity_curve=equity_curve,
            start_date=df.index[0],
            end_date=df.index[-1],
        )

    def print_summary(self, result: BacktestResult):
        """Imprime resumen del backtesting."""
        print("=" * 60)
        print(f"RESUMEN DE BACKTESTING - {result.symbol}")
        print("=" * 60)
        print(f"Periodo: {result.start_date.date()} -> {result.end_date.date()}")
        print(f"Capital Inicial: ${self.initial_capital:,.2f}")
        print(f"Capital Final: ${self.initial_capital + result.total_pnl:,.2f}")
        print(f"PnL Total: ${result.total_pnl:,.2f} ({result.total_pnl_percent:+.2f}%)")
        print("-" * 60)
        print(f"Total Operaciones: {result.total_trades}")
        print(f"Ganadas: {result.winning_trades} ({result.win_rate:.1f}%)")
        print(f"Perdidas: {result.losing_trades} ({100 - result.win_rate:.1f}%)")
        print(f"Promedio Ganancia: ${result.avg_win:.2f}")
        print(f"Promedio Pérdida: ${result.avg_loss:.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Holding Promedio: {result.avg_holding_period}")
        print("=" * 60)
