"""
Gestor de Base de Datos - Almacena trades, señales, configuraciones y métricas.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Maneja todas las operaciones de base de datos.
    Usa SQLite por defecto (modo desarrollo).
    """

    def __init__(self, db_path: str = "data/trading_bot.db"):
        self.db_path = db_path
        
        # Crear directorio si no existe
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Inicializa la base de datos y crea tablas si no existen."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            self._create_tables()
            logger.info(f"Base de datos inicializada: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise

    def _create_tables(self):
        """Crea todas las tablas necesarias."""
        cursor = self.conn.cursor()

        # Trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL,
                pnl_percent REAL,
                fees REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                reason TEXT,
                strategy TEXT,
                market_type TEXT
            )
        """)

        # Señales generadas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL,
                technical_score REAL,
                sentiment_score REAL,
                fundamental_score REAL,
                reason TEXT,
                was_executed INTEGER DEFAULT 0,
                executed_trade_id TEXT,
                created_at TIMESTAMP
            )
        """)

        # Configuración
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP
            )
        """)

        # Métricas de rendimiento
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                symbol TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                total_pnl REAL,
                max_drawdown REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                capital REAL,
                created_at TIMESTAMP
            )
        """)

        # Historial de errores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                level TEXT,
                module TEXT,
                message TEXT,
                traceback TEXT
            )
        """)

        # Datos de mercado (caché)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp TIMESTAMP,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)

        self.conn.commit()

    # === TRADES ===

    def save_trade(self, trade_data: Dict) -> int:
        """Guarda un trade en la base de datos."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, direction, entry_price, exit_price, quantity,
             stop_loss, take_profit, pnl, pnl_percent, fees, status,
             entry_time, exit_time, reason, strategy, market_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data.get("trade_id"),
            trade_data.get("symbol"),
            trade_data.get("direction"),
            trade_data.get("entry_price"),
            trade_data.get("exit_price"),
            trade_data.get("quantity"),
            trade_data.get("stop_loss"),
            trade_data.get("take_profit"),
            trade_data.get("pnl"),
            trade_data.get("pnl_percent"),
            trade_data.get("fees", 0),
            trade_data.get("status", "open"),
            trade_data.get("entry_time", datetime.now()),
            trade_data.get("exit_time"),
            trade_data.get("reason"),
            trade_data.get("strategy"),
            trade_data.get("market_type"),
        ))
        
        self.conn.commit()
        return cursor.lastrowid

    def update_trade(self, trade_id: str, updates: Dict):
        """Actualiza un trade existente."""
        if not updates:
            return

        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [trade_id]

        cursor = self.conn.cursor()
        cursor.execute(
            f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
            values,
        )
        self.conn.commit()

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Obtiene un trade por su ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

    def get_open_trades(self) -> List[Dict]:
        """Obtiene todos los trades abiertos."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[Dict]:
        """Obtiene historial de trades."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # === SEÑALES ===

    def save_signal(self, signal_data: Dict) -> int:
        """Guarda una señal generada."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO signals
            (signal_id, symbol, direction, price, confidence,
             technical_score, sentiment_score, fundamental_score,
             reason, was_executed, executed_trade_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data.get("signal_id"),
            signal_data.get("symbol"),
            signal_data.get("direction"),
            signal_data.get("price"),
            signal_data.get("confidence"),
            signal_data.get("technical_score"),
            signal_data.get("sentiment_score"),
            signal_data.get("fundamental_score"),
            signal_data.get("reason"),
            signal_data.get("was_executed", 0),
            signal_data.get("executed_trade_id"),
            signal_data.get("created_at", datetime.now()),
        ))
        
        self.conn.commit()
        return cursor.lastrowid

    def mark_signal_executed(self, signal_id: str, trade_id: str):
        """Marca una señal como ejecutada."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE signals SET was_executed = 1, executed_trade_id = ? WHERE signal_id = ?",
            (trade_id, signal_id),
        )
        self.conn.commit()

    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Obtiene señales recientes."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # === CONFIGURACIÓN ===

    def save_config(self, key: str, value: Any):
        """Guarda un valor de configuración."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), datetime.now()),
        )
        self.conn.commit()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        
        if row:
            try:
                return json.loads(row["value"])
            except:
                return row["value"]
        return default

    def get_all_config(self) -> Dict:
        """Obtiene toda la configuración."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM config")
        
        config = {}
        for row in cursor.fetchall():
            try:
                config[row["key"]] = json.loads(row["value"])
            except:
                config[row["key"]] = row["value"]
        
        return config

    # === MÉTRICAS ===

    def save_performance_metrics(self, metrics: Dict):
        """Guarda métricas de rendimiento diarias."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_metrics
            (date, symbol, total_trades, winning_trades, losing_trades,
             win_rate, total_pnl, max_drawdown, profit_factor,
             sharpe_ratio, capital, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.get("date", datetime.now().date()),
            metrics.get("symbol", "ALL"),
            metrics.get("total_trades", 0),
            metrics.get("winning_trades", 0),
            metrics.get("losing_trades", 0),
            metrics.get("win_rate", 0),
            metrics.get("total_pnl", 0),
            metrics.get("max_drawdown", 0),
            metrics.get("profit_factor", 0),
            metrics.get("sharpe_ratio", 0),
            metrics.get("capital", 0),
            datetime.now(),
        ))
        
        self.conn.commit()

    def get_performance_history(
        self, days: int = 30, symbol: Optional[str] = None
    ) -> List[Dict]:
        """Obtiene historial de métricas."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM performance_metrics
            WHERE date >= date('now', '-' || ? || ' days')
        """
        params = [days]
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    # === ERRORES ===

    def log_error(self, level: str, module: str, message: str, traceback: str = ""):
        """Registra un error en la base de datos."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO error_log (timestamp, level, module, message, traceback) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(), level, module, message, traceback),
        )
        self.conn.commit()

    def get_error_log(self, limit: int = 100) -> List[Dict]:
        """Obtiene el log de errores."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM error_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # === DATOS DE MERCADO ===

    def save_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Guarda datos de mercado en caché."""
        cursor = self.conn.cursor()
        
        for idx, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO market_data
                (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                timeframe,
                idx.to_pydatetime(),
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            ))
        
        self.conn.commit()

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Obtiene datos de mercado cacheados."""
        query = """
            SELECT * FROM market_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, timeframe]

        if from_date:
            query += " AND timestamp >= ?"
            params.append(from_date)
        if to_date:
            query += " AND timestamp <= ?"
            params.append(to_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, self.conn, params=params)
        
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        return df

    # === UTILIDADES ===

    def get_statistics(self) -> Dict:
        """Obtiene estadísticas generales de la base de datos."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total de trades
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing,
                COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades
        """)
        row = cursor.fetchone()
        if row:
            stats["trades"] = dict(row)
        
        # Señales recientes
        cursor.execute("SELECT COUNT(*) as total FROM signals")
        row = cursor.fetchone()
        if row:
            stats["signals"] = dict(row)
        
        # Errores recientes (24h)
        cursor.execute("""
            SELECT COUNT(*) as errors_24h
            FROM error_log
            WHERE timestamp >= datetime('now', '-1 day')
        """)
        row = cursor.fetchone()
        if row:
            stats["errors"] = dict(row)
        
        # Capital actual
        cursor.execute("""
            SELECT capital FROM performance_metrics
            ORDER BY date DESC LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            stats["last_capital"] = row["capital"]
        
        return stats

    def close(self):
        """Cierra la conexión a la base de datos."""
        if self.conn:
            self.conn.close()
            logger.info("Conexión a base de datos cerrada")
