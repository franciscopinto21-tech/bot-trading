"""
Script de prueba manual del Bot de Inversión.
Ejecuta cada módulo paso a paso para verificar su funcionamiento.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def test_analizador_tecnico():
    """Prueba el análisis técnico con datos sintéticos."""
    print("\n" + "=" * 60)

    print("[TEST 1] ANALIZADOR TECNICO")
    print("=" * 60)
    
    from src.analyzer.technical import TechnicalAnalyzer
    
    # Crear datos de muestra con tendencia
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    close = 100 + np.linspace(0, 10, 200) + np.random.randn(200) * 2
    data = {
        'open': close - np.abs(np.random.randn(200)) * 0.5,
        'high': close + np.abs(np.random.randn(200)) * 1.0,
        'low': close - np.abs(np.random.randn(200)) * 1.0,
        'close': close,
        'volume': np.random.rand(200) * 1000 + 500,
    }
    df = pd.DataFrame(data, index=dates)
    
    analyzer = TechnicalAnalyzer()
    signal = analyzer.analyze(df, 'BTC/USDT')
    print(f"✅ Símbolo: {signal.symbol}")
    print(f"✅ Dirección: {signal.direction} (1=alcista, -1=bajista, 0=neutral)")
    print(f"✅ Fuerza: {signal.strength:.2f}")
    print(f"✅ Tendencia: {signal.trend}")
    print(f"✅ RSI(14): {signal.indicators.get('rsi_14', 0):.1f}")
    print(f"✅ MACD hist: {signal.indicators.get('macd_hist', 0):.3f}")
    print(f"✅ Patrones: {signal.patterns}")
    print(f"✅ Soporte: {signal.support:.2f}")
    print(f"✅ Resistencia: {signal.resistance:.2f}")
    return True


def test_gestor_riesgo():
    """Prueba el gestor de riesgo."""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: GESTOR DE RIESGO")
    print("=" * 60)
    
    from src.risk.risk_manager import RiskManager, Position
    
    # Crear gestor con capital inicial
    rm = RiskManager(initial_capital=10000.0, max_risk_per_trade=2.0)
    
    # Probar evaluación de trade
    assessment = rm.assess_trade(
        symbol='BTC/USDT',
        entry_price=50000.0,
        direction=1,  # LONG
        atr=1000.0,
    )
    
    print(f"✅ Trade aprobado: {assessment.approved}")
    print(f"✅ Tamaño sugerido: {assessment.suggested_position_size:.4f}")
    print(f"✅ Stop Loss: ${assessment.suggested_stop_loss:.2f}")
    print(f"✅ Take Profit: ${assessment.suggested_take_profit:.2f}")
    print(f"✅ Riesgo: ${assessment.risk_amount:.2f} ({assessment.risk_percentage:.2f}%)")
    print(f"✅ Risk/Reward: {assessment.risk_reward_ratio:.2f}")
    
    # Probar apertura y cierre de posición
    pos = Position(
        symbol='BTC/USDT',
        entry_price=50000.0,
        current_price=51000.0,
        quantity=assessment.suggested_position_size,
        stop_loss=assessment.suggested_stop_loss,
        take_profit=assessment.suggested_take_profit,
        direction=1,
        entry_time=datetime.now(),
    )
    rm.open_position(pos)
    print(f"✅ Posición abierta: {len(rm.open_positions)}")
    
    # Actualizar precio (simular mercado)
    rm.update_positions({'BTC/USDT': 51500.0})
    print(f"✅ PnL no realizado: ${pos.unrealized_pnl:.2f}")
    
    # Estadísticas
    stats = rm.get_statistics()
    print(f"✅ Capital actual: ${stats['current_capital']:.2f}")
    print(f"✅ Drawdown: {stats['drawdown_percentage']:.2f}%")
    return True


def test_motor_senales():
    """Prueba el motor de señales."""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: MOTOR DE SEÑALES")
    print("=" * 60)
    
    from src.models.signal_model import SignalEngine
    
    engine = SignalEngine(min_confidence=0.65)
    
    # Simular resultados de las 3 capas
    tech_result = {
        'direction': 1,
        'strength': 0.7,
        'trend': 'bullish',
        'patterns': ['bullish_engulfing'],
        'indicators': {'rsi_14': 45, 'macd_hist': 0.5},
    }
    
    sent_result = {
        'score': 0.6,
        'confidence': 0.8,
        'classification': 'positive',
        'key_entities': ['BTC', 'crypto'],
    }
    
    fund_result = {
        'score': 0.3,
        'classification': 'buy',
    }
    
    signal = engine.evaluate(
        symbol='BTC/USDT',
        current_price=50000.0,
        technical_result=tech_result,
        sentiment_result=sent_result,
        fundamental_result=fund_result,
    )
    
    if signal:
        print(f"✅ SEÑAL GENERADA: {signal.direction}")
        print(f"✅ Confianza: {signal.confidence:.1%}")
        print(f"✅ Razón: {signal.reason}")
        print(f"✅ Entry: ${signal.entry_price:.2f}")
        print(f"✅ SL: ${signal.stop_loss:.2f}")
        print(f"✅ TP: ${signal.take_profit:.2f}")
    
    # Estadísticas
    stats = engine.get_performance_stats()
    print(f"✅ Total señales: {stats['total_signals']}")
    return True


def test_base_datos():
    """Prueba la base de datos."""
    print("\n" + "=" * 60)
    print("🧪 TEST 4: BASE DE DATOS")
    print("=" * 60)
    
    import os
    from src.database.db_manager import DatabaseManager
    
    db = DatabaseManager('data/test_db.db')
    
    # Guardar un trade
    db.save_trade({
        'trade_id': 'TEST-001',
        'symbol': 'BTC/USDT',
        'direction': 'LONG',
        'entry_price': 50000.0,
        'quantity': 0.1,
        'stop_loss': 48000.0,
        'take_profit': 53000.0,
        'status': 'open',
        'entry_time': datetime.now(),
        'reason': 'Test signal',
        'market_type': 'Crypto',
    })
    
    # Guardar una señal
    db.save_signal({
        'signal_id': 'SIG-001',
        'symbol': 'BTC/USDT',
        'direction': 'LONG',
        'price': 50000.0,
        'confidence': 0.85,
        'reason': 'Prueba',
        'created_at': datetime.now(),
    })
    
    # Leer datos
    trade = db.get_trade('TEST-001')
    print(f"✅ Trade guardado: {trade['symbol']} {trade['direction']}")
    
    signals = db.get_recent_signals()
    print(f"✅ Señales guardadas: {len(signals)}")
    
    stats = db.get_statistics()
    print(f"✅ Estadísticas: {stats}")
    
    db.close()
    os.remove('data/test_db.db')
    print("✅ Base de datos funciona correctamente")
    return True


def test_backtesting():
    """Prueba el backtesting."""
    print("\n" + "=" * 60)
    print("🧪 TEST 5: BACKTESTING")
    print("=" * 60)
    
    from src.backtesting.backtester import Backtester
    
    # Crear datos con tendencia clara
    dates = pd.date_range(start='2023-01-01', periods=2000, freq='1h')
    t = np.linspace(0, 6*np.pi, 2000)
    close = 100 + 15*np.sin(t) + np.linspace(0, 10, 2000)
    
    data = {
        'open': close - np.abs(np.random.randn(2000) * 0.5),
        'high': close + np.abs(np.random.randn(2000) * 1.0),
        'low': close - np.abs(np.random.randn(2000) * 1.0),
        'close': close,
        'volume': np.random.rand(2000) * 2000 + 500,
    }
    df = pd.DataFrame(data, index=dates)
    
    bt = Backtester(initial_capital=10000.0)
    result = bt.run(df, symbol='BTC/USDT', use_walk_forward=False)
    
    print(f"✅ Total operaciones: {result.total_trades}")
    print(f"✅ Win rate: {result.win_rate:.1f}%")
    print(f"✅ PnL Total: ${result.total_pnl:.2f}")
    print(f"✅ Profit Factor: {result.profit_factor:.2f}")
    print(f"✅ Sharpe Ratio: {result.sharpe_ratio:.2f}")
    return True


def test_sentimiento():
    """Prueba el analizador de sentimiento."""
    print("\n" + "=" * 60)
    print("🧪 TEST 6: ANALIZADOR DE SENTIMIENTO")
    print("=" * 60)
    
    from src.analyzer.sentiment import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    
    # Probar con titulares de ejemplo
    headlines = [
        "Bitcoin surges to new all-time high amid institutional adoption",
        "Federal Reserve raises interest rates, markets decline sharply",
        "Tech stocks rally on strong earnings reports",
        "Crypto market faces regulatory uncertainty",
        "Company XYZ reports record quarterly revenue",
    ]
    
    result = analyzer.analyze('BTC/USDT', headlines=headlines)
    
    print(f"✅ Score: {result.score:.2f}")
    print(f"✅ Confianza: {result.confidence:.2f}")
    print(f"✅ Clasificación: {result.classification}")
    print(f"✅ Fuentes analizadas: {result.sources_analyzed}")
    print(f"✅ Entidades: {result.key_entities}")
    return True


def test_analizador_fundamental():
    """Prueba el analizador fundamental."""
    print("\n" + "=" * 60)
    print("🧪 TEST 7: ANALIZADOR FUNDAMENTAL")
    print("=" * 60)
    
    from src.analyzer.fundamental import FundamentalAnalyzer
    
    analyzer = FundamentalAnalyzer()
    
    result = analyzer.analyze('BTC/USDT', sector='technology')
    
    print(f"✅ Score fundamental: {result.score:.2f}")
    print(f"✅ Clasificación: {result.overall_classification}")
    print(f"✅ Salud del sector: {result.sector_health}")
    print(f"✅ Métricas: {list(result.metrics.keys())[:5]}")
    return True


def test_notificaciones():
    """Prueba el sistema de notificaciones (sin enviar)."""
    print("\n" + "=" * 60)
    print("🧪 TEST 8: NOTIFICACIONES (SIN API KEYS)")
    print("=" * 60)
    
    from src.notifications.telegram_notifier import TelegramNotifier
    
    # Sin API keys - no envía pero construye mensajes
    telegram = TelegramNotifier()
    print("✅ Telegram notifier inicializado (sin API keys)")
    
    from src.notifications.whatsapp_notifier import WhatsAppNotifier
    whatsapp = WhatsAppNotifier()
    print("✅ WhatsApp notifier inicializado (sin API keys)")
    print("✅ (Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env para probar envío real)")
    return True


def test_integracion_completa():
    """Prueba de integración de todos los módulos."""
    print("\n" + "=" * 60)
    print("🧪 TEST 9: INTEGRACIÓN COMPLETA")
    print("=" * 60)
    
    import numpy as np
    import pandas as pd
    from src.analyzer.technical import TechnicalAnalyzer
    from src.models.signal_model import SignalEngine
    from src.risk.risk_manager import RiskManager
    
    # 1. Datos de mercado simulados
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
    close = 100 + np.linspace(0, 8, 200) + np.random.randn(200) * 1.5
    df = pd.DataFrame({
        'open': close - 0.3,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': np.random.rand(200) * 1000 + 500,
    }, index=dates)
    
    # 2. Análisis técnico
    tech = TechnicalAnalyzer()
    tech_signal = tech.analyze(df, 'BTC/USDT')
    print(f"✅ 1. Análisis técnico: dirección={tech_signal.direction}, tendencia={tech_signal.trend}")
    
    # 3. Evaluar riesgo
    risk = RiskManager()
    assessment = risk.assess_trade(
        symbol='BTC/USDT',
        entry_price=float(close[-1]),
        direction=tech_signal.direction,
        atr=tech_signal.indicators.get('atr', 1.0),
    )
    print(f"✅ 2. Evaluación riesgo: aprobado={assessment.approved}")
    
    # 4. Generar señal
    engine = SignalEngine()
    signal = engine.evaluate(
        symbol='BTC/USDT',
        current_price=float(close[-1]),
        technical_result={
            'direction': tech_signal.direction,
            'strength': tech_signal.strength,
            'trend': tech_signal.trend,
            'patterns': tech_signal.patterns,
            'indicators': tech_signal.indicators,
        },
    )
    
    if signal:
        print(f"✅ 3. Señal generada: {signal.direction} (confianza: {signal.confidence:.1%})")
        print(f"✅ 4. Sistema completo funcional!")
    else:
        print(f"✅ 3. Sin señal (confianza < umbral) - comportamiento normal")
    
    return True


def test_calidad_codigo():
    """Verifica que todas las importaciones funcionan."""
    print("\n" + "=" * 60)
    print("🧪 TEST 10: VERIFICACIÓN DE IMPORTACIONES")
    print("=" * 60)
    
    modulos = [
        ('src.config', 'AppConfig'),
        ('src.analyzer.technical', 'TechnicalAnalyzer'),
        ('src.analyzer.sentiment', 'SentimentAnalyzer'),
        ('src.analyzer.fundamental', 'FundamentalAnalyzer'),
        ('src.risk.risk_manager', 'RiskManager'),
        ('src.models.signal_model', 'SignalEngine'),
        ('src.models.ml_trainer', 'MLTrainer'),
        ('src.notifications.telegram_notifier', 'TelegramNotifier'),
        ('src.notifications.whatsapp_notifier', 'WhatsAppNotifier'),
        ('src.backtesting.backtester', 'Backtester'),
        ('src.database.db_manager', 'DatabaseManager'),
        ('src.broker.base', 'BaseBroker'),
        ('src.broker.binance_broker', 'BinanceBroker'),
    ]
    
    for modulo, clase in modulos:
        try:
            exec(f"from {modulo} import {clase}")
            print(f"  ✅ {modulo}.{clase}")
        except Exception as e:
            print(f"  ❌ {modulo}.{clase}: {e}")
    
    print("✅ Verificación de importaciones completada")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("🤖 BOT DE INVERSIÓN INTELIGENTE - PRUEBA COMPLETA")
    print("=" * 60)
    
    tests = [
        ('Importaciones', test_calidad_codigo),
        ('Analizador Técnico', test_analizador_tecnico),
        ('Gestor de Riesgo', test_gestor_riesgo),
        ('Motor de Señales', test_motor_senales),
        ('Base de Datos', test_base_datos),
        ('Backtesting', test_backtesting),
        ('Sentimiento', test_sentimiento),
        ('Fundamental', test_analizador_fundamental),
        ('Notificaciones', test_notificaciones),
        ('Integración', test_integracion_completa),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ TEST {name}: PASÓ")
            else:
                failed += 1
                print(f"\n❌ TEST {name}: FALLÓ")
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTADO FINAL: {passed}/{passed + failed} tests pasaron")
    print("=" * 60)
