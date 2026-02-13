"""
Settings editables para el bot live.

Este archivo centraliza TODOS los parametros ajustables del bot.
Para cambiar la estrategia, edita los valores aqui y reinicia el bot.

Los parametros estan organizados por seccion:
  - STRATEGY: parametros de señal y entrada/salida
  - RISK: limites de riesgo y position sizing
  - TIMING: ventanas temporales
  - SYSTEM: configuracion del sistema (polling, etc.)
"""


# ============================================================================
# STRATEGY - Parametros de señal
# ============================================================================

# BTC return minimo para generar señal de entrada (HARD FILTER).
# Aplica tanto para ML como para threshold fallback.
#
# VALIDACION (85 trades reales, Feb 10-12 2026):
#   0.050% -> 78T, 51% WR, -$151 (entra en ruido)
#   0.065% -> 43T, 56% WR, -$ 40 (anterior, sigue perdiendo)
#   0.080% -> 31T, 65% WR, +$ 39 <-- OPTIMO (filtro real)
#   0.100% -> 25T, 60% WR, -$ 16
#   0.150% ->  7T, 86% WR, +$ 46 (muy selectivo)
#
# A 0.065% entramos en ruido (48% reversals, senal mediana 0.062%).
# A 0.080% filtramos el ruido: senal mediana sube a 0.10%, WR a 65%.
BTC_THRESHOLD = 0.00080  # 0.080%


# Take Profit por precio de token.
# Si nuestro token sube a este precio, cerramos con ganancia.
# A 0.97 evitamos el riesgo de reversal de ultimo segundo.
TAKE_PROFIT_PRICE = 0.97

# Stop Loss por caida porcentual del token desde entry.
# 80% = casi desactivado (solo triggers si token baja a 20% del entry).
# Ejemplo: entry $0.60 -> SL se activa a $0.12 (casi nunca antes de resolucion).
TOKEN_STOP_LOSS_PCT = 0.80  # 80% de caida desde entry price (casi desactivado)


# ============================================================================
# ENTRY FILTERS - Filtros de entrada
# ============================================================================

# Rango de precios de token aceptables para compra.
#
# VALIDACION (85 trades reales, Feb 10-12):
#   $0.30-$0.45: 3T, 67% WR, +$26 (pocos trades pero muy rentables)
#   $0.45-$0.55: 22T, 32% WR, -$107 (WR muy bajo)
#   $0.55-$0.65: 45T, 51% WR, -$90 (volumen pero breakeven WR=58%)
#   $0.65-$0.80: 15T, 67% WR, -$8 (WR alto pero gain de $5 vs loss de $12)
#
# Combinacion optima (Threshold 0.08% + Max $0.60 + SL 50%):
#   16T, 69% WR, +$115, MaxDD $22
ENTRY_PRICE_MIN = 0.30
ENTRY_PRICE_MAX = 0.60

# Senal fuerte ya no permite precios mas altos.
# Los datos muestran que entry > $0.60 tiene payout asimetrico negativo.
STRONG_SIGNAL_THRESHOLD = 0.0015  # 0.15% (muy selectivo)
ENTRY_PRICE_MAX_STRONG = 0.65     # max entry con senal muy fuerte


# ============================================================================
# TIMING - Ventanas temporales
# ============================================================================

# Segundos desde inicio del mercado para entrar.
# A minuto 5, BTC ya tiene tendencia establecida.
# Ventana hasta min 10 para capturar oportunidades tardias.
ENTRY_SECOND_MIN = 300   # minuto 5
ENTRY_SECOND_MAX = 600   # minuto 10


# ============================================================================
# RISK - Gestion de riesgo
# ============================================================================

# Tamano de cada apuesta en dolares.
BET_SIZE = 20.00

# Fee de Polymarket por lado (2% = 0.02)
# Round trip = 4% = $0.80 por trade de $20
FEE_RATE = 0.02

# Maximo de perdida diaria en dolares.
# Cuando se alcanza, el bot deja de operar hasta el dia siguiente.
MAX_DAILY_LOSS = 100.00

# Maximo de trades por mercado (15 min window).
MAX_TRADES_PER_MARKET = 1


# ============================================================================
# SYSTEM - Configuracion del sistema
# ============================================================================

# Intervalo de polling en segundos.
POLL_INTERVAL = 1.0

# Cada cuantos segundos verificar transicion de mercado.
MARKET_CHECK_INTERVAL = 10.0

# Cada cuantos segundos imprimir status en consola.
STATUS_INTERVAL = 4.0

# Cada cuantos segundos imprimir dashboard completo.
FULL_STATS_INTERVAL = 300.0

# Logs verbosos
VERBOSE = True


# ============================================================================
# ML PREDICTOR - Parametros del modelo predictivo
# ============================================================================

# Si True, usa el modelo ML entrenado para decisions de entrada.
# Si False, o modelo no existe, usa fallback de thresholds.
USE_ML_PREDICTOR = True

# Confianza minima del modelo para entrar a un trade.
# P(UP) > 0.58 -> comprar UP, P(UP) < 0.42 -> comprar DOWN
# Rango: 0.50 (entra en todo) a 1.00 (nunca entra)
ML_MIN_CONFIDENCE = 0.65

# Path al modelo entrenado (relativo al directorio python/)
ML_MODEL_PATH = "ml/model.pkl"


# ============================================================================
# Helper: exportar como dict (usado por Trader)
# ============================================================================

def get_strategy_config() -> dict:
    """Retorna configuracion como diccionario."""
    return {
        "btc_threshold": BTC_THRESHOLD,
        "take_profit_price": TAKE_PROFIT_PRICE,
        "token_stop_loss_pct": TOKEN_STOP_LOSS_PCT,
        "entry_price_min": ENTRY_PRICE_MIN,
        "entry_price_max": ENTRY_PRICE_MAX,
        "strong_signal_threshold": STRONG_SIGNAL_THRESHOLD,
        "entry_price_max_strong": ENTRY_PRICE_MAX_STRONG,
        "entry_second_min": ENTRY_SECOND_MIN,
        "entry_second_max": ENTRY_SECOND_MAX,
        "bet_size": BET_SIZE,
        "fee_rate": FEE_RATE,
        "max_daily_loss": MAX_DAILY_LOSS,
        "max_trades_per_market": MAX_TRADES_PER_MARKET,
        "use_ml_predictor": USE_ML_PREDICTOR,
        "ml_min_confidence": ML_MIN_CONFIDENCE,
    }
