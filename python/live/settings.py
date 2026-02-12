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

# BTC return minimo para generar señal de entrada.
#
# VALIDACION (Session D live, Feb 11 2026, 31 trades):
#   0.030% -> 31T, 61% WR, +$12.75, MaxDD $114 (entra en todo)
#   0.055% -> 27T, 63% WR, +$22.32, MaxDD  $99
#   0.060% -> 17T, 65% WR, +$44.53, MaxDD  $69
#   0.065% -> 13T, 69% WR, +$56.71, MaxDD  $29 <-- OPTIMO
#   0.075% -> 7T,  86% WR, +$76.94, MaxDD   $0 (muy selectivo)
#
# A 0.065%, filtra el ruido (0.03-0.06%) que tiene 50% WR = coin flip.
# Solo entra cuando BTC tiene momentum REAL (>= 0.065%).
BTC_THRESHOLD = 0.00065  # 0.065%


# Take Profit por precio de token.
# Si nuestro token sube a este precio, cerramos con ganancia.
# A 0.97 evitamos el riesgo de reversal de ultimo segundo.
TAKE_PROFIT_PRICE = 0.97

# SL y BTC reversal DESACTIVADOS.
# Tokens tienen volatilidad intra-mercado muy alta (pueden caer 40%
# y resolver a favor). Solo TP a $0.97 + resolution.


# ============================================================================
# ENTRY FILTERS - Filtros de entrada
# ============================================================================

# Rango de precios de token aceptables para compra.
# A minuto 5 con 0.065% threshold, entry tipico: $0.52-$0.60.
# Un entry a $0.53 con TP a $0.97 = +$15.80 neto
# Un entry a $0.60 con TP a $0.97 = +$11.53 neto
# Peor caso (resolucion en contra) pierde ~$20.80
ENTRY_PRICE_MIN = 0.30
ENTRY_PRICE_MAX = 0.60


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
# Helper: exportar como dict (usado por Trader)
# ============================================================================

def get_strategy_config() -> dict:
    """Retorna configuracion como diccionario."""
    return {
        "btc_threshold": BTC_THRESHOLD,
        "take_profit_price": TAKE_PROFIT_PRICE,
        "entry_price_min": ENTRY_PRICE_MIN,
        "entry_price_max": ENTRY_PRICE_MAX,
        "entry_second_min": ENTRY_SECOND_MIN,
        "entry_second_max": ENTRY_SECOND_MAX,
        "bet_size": BET_SIZE,
        "fee_rate": FEE_RATE,
        "max_daily_loss": MAX_DAILY_LOSS,
        "max_trades_per_market": MAX_TRADES_PER_MARKET,
    }
