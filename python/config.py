"""
OPUS Trading Bot v3.0 - Configuracion central.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRICES_CSV = DATA_DIR / "prices.csv"
TRADES_CSV = DATA_DIR / "trades.csv"
CACHE_DIR = Path(__file__).parent / "cache"

# --- Binance ---
BINANCE_REST_URL = "https://api.binance.com"
BINANCE_SYMBOL = "BTCUSDT"

# --- Polymarket fees ---
FEE_RATE = 0.02       # 2% por lado
BET_SIZE = 20.00       # dolares por trade

# --- Estrategia ---
MIN_EDGE_THRESHOLD = 0.05   # 5c minimo de edge para entrar
KELLY_FRACTION = 0.25       # 25% del Kelly completo
MAX_ENTRIES_PER_MARKET = 2
ENTRY_SECOND_MIN = 2 * 60   # minuto 2 (120s)
ENTRY_SECOND_MAX = 10 * 60  # minuto 10 (600s)

# --- Timezone ---
ET_UTC_OFFSET_HOURS = -5    # EST (febrero)

# --- Mercado ---
MARKET_DURATION_SECONDS = 900  # 15 minutos

