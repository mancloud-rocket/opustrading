"""
Binance BTC/USDT price feed.
Usa REST API para maxima fiabilidad (polling cada segundo).
Opcionalmente puede usar WebSocket como upgrade futuro.
"""

import time
import threading
import requests
from typing import Optional


BINANCE_REST = "https://api.binance.com"
SYMBOL = "BTCUSDT"


class BinanceFeed:
    """
    Obtiene el precio actual de BTC/USDT desde Binance.
    Thread-safe para uso concurrente.
    """

    def __init__(self):
        self._price: Optional[float] = None
        self._last_update: float = 0.0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._consecutive_errors = 0

    @property
    def price(self) -> Optional[float]:
        with self._lock:
            return self._price

    @property
    def last_update(self) -> float:
        with self._lock:
            return self._last_update

    @property
    def age_seconds(self) -> float:
        with self._lock:
            if self._last_update == 0:
                return float("inf")
            return time.time() - self._last_update

    def fetch_price(self) -> Optional[float]:
        """
        Obtiene el precio actual de BTC/USDT via REST.
        Retorna el precio o None si falla.
        """
        try:
            resp = self._session.get(
                f"{BINANCE_REST}/api/v3/ticker/price",
                params={"symbol": SYMBOL},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            price = float(data["price"])

            with self._lock:
                self._price = price
                self._last_update = time.time()
                self._consecutive_errors = 0

            return price

        except Exception:
            with self._lock:
                self._consecutive_errors += 1
            return None

    def get_price_or_cached(self, max_age: float = 5.0) -> Optional[float]:
        """
        Intenta obtener precio fresco. Si falla, retorna cached si es reciente.
        """
        price = self.fetch_price()
        if price is not None:
            return price

        # Fallback a cache
        if self.age_seconds <= max_age:
            return self.price

        return None

    def fetch_kline_open(self, timestamp_ms: int) -> Optional[float]:
        """
        Obtiene el precio de apertura de la vela de 1 minuto que contiene
        el timestamp dado. Esto nos da el precio BTC al inicio EXACTO
        del minuto, que es lo que Polymarket usa para resolver.
        """
        try:
            resp = self._session.get(
                f"{BINANCE_REST}/api/v3/klines",
                params={
                    "symbol": SYMBOL,
                    "interval": "1m",
                    "startTime": timestamp_ms,
                    "limit": 1,
                },
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                # kline format: [open_time, open, high, low, close, ...]
                return float(data[0][1])  # open price
            return None
        except Exception:
            return None

    def fetch_price_at_market_start(self, market_start_utc) -> Optional[float]:
        """
        Obtiene el precio BTC al inicio exacto del mercado.
        market_start_utc: datetime UTC del inicio del mercado.
        Retorna el precio open de la vela de 1m que contiene ese timestamp.
        """
        try:
            ts_ms = int(market_start_utc.timestamp() * 1000)
            return self.fetch_kline_open(ts_ms)
        except Exception:
            return None

    def is_healthy(self) -> bool:
        """Retorna True si el feed esta funcionando."""
        return self.age_seconds < 10.0 and self._consecutive_errors < 5

