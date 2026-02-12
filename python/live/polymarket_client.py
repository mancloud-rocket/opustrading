"""
Polymarket API client para descubrimiento y precios de mercados BTC 15-min.

APIs:
  - Gamma: https://gamma-api.polymarket.com  (descubrimiento de mercados)
  - CLOB:  https://clob.polymarket.com       (precios / order book)
"""

import time
import math
import json
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class MarketInfo:
    """Informacion de un mercado activo."""
    title: str
    slug: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    start_time: datetime       # UTC
    end_time: datetime         # UTC
    discovered_at: float = 0.0  # time.time()

    @property
    def elapsed_seconds(self) -> float:
        now = datetime.now(timezone.utc)
        return (now - self.start_time).total_seconds()

    @property
    def remaining_seconds(self) -> float:
        return 900.0 - self.elapsed_seconds

    @property
    def is_expired(self) -> bool:
        return self.elapsed_seconds >= 900.0

    @property
    def market_minute(self) -> float:
        return self.elapsed_seconds / 60.0


@dataclass
class PriceSnapshot:
    """Snapshot de precios de un mercado."""
    up: Optional[float] = None
    down: Optional[float] = None
    timestamp: float = 0.0
    market_closed: bool = False
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return (
            self.up is not None
            and self.down is not None
            and 0.01 <= self.up <= 0.99
            and 0.01 <= self.down <= 0.99
        )

    @property
    def spread(self) -> float:
        if self.up is None or self.down is None:
            return 0.0
        return abs(self.up - self.down)

    @property
    def leader(self) -> str:
        if self.up is None or self.down is None:
            return "NONE"
        if self.up > self.down:
            return "UP"
        elif self.down > self.up:
            return "DOWN"
        return "TIE"


class PolymarketClient:
    """
    Cliente para Polymarket APIs.
    Descubre mercados BTC 15-min y obtiene precios.
    """

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self._current_market: Optional[MarketInfo] = None
        self._consecutive_errors = 0

    @property
    def current_market(self) -> Optional[MarketInfo]:
        return self._current_market

    # ------------------------------------------------------------------
    # Slug generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_current_slug() -> str:
        """
        Genera el slug del mercado activo actual.
        Los mercados empiezan cada 15 minutos: :00, :15, :30, :45
        Slug format: btc-updown-15m-{unix_timestamp}
        """
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        market_start = now.replace(minute=minutes, second=0, microsecond=0)
        ts = int(market_start.timestamp())
        return f"btc-updown-15m-{ts}"

    @staticmethod
    def generate_next_slug() -> Tuple[str, datetime]:
        """
        Genera el slug del PROXIMO mercado (que aun no ha empezado).
        Retorna (slug, start_time_utc).
        """
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        current_start = now.replace(minute=minutes, second=0, microsecond=0)
        from datetime import timedelta
        next_start = current_start + timedelta(minutes=15)
        ts = int(next_start.timestamp())
        return f"btc-updown-15m-{ts}", next_start

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def find_current_market(self, retries: int = 3) -> Optional[MarketInfo]:
        """
        Busca el mercado BTC 15-min activo actual en Polymarket.
        Retorna MarketInfo o None si no se encuentra.
        """
        slug = self.generate_current_slug()

        for attempt in range(1, retries + 1):
            try:
                # Buscar evento por slug
                resp = self._session.get(
                    f"{GAMMA_API}/events",
                    params={"slug": slug},
                    timeout=10,
                )
                resp.raise_for_status()
                events = resp.json()

                if not events or len(events) == 0:
                    if attempt < retries:
                        time.sleep(1)
                    continue

                event = events[0]
                markets = event.get("markets", [])
                if not markets:
                    if attempt < retries:
                        time.sleep(1)
                    continue

                market = markets[0]

                # Parsear token IDs
                clob_token_ids = market.get("clobTokenIds", [])
                if isinstance(clob_token_ids, str):
                    clob_token_ids = json.loads(clob_token_ids)

                outcomes = market.get("outcomes", [])
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)

                # Encontrar indices UP y DOWN
                up_idx = None
                down_idx = None
                for i, o in enumerate(outcomes):
                    if o.lower() == "up":
                        up_idx = i
                    elif o.lower() == "down":
                        down_idx = i

                if up_idx is None or down_idx is None:
                    if attempt < retries:
                        time.sleep(1)
                    continue

                if len(clob_token_ids) < 2:
                    if attempt < retries:
                        time.sleep(1)
                    continue

                # Timestamp del slug -> start time
                slug_ts = int(slug.split("-")[-1])
                start_time = datetime.fromtimestamp(slug_ts, tz=timezone.utc)
                from datetime import timedelta
                end_time = start_time + timedelta(seconds=900)

                market_info = MarketInfo(
                    title=event.get("title", market.get("question", f"Market {slug}")),
                    slug=slug,
                    condition_id=market.get("conditionId", ""),
                    up_token_id=clob_token_ids[up_idx],
                    down_token_id=clob_token_ids[down_idx],
                    start_time=start_time,
                    end_time=end_time,
                    discovered_at=time.time(),
                )

                self._current_market = market_info
                self._consecutive_errors = 0
                return market_info

            except Exception:
                if attempt < retries:
                    time.sleep(1)

        return None

    # ------------------------------------------------------------------
    # Price fetching
    # ------------------------------------------------------------------

    def fetch_price(self, token_id: str) -> Tuple[Optional[float], bool]:
        """
        Obtiene el precio de un token via CLOB API.
        Retorna (price, market_closed).
        """
        try:
            resp = self._session.get(
                f"{CLOB_API}/price",
                params={"token_id": token_id, "side": "buy"},
                timeout=5,
            )

            if resp.status_code in (404, 422):
                return None, True  # Market closed

            resp.raise_for_status()
            data = resp.json()
            price = float(data.get("price", 0))
            if 0.01 <= price <= 0.99:
                return price, False
            return None, False

        except Exception:
            return None, False

    def fetch_prices(self) -> PriceSnapshot:
        """
        Obtiene precios UP y DOWN del mercado actual.
        Retorna PriceSnapshot.
        """
        if self._current_market is None:
            return PriceSnapshot(error="No active market")

        up_price, up_closed = self.fetch_price(self._current_market.up_token_id)
        down_price, down_closed = self.fetch_price(self._current_market.down_token_id)

        closed = up_closed or down_closed

        return PriceSnapshot(
            up=up_price,
            down=down_price,
            timestamp=time.time(),
            market_closed=closed,
        )

    # ------------------------------------------------------------------
    # Market transition
    # ------------------------------------------------------------------

    def check_market_transition(self) -> bool:
        """
        Verifica si necesitamos cambiar de mercado.
        Retorna True si se encontro nuevo mercado.
        """
        current_slug = self.generate_current_slug()

        if self._current_market is None:
            return self.find_current_market() is not None

        if current_slug != self._current_market.slug:
            # Nuevo periodo de 15 minutos
            self._current_market = None
            return self.find_current_market() is not None

        if self._current_market.is_expired:
            self._current_market = None
            return self.find_current_market() is not None

        return False

    def is_healthy(self) -> bool:
        return self._consecutive_errors < 10

    def seconds_until_next_market(self) -> float:
        """Segundos hasta que empiece el proximo mercado."""
        now = datetime.now(timezone.utc)
        minutes = (now.minute // 15) * 15
        current_start = now.replace(minute=minutes, second=0, microsecond=0)
        from datetime import timedelta
        next_start = current_start + timedelta(minutes=15)
        return (next_start - now).total_seconds()

