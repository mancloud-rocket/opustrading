"""
Parser para prices.csv de Polymarket.
Maneja multiples formatos de CSV que cambiaron durante la recoleccion.

Formato antiguo (col3 = zona string):
  timestamp, market, minute, ZONE, up, down, spread, high_up, high_down, position, extra, extra

Formato nuevo (col3 = precio float):
  timestamp, market, minute, up, down, spread, leader, crossovers, position, event
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ET_UTC_OFFSET_HOURS


# ---------------------------------------------------------------------------
# Utilidades de parseo
# ---------------------------------------------------------------------------

def parse_minute_str(minute_str: str) -> float:
    """Convierte 'M:SS' o 'MM:SS' a segundos totales."""
    parts = minute_str.strip().split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return -1.0
    return -1.0


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_row(parts: List[str]) -> Optional[Dict]:
    """
    Parsea una linea del CSV. Devuelve dict con:
      timestamp (str), market (str), elapsed_seconds (float),
      up_price (float), down_price (float)
    o None si la linea no es valida.
    """
    if len(parts) < 7:
        return None
    try:
        ts_str = parts[0].strip()
        market = parts[1].strip()
        minute_str = parts[2].strip()

        elapsed = parse_minute_str(minute_str)
        if elapsed < 0:
            return None

        col3 = parts[3].strip()

        if _is_float(col3):
            # Formato nuevo: col3=up, col4=down
            up = float(col3)
            down = float(parts[4].strip())
        else:
            # Formato antiguo: col3=zona, col4=up, col5=down
            if len(parts) < 8:
                return None
            up = float(parts[4].strip())
            down = float(parts[5].strip())

        if not (0.0 <= up <= 1.0 and 0.0 <= down <= 1.0):
            return None

        return {
            "timestamp": ts_str,
            "market": market,
            "elapsed_seconds": elapsed,
            "up_price": up,
            "down_price": down,
        }
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Parseo del nombre de mercado -> timestamps UTC
# ---------------------------------------------------------------------------

_MARKET_RE = re.compile(
    r"Bitcoin Up or Down - (February \d+)[;,]\s*"
    r"(\d{1,2}:\d{2}[AP]M)\s*-\s*(\d{1,2}:\d{2}[AP]M)\s*ET",
    re.IGNORECASE,
)


def parse_market_times(name: str, year: int = 2026) -> Optional[Dict]:
    """
    Extrae start/end UTC de un nombre de mercado.
    Ej: 'Bitcoin Up or Down - February 6; 3:15PM-3:30PM ET'
    Devuelve dict con start_utc, end_utc (datetime aware).
    """
    m = _MARKET_RE.search(name)
    if not m:
        return None

    date_str = m.group(1)   # "February 6"
    start_s = m.group(2)    # "3:15PM"
    end_s = m.group(3)      # "3:30PM"

    base = datetime.strptime(f"{date_str} {year}", "%B %d %Y")
    st = datetime.strptime(start_s, "%I:%M%p")
    et = datetime.strptime(end_s, "%I:%M%p")

    start_et = base.replace(hour=st.hour, minute=st.minute, second=0)
    end_et = base.replace(hour=et.hour, minute=et.minute, second=0)

    if end_et <= start_et:
        end_et += timedelta(days=1)

    tz_et = timezone(timedelta(hours=ET_UTC_OFFSET_HOURS))
    start_utc = start_et.replace(tzinfo=tz_et).astimezone(timezone.utc)
    end_utc = end_et.replace(tzinfo=tz_et).astimezone(timezone.utc)

    return {"start_utc": start_utc, "end_utc": end_utc}


# ---------------------------------------------------------------------------
# Carga principal
# ---------------------------------------------------------------------------

def load_markets(csv_path: str) -> Dict[str, pd.DataFrame]:
    """
    Lee prices.csv y devuelve {market_name: DataFrame}.
    Cada DataFrame tiene columnas:
      timestamp (datetime64 UTC), elapsed_seconds, up_price, down_price
    Ordenado por timestamp.
    """
    rows: List[Dict] = []
    errors = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("timestamp,"):
                continue
            parts = line.split(",")
            parsed = parse_row(parts)
            if parsed:
                rows.append(parsed)
            else:
                errors += 1

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    markets: Dict[str, pd.DataFrame] = {}
    for name, grp in df.groupby("market"):
        mdf = grp.sort_values("timestamp").reset_index(drop=True)
        # Eliminar duplicados por elapsed_seconds (quedarse con el primero)
        mdf = mdf.drop_duplicates(subset=["elapsed_seconds"], keep="first")
        markets[name] = mdf

    return markets


def determine_resolution(market_df: pd.DataFrame) -> Optional[str]:
    """
    Determina si gano UP o DOWN basandose en los ultimos ticks.
    Devuelve 'UP', 'DOWN', o None si no se puede determinar.
    """
    if len(market_df) < 20:
        return None

    n = max(30, int(len(market_df) * 0.05))
    tail = market_df.tail(n)

    avg_up = tail["up_price"].mean()
    avg_down = tail["down_price"].mean()
    max_elapsed = tail["elapsed_seconds"].max()

    # Resolucion clara: un lado domina
    if avg_up > 0.80 and avg_down < 0.25:
        return "UP"
    if avg_down > 0.80 and avg_up < 0.25:
        return "DOWN"

    # Cerca de la resolucion (minuto >= 13) y un lado lidera
    if max_elapsed >= 780:
        if avg_up > avg_down + 0.20:
            return "UP"
        if avg_down > avg_up + 0.20:
            return "DOWN"

    return None


def build_market_summary(markets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye un resumen de todos los mercados:
    nombre, start/end UTC, num_ticks, resolucion, rango de minutos cubiertos.
    """
    rows = []
    for name, mdf in markets.items():
        times = parse_market_times(name)
        res = determine_resolution(mdf)
        rows.append({
            "market": name,
            "start_utc": times["start_utc"] if times else None,
            "end_utc": times["end_utc"] if times else None,
            "num_ticks": len(mdf),
            "min_second": mdf["elapsed_seconds"].min(),
            "max_second": mdf["elapsed_seconds"].max(),
            "resolution_poly": res,
        })
    return pd.DataFrame(rows)

