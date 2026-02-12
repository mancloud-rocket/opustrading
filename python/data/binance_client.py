"""
Cliente para descargar datos historicos de Binance (klines).
Usa la API REST publica, sin API key.
Cache en disco para no re-descargar.
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from config import BINANCE_REST_URL, BINANCE_SYMBOL, CACHE_DIR


def download_klines(
    start_utc: datetime,
    end_utc: datetime,
    symbol: str = BINANCE_SYMBOL,
    interval: str = "1m",
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Descarga klines historicas de Binance.
    Retorna DataFrame con columnas:
      open_time, open, high, low, close, volume, close_time,
      quote_volume, trades, taker_buy_vol, taker_buy_quote_vol
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = (
        f"{symbol}_{interval}"
        f"_{int(start_utc.timestamp())}_{int(end_utc.timestamp())}"
    )
    cache_file = cache_dir / f"{cache_key}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    all_klines = []
    cur_start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)

    while cur_start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur_start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(
            f"{BINANCE_REST_URL}/api/v3/klines",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        klines = resp.json()

        if not klines:
            break

        all_klines.extend(klines)
        cur_start_ms = int(klines[-1][6]) + 1  # close_time + 1ms
        time.sleep(0.12)  # cortesia con rate limits

    if not all_klines:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_vol", "taker_buy_quote_vol", "_ignore",
    ]
    df = pd.DataFrame(all_klines, columns=cols).drop(columns=["_ignore"])

    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_vol", "taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df.to_parquet(cache_file)
    return df


def get_btc_price_at(
    klines_df: pd.DataFrame,
    target_utc,
) -> Optional[float]:
    """
    Obtiene el precio BTC mas cercano a un timestamp dado.
    Usa la columna 'close' de la vela mas cercana.
    """
    if klines_df.empty:
        return None

    target = pd.Timestamp(target_utc)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")

    # Asegurar que open_time tiene tz
    ot = klines_df["open_time"]
    if ot.dt.tz is None:
        ot = ot.dt.tz_localize("UTC")

    idx = ot.searchsorted(target)
    idx = min(idx, len(klines_df) - 1)
    return float(klines_df.iloc[idx]["close"])


def determine_resolution_binance(
    klines_df: pd.DataFrame,
    start_utc,
    end_utc,
    verbose: bool = False,
) -> Optional[str]:
    """
    Determina si BTC subio o bajo entre start y end.
    Retorna 'UP', 'DOWN', o None.
    """
    try:
        p_start = get_btc_price_at(klines_df, start_utc)
        p_end = get_btc_price_at(klines_df, end_utc)

        if p_start is None or p_end is None:
            if verbose:
                print(f"      [BnRes] start={start_utc} end={end_utc} "
                      f"-> p_start={p_start} p_end={p_end} -> None")
            return None

        if verbose:
            pct = (p_end - p_start) / p_start * 100
            direction = "UP" if p_end > p_start else ("DOWN" if p_end < p_start else "FLAT")
            print(f"      [BnRes] BTC {p_start:.2f} -> {p_end:.2f} "
                  f"({pct:+.4f}%) => {direction}")

        if p_end > p_start:
            return "UP"
        elif p_end < p_start:
            return "DOWN"
        return None
    except Exception as e:
        if verbose:
            print(f"      [BnRes] ERROR: {e}")
        return None


def get_btc_returns_series(
    klines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula retornos logaritmicos por cada vela.
    Retorna DataFrame con open_time, btc_close y log_return.
    """
    if klines_df.empty:
        return pd.DataFrame(columns=["open_time", "btc_close", "log_return"])

    df = klines_df[["open_time", "close"]].copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna().reset_index(drop=True)
    df = df.rename(columns={"close": "btc_close"})
    return df
