"""
Motor de features para el modelo predictivo.

Calcula features desde:
  1. Datos de Polymarket (precios UP/DOWN intra-mercado)
  2. Datos de Binance (precio BTC, retornos, volumen)

Todos los features se computan de manera causal (sin data leakage):
solo usan informacion disponible hasta el instante t.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Features de Polymarket (solo datos del mercado)
# ---------------------------------------------------------------------------

def compute_polymarket_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dado un DataFrame de un mercado con columnas:
      timestamp, elapsed_seconds, up_price, down_price
    Devuelve el mismo DataFrame con columnas de features adicionales.
    """
    df = market_df.copy()
    up = df["up_price"].values
    down = df["down_price"].values
    elapsed = df["elapsed_seconds"].values

    # --- Features basicas ---
    df["spread"] = up - down
    df["abs_spread"] = np.abs(df["spread"])
    df["leader"] = np.where(up > down, 1, -1)  # 1=UP lidera, -1=DOWN lidera
    df["mid_price"] = (up + down) / 2.0
    df["imbalance"] = (up - down) / np.maximum(up + down, 0.01)

    # --- Tiempo ---
    df["elapsed_frac"] = elapsed / 900.0  # fraccion del mercado transcurrida
    df["time_remaining"] = 900.0 - elapsed
    df["time_remaining_frac"] = df["time_remaining"] / 900.0

    # --- Momentum de precios UP/DOWN (cambio en ventanas) ---
    for window in [5, 10, 20, 50]:
        df[f"up_momentum_{window}"] = df["up_price"].diff(window)
        df[f"down_momentum_{window}"] = df["down_price"].diff(window)
        df[f"spread_momentum_{window}"] = df["spread"].diff(window)

    # --- Velocidad de cambio (derivada primera) ---
    df["up_velocity"] = df["up_price"].diff(1)
    df["down_velocity"] = df["down_price"].diff(1)
    df["spread_velocity"] = df["spread"].diff(1)

    # --- Aceleracion (derivada segunda) ---
    df["up_accel"] = df["up_velocity"].diff(1)
    df["down_accel"] = df["down_velocity"].diff(1)

    # --- Volatilidad rolling ---
    for window in [10, 20, 50]:
        df[f"up_volatility_{window}"] = (
            df["up_price"].rolling(window, min_periods=3).std()
        )
        df[f"down_volatility_{window}"] = (
            df["down_price"].rolling(window, min_periods=3).std()
        )

    # --- Precio relativo al rango historico (dentro del mercado) ---
    df["up_expanding_max"] = df["up_price"].expanding(min_periods=1).max()
    df["up_expanding_min"] = df["up_price"].expanding(min_periods=1).min()
    df["down_expanding_max"] = df["down_price"].expanding(min_periods=1).max()
    df["down_expanding_min"] = df["down_price"].expanding(min_periods=1).min()

    up_range = df["up_expanding_max"] - df["up_expanding_min"]
    df["up_range_position"] = np.where(
        up_range > 0.001,
        (df["up_price"] - df["up_expanding_min"]) / up_range,
        0.5,
    )

    # --- Crossover detection ---
    leader_changes = df["leader"].diff().abs()
    df["crossover_event"] = (leader_changes > 0).astype(int)
    df["crossovers_cumsum"] = df["crossover_event"].cumsum()

    # --- Duracion del lider actual (en ticks) ---
    # Contar cuantos ticks consecutivos lleva el lider actual
    leader_blocks = (df["leader"] != df["leader"].shift(1)).cumsum()
    df["leader_duration_ticks"] = df.groupby(leader_blocks).cumcount() + 1

    # --- Media movil del precio (suavizado) ---
    for window in [10, 30]:
        df[f"up_sma_{window}"] = (
            df["up_price"].rolling(window, min_periods=1).mean()
        )
        df[f"down_sma_{window}"] = (
            df["down_price"].rolling(window, min_periods=1).mean()
        )

    # --- Distancia al 50/50 ---
    df["dist_from_even"] = np.abs(df["up_price"] - 0.50)

    # --- RSI simplificado sobre up_price ---
    delta = df["up_price"].diff(1)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=3).mean()
    avg_loss = loss.rolling(14, min_periods=3).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    df["up_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    return df


# ---------------------------------------------------------------------------
# Features de Binance (precio BTC externo)
# ---------------------------------------------------------------------------

def compute_binance_features(
    btc_df: pd.DataFrame,
    market_start_utc: pd.Timestamp,
    market_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    """
    Dado un DataFrame de Binance con columnas:
      open_time, btc_close, log_return
    y el rango del mercado, calcula features de BTC.
    Devuelve DataFrame indexado por open_time.
    """
    df = btc_df.copy()

    # Precio al inicio del mercado (referencia)
    mask_start = df["open_time"] <= market_start_utc
    if mask_start.any():
        btc_at_start = df.loc[mask_start, "btc_close"].iloc[-1]
    else:
        btc_at_start = df["btc_close"].iloc[0]

    # Retorno acumulado desde inicio del mercado
    df["btc_return_from_start"] = (df["btc_close"] - btc_at_start) / btc_at_start

    # Retorno es positivo => BTC sube => senial UP
    df["btc_direction_signal"] = np.sign(df["btc_return_from_start"])

    # Momentum multi-escala (en numero de velas)
    for lookback in [1, 3, 5, 10, 15, 30]:
        col = f"btc_return_{lookback}"
        df[col] = df["btc_close"].pct_change(lookback)

    # Volatilidad realizada
    for window in [5, 10, 30]:
        df[f"btc_realized_vol_{window}"] = (
            df["log_return"].rolling(window, min_periods=2).std()
        )

    # VWAP deviation (si tenemos volumen) - usamos close como proxy
    if "volume" in df.columns:
        df["btc_vwap_10"] = (
            (df["btc_close"] * df["volume"]).rolling(10, min_periods=1).sum()
            / df["volume"].rolling(10, min_periods=1).sum()
        )
        df["btc_vwap_deviation"] = (
            (df["btc_close"] - df["btc_vwap_10"]) / df["btc_vwap_10"]
        )

    # Taker buy ratio (si disponible) => presion compradora
    if "taker_buy_vol" in df.columns and "volume" in df.columns:
        df["taker_buy_ratio"] = (
            df["taker_buy_vol"] / np.maximum(df["volume"], 1e-9)
        )
        df["taker_buy_ratio_sma5"] = (
            df["taker_buy_ratio"].rolling(5, min_periods=1).mean()
        )

    # Aceleracion del retorno
    df["btc_return_accel"] = df["btc_return_1"].diff(1) if "btc_return_1" in df.columns else 0.0

    return df


# ---------------------------------------------------------------------------
# Merge de features: Polymarket + Binance alineados por tiempo
# ---------------------------------------------------------------------------

def merge_features(
    poly_df: pd.DataFrame,
    btc_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Alinea features de Binance con ticks de Polymarket usando merge_asof.
    El resultado tiene las columnas de ambos DataFrames, alineadas por timestamp.
    """
    if btc_features_df.empty:
        return poly_df

    poly = poly_df.copy()
    btc = btc_features_df.copy()

    # Asegurar que ambos estan ordenados y con timezone
    poly = poly.sort_values("timestamp")
    btc = btc.sort_values("open_time")

    # merge_asof: para cada tick de Polymarket, tomar el dato de Binance
    # mas reciente (backward)
    btc_cols = [c for c in btc.columns if c not in ("open_time",)]
    btc_merge = btc[["open_time"] + btc_cols].copy()

    merged = pd.merge_asof(
        poly,
        btc_merge,
        left_on="timestamp",
        right_on="open_time",
        direction="backward",
    )

    return merged


# ---------------------------------------------------------------------------
# Lista de nombres de features para el modelo
# ---------------------------------------------------------------------------

POLY_FEATURE_COLS = [
    "up_price", "down_price", "spread", "abs_spread", "imbalance",
    "elapsed_frac", "time_remaining_frac",
    "up_momentum_5", "up_momentum_10", "up_momentum_20", "up_momentum_50",
    "down_momentum_5", "down_momentum_10", "down_momentum_20", "down_momentum_50",
    "spread_momentum_5", "spread_momentum_10", "spread_momentum_20",
    "up_velocity", "down_velocity", "spread_velocity",
    "up_accel", "down_accel",
    "up_volatility_10", "up_volatility_20", "up_volatility_50",
    "down_volatility_10", "down_volatility_20", "down_volatility_50",
    "up_range_position",
    "crossovers_cumsum", "leader_duration_ticks",
    "dist_from_even", "up_rsi_14",
]

BINANCE_FEATURE_COLS = [
    "btc_return_from_start", "btc_direction_signal",
    "btc_return_1", "btc_return_3", "btc_return_5",
    "btc_return_10", "btc_return_15", "btc_return_30",
    "btc_realized_vol_5", "btc_realized_vol_10", "btc_realized_vol_30",
    "btc_vwap_deviation",
    "taker_buy_ratio", "taker_buy_ratio_sma5",
    "btc_return_accel",
]


def get_available_features(df: pd.DataFrame) -> list:
    """Devuelve la lista de columnas de features presentes en el DataFrame."""
    all_feats = POLY_FEATURE_COLS + BINANCE_FEATURE_COLS
    return [c for c in all_feats if c in df.columns]

