"""
Estrategias de trading para backtest.

Cada estrategia implementa:
  - should_enter(row, history) -> Optional[dict]  (None = no operar)
  - should_exit(row, position) -> Optional[dict]   (None = mantener)

Las estrategias son CAUSALES: solo usan informacion disponible al instante t.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

from config import (
    FEE_RATE, BET_SIZE,
    ENTRY_SECOND_MIN, ENTRY_SECOND_MAX,
    MAX_ENTRIES_PER_MARKET,
)


@dataclass
class Position:
    side: str           # 'UP' o 'DOWN'
    entry_price: float
    entry_second: float
    entry_idx: int
    stop_loss: float
    take_profit: float
    # Context at entry (for reversal stops)
    btc_ret_at_entry: float = 0.0


@dataclass
class TradeResult:
    side: str
    entry_price: float
    exit_price: float
    entry_second: float
    exit_second: float
    reason: str         # TAKE_PROFIT, STOP_LOSS, TIME_STOP, RESOLUTION, BTC_REVERSAL
    pnl_gross: float
    pnl_net: float


def compute_pnl(entry_price: float, exit_price: float) -> tuple:
    """Calcula PnL bruto y neto para un trade."""
    # shares = BET_SIZE / entry_price
    # revenue = shares * exit_price
    # gross = revenue - BET_SIZE
    gross = BET_SIZE * (exit_price - entry_price) / entry_price
    fees = BET_SIZE * FEE_RATE * 2  # entrada + salida
    net = gross - fees
    return gross, net


# ============================================================================
# ESTRATEGIA 1: Cheap Token + Hold to Resolution
# ============================================================================

class CheapTokenHoldStrategy:
    """
    TESIS: Comprar tokens baratos (0.25-0.50) cuando BTC confirma
    la direccion, y MANTENER HASTA RESOLUCION.

    Por que funciona:
    - A 0.35, solo necesitas >35% WR para ser profitable
    - BTC momentum en primeros minutos predice resolucion ~55-60%
    - Sin TP/SL fijos: la resolucion te da 1.00 o 0.00
    - Edge = probabilidad_real - precio_pagado

    Salida SOLO por:
    - Resolucion del mercado (ganas 1.00 o pierdes todo)
    """

    name = "Cheap Token + Hold to Resolution"

    def __init__(
        self,
        min_btc_abs_return: float = 0.0015,  # 0.15% minimo de movimiento BTC
        max_entry_price: float = 0.50,        # no comprar tokens > 50c
        min_entry_price: float = 0.25,        # no comprar tokens < 25c (demasiado contra)
        entry_second_min: float = 180,        # minuto 3 (dar tiempo a BTC)
        entry_second_max: float = 420,        # minuto 7 (tokens aun baratos)
    ):
        self.min_btc_abs_return = min_btc_abs_return
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.entry_second_min = entry_second_min
        self.entry_second_max = entry_second_max

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        if entries_so_far >= 1:  # Solo 1 trade por mercado
            return None

        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < self.entry_second_min or elapsed > self.entry_second_max:
            return None

        btc_ret = row.get("btc_return_from_start")
        if btc_ret is None or (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            return None

        # BTC subiendo => compra UP si esta barato
        if btc_ret > self.min_btc_abs_return:
            price = row.get("up_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "UP",
                    "price": price,
                    "sl": 0.0,   # sin stop loss
                    "tp": 99.0,  # sin take profit (hold to resolution)
                    "btc_ret_at_entry": btc_ret,
                }

        # BTC bajando => compra DOWN si esta barato
        if btc_ret < -self.min_btc_abs_return:
            price = row.get("down_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "DOWN",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)

        # Solo salir en resolucion (minuto ~15)
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}

        return None


# ============================================================================
# ESTRATEGIA 2: Cheap Token + BTC Reversal Stop
# ============================================================================

class CheapTokenReversalStopStrategy:
    """
    Como CheapTokenHold pero con STOP POR REVERSAL DE BTC.

    Si BTC revierte direccion (retorno cambia de signo y supera threshold),
    cortamos la perdida. Esto protege contra:
    - Falsos breakouts de BTC
    - Reversiones violentas

    Pero NO usamos stop-loss por PRECIO de Polymarket, porque:
    - Los precios de Polymarket son ruidosos intra-mercado
    - Un dip temporal de 0.35 a 0.25 no invalida la tesis si BTC sigue up
    """

    name = "Cheap Token + BTC Reversal Stop"

    def __init__(
        self,
        min_btc_abs_return: float = 0.0015,
        max_entry_price: float = 0.50,
        min_entry_price: float = 0.25,
        entry_second_min: float = 180,
        entry_second_max: float = 420,
        reversal_threshold: float = 0.0010,   # BTC debe revertir 0.10% para stop
    ):
        self.min_btc_abs_return = min_btc_abs_return
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.entry_second_min = entry_second_min
        self.entry_second_max = entry_second_max
        self.reversal_threshold = reversal_threshold

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        if entries_so_far >= 1:
            return None

        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < self.entry_second_min or elapsed > self.entry_second_max:
            return None

        btc_ret = row.get("btc_return_from_start")
        if btc_ret is None or (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            return None

        if btc_ret > self.min_btc_abs_return:
            price = row.get("up_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "UP",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        if btc_ret < -self.min_btc_abs_return:
            price = row.get("down_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "DOWN",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)

        # Resolucion
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}

        # BTC reversal check
        btc_ret = row.get("btc_return_from_start")
        if btc_ret is not None and not (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            if pos.side == "UP":
                # Entramos porque BTC subia. Si ahora BTC esta ABAJO, cortar.
                if btc_ret < -self.reversal_threshold:
                    return {"price": current, "reason": "BTC_REVERSAL"}
            elif pos.side == "DOWN":
                # Entramos porque BTC bajaba. Si ahora BTC esta ARRIBA, cortar.
                if btc_ret > self.reversal_threshold:
                    return {"price": current, "reason": "BTC_REVERSAL"}

        return None


# ============================================================================
# ESTRATEGIA 3: Cheap Token + Adaptive Z-Score
# ============================================================================

class CheapTokenZScoreStrategy:
    """
    En vez de un threshold fijo de BTC return, usa Z-SCORE:
    btc_return / volatilidad_reciente > threshold

    Esto se adapta al regimen de volatilidad:
    - En mercado tranquilo: 0.05% puede ser significativo
    - En crash: necesitas 0.5%+ para que sea señal

    Tambien incorpora:
    - Confirmacion de Polymarket (el lider debe ser consistente)
    - Filtro de volatilidad del token (baja vol = mas confiable)
    """

    name = "Cheap Token + Z-Score Adaptive"

    def __init__(
        self,
        min_z_score: float = 1.0,             # BTC return > 1 std dev
        max_entry_price: float = 0.50,
        min_entry_price: float = 0.25,
        entry_second_min: float = 180,
        entry_second_max: float = 420,
        min_leader_duration: int = 10,         # lider al menos 10 ticks
        max_token_vol: float = 0.08,           # volatilidad del token
        reversal_threshold_z: float = -0.5,    # BTC z-score flips
    ):
        self.min_z_score = min_z_score
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.entry_second_min = entry_second_min
        self.entry_second_max = entry_second_max
        self.min_leader_duration = min_leader_duration
        self.max_token_vol = max_token_vol
        self.reversal_threshold_z = reversal_threshold_z

    def _get_btc_zscore(self, row: dict) -> Optional[float]:
        """Calcula z-score del retorno BTC."""
        btc_ret = row.get("btc_return_from_start")
        if btc_ret is None or (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            return None

        btc_vol = row.get("btc_realized_vol_5")
        if btc_vol is None or (isinstance(btc_vol, float) and np.isnan(btc_vol)):
            btc_vol = row.get("btc_realized_vol_10")
        if btc_vol is None or (isinstance(btc_vol, float) and np.isnan(btc_vol)):
            return None
        if btc_vol < 1e-9:
            return None

        return btc_ret / btc_vol

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        if entries_so_far >= 1:
            return None

        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < self.entry_second_min or elapsed > self.entry_second_max:
            return None

        z = self._get_btc_zscore(row)
        if z is None:
            return None

        # Confirmacion de Polymarket
        leader_dur = row.get("leader_duration_ticks", 0)
        if leader_dur < self.min_leader_duration:
            return None

        # Volatilidad del token (quiero estabilidad)
        token_vol = row.get("up_volatility_20", 999)
        if token_vol > self.max_token_vol:
            return None

        btc_ret = row.get("btc_return_from_start", 0)

        if z > self.min_z_score:
            price = row.get("up_price", 0)
            leader = row.get("leader", 0)
            # Poly debe confirmar: UP es lider
            if leader != 1:
                return None
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "UP",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        if z < -self.min_z_score:
            price = row.get("down_price", 0)
            leader = row.get("leader", 0)
            if leader != -1:
                return None
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "DOWN",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)

        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}

        # Z-score reversal
        z = self._get_btc_zscore(row)
        if z is not None:
            if pos.side == "UP" and z < self.reversal_threshold_z:
                return {"price": current, "reason": "BTC_REVERSAL"}
            if pos.side == "DOWN" and z > -self.reversal_threshold_z:
                return {"price": current, "reason": "BTC_REVERSAL"}

        return None


# ============================================================================
# ESTRATEGIA 4: Ultra Cheap Sniper (0.25-0.40 only)
# ============================================================================

class UltraCheapSniperStrategy:
    """
    MÁXIMO riesgo/recompensa: solo compra tokens entre 0.25-0.40.

    Logica:
    - A 0.30, si ganas: +233%. Si pierdes: -100%.
    - Solo necesitas 30% WR para breakeven.
    - Con BTC momentum, WR estimado ~50-55%.
    - Edge masivo.

    HOLD TO RESOLUTION. Sin TP. Sin SL.
    Maxima conviccion, minimo numero de trades.
    """

    name = "Ultra Cheap Sniper (0.25-0.40)"

    def __init__(
        self,
        min_btc_abs_return: float = 0.0020,   # 0.20% - señal mas fuerte
        max_entry_price: float = 0.40,         # solo tokens MUY baratos
        min_entry_price: float = 0.25,
        entry_second_min: float = 180,         # minuto 3
        entry_second_max: float = 360,         # minuto 6 (mas selectivo)
        min_spread: float = 0.10,              # spread minimo (evitar 50/50)
    ):
        self.min_btc_abs_return = min_btc_abs_return
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.entry_second_min = entry_second_min
        self.entry_second_max = entry_second_max
        self.min_spread = min_spread

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        if entries_so_far >= 1:
            return None

        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < self.entry_second_min or elapsed > self.entry_second_max:
            return None

        btc_ret = row.get("btc_return_from_start")
        if btc_ret is None or (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            return None

        # Verificar que hay spread suficiente (no mercado 50/50)
        abs_spread = row.get("abs_spread", 0)
        if abs_spread < self.min_spread:
            return None

        if btc_ret > self.min_btc_abs_return:
            price = row.get("up_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "UP",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        if btc_ret < -self.min_btc_abs_return:
            price = row.get("down_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "DOWN",
                    "price": price,
                    "sl": 0.0,
                    "tp": 99.0,
                    "btc_ret_at_entry": btc_ret,
                }

        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)

        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}

        return None


# ============================================================================
# ESTRATEGIA LEGACY: Binance Momentum (para comparacion)
# ============================================================================

class BinanceMomentumStrategy:
    """
    Estrategia original de referencia (v1).
    """

    name = "Binance Momentum (Legacy v1)"

    def __init__(
        self,
        min_btc_return: float = 0.0008,
        max_entry_price: float = 0.75,
        min_entry_price: float = 0.30,
        stop_loss_margin: float = 0.15,
        take_profit_price: float = 0.92,
        time_stop_second: float = 780,
    ):
        self.min_btc_return = min_btc_return
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.stop_loss_margin = stop_loss_margin
        self.take_profit_price = take_profit_price
        self.time_stop_second = time_stop_second

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < ENTRY_SECOND_MIN or elapsed > ENTRY_SECOND_MAX:
            return None
        if entries_so_far >= MAX_ENTRIES_PER_MARKET:
            return None

        btc_ret = row.get("btc_return_from_start")
        if btc_ret is None or (isinstance(btc_ret, float) and np.isnan(btc_ret)):
            return None

        if btc_ret > self.min_btc_return:
            price = row.get("up_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "UP",
                    "price": price,
                    "sl": max(price - self.stop_loss_margin, 0.01),
                    "tp": self.take_profit_price,
                }

        if btc_ret < -self.min_btc_return:
            price = row.get("down_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {
                    "side": "DOWN",
                    "price": price,
                    "sl": max(price - self.stop_loss_margin, 0.01),
                    "tp": self.take_profit_price,
                }

        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)
        elapsed = row.get("elapsed_seconds", 0)

        if current >= pos.take_profit:
            return {"price": current, "reason": "TAKE_PROFIT"}
        if current <= pos.stop_loss:
            return {"price": current, "reason": "STOP_LOSS"}
        if elapsed >= self.time_stop_second:
            return {"price": current, "reason": "TIME_STOP"}
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}
        return None


# ============================================================================
# ESTRATEGIA LEGACY: Smart Timing (solo Polymarket)
# ============================================================================

class SmartTimingStrategy:
    name = "Smart Timing (Polymarket Only)"

    def __init__(
        self,
        min_leader_duration: int = 30,
        min_abs_spread: float = 0.15,
        max_entry_price: float = 0.78,
        min_entry_price: float = 0.52,
        stop_loss_margin: float = 0.18,
        take_profit_price: float = 0.93,
        time_stop_second: float = 780,
    ):
        self.min_leader_duration = min_leader_duration
        self.min_abs_spread = min_abs_spread
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.stop_loss_margin = stop_loss_margin
        self.take_profit_price = take_profit_price
        self.time_stop_second = time_stop_second

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < ENTRY_SECOND_MIN or elapsed > ENTRY_SECOND_MAX:
            return None
        if entries_so_far >= MAX_ENTRIES_PER_MARKET:
            return None

        leader_dur = row.get("leader_duration_ticks", 0)
        abs_spread = row.get("abs_spread", 0)
        if leader_dur < self.min_leader_duration or abs_spread < self.min_abs_spread:
            return None

        leader = row.get("leader", 0)
        if leader == 1:
            price = row.get("up_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {"side": "UP", "price": price,
                        "sl": max(price - self.stop_loss_margin, 0.01),
                        "tp": self.take_profit_price}
        elif leader == -1:
            price = row.get("down_price", 0)
            if self.min_entry_price <= price <= self.max_entry_price:
                return {"side": "DOWN", "price": price,
                        "sl": max(price - self.stop_loss_margin, 0.01),
                        "tp": self.take_profit_price}
        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)
        elapsed = row.get("elapsed_seconds", 0)
        if current >= pos.take_profit:
            return {"price": current, "reason": "TAKE_PROFIT"}
        if current <= pos.stop_loss:
            return {"price": current, "reason": "STOP_LOSS"}
        if elapsed >= self.time_stop_second:
            return {"price": current, "reason": "TIME_STOP"}
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}
        return None


# ============================================================================
# ESTRATEGIA LEGACY: Calibration Edge
# ============================================================================

class CalibrationEdgeStrategy:
    name = "Calibration Edge"

    def __init__(
        self,
        entry_price_min: float = 0.60,
        entry_price_max: float = 0.78,
        max_volatility: float = 0.04,
        min_elapsed: float = 240,
        max_elapsed: float = 480,
        stop_loss_margin: float = 0.20,
        take_profit_price: float = 0.95,
        time_stop_second: float = 840,
    ):
        self.entry_price_min = entry_price_min
        self.entry_price_max = entry_price_max
        self.max_volatility = max_volatility
        self.min_elapsed = min_elapsed
        self.max_elapsed = max_elapsed
        self.stop_loss_margin = stop_loss_margin
        self.take_profit_price = take_profit_price
        self.time_stop_second = time_stop_second

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < self.min_elapsed or elapsed > self.max_elapsed:
            return None
        if entries_so_far >= MAX_ENTRIES_PER_MARKET:
            return None

        vol = row.get("up_volatility_20", 999)
        if vol > self.max_volatility:
            return None

        up = row.get("up_price", 0)
        down = row.get("down_price", 0)
        if up > down and self.entry_price_min <= up <= self.entry_price_max:
            return {"side": "UP", "price": up,
                    "sl": max(up - self.stop_loss_margin, 0.01),
                    "tp": self.take_profit_price}
        if down > up and self.entry_price_min <= down <= self.entry_price_max:
            return {"side": "DOWN", "price": down,
                    "sl": max(down - self.stop_loss_margin, 0.01),
                    "tp": self.take_profit_price}
        return None

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)
        elapsed = row.get("elapsed_seconds", 0)
        if current >= pos.take_profit:
            return {"price": current, "reason": "TAKE_PROFIT"}
        if current <= pos.stop_loss:
            return {"price": current, "reason": "STOP_LOSS"}
        if elapsed >= self.time_stop_second:
            return {"price": current, "reason": "TIME_STOP"}
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}
        return None


# ============================================================================
# ESTRATEGIA LEGACY: Combined Ensemble
# ============================================================================

class CombinedStrategy:
    name = "Combined Ensemble"

    def __init__(
        self,
        min_btc_return: float = 0.0005,
        min_leader_duration: int = 15,
        min_abs_spread: float = 0.10,
        max_entry_price: float = 0.78,
        min_entry_price: float = 0.35,
        stop_loss_margin: float = 0.16,
        take_profit_price: float = 0.92,
        time_stop_second: float = 780,
    ):
        self.min_btc_return = min_btc_return
        self.min_leader_duration = min_leader_duration
        self.min_abs_spread = min_abs_spread
        self.max_entry_price = max_entry_price
        self.min_entry_price = min_entry_price
        self.stop_loss_margin = stop_loss_margin
        self.take_profit_price = take_profit_price
        self.time_stop_second = time_stop_second

    def should_enter(self, row: dict, entries_so_far: int) -> Optional[Dict]:
        elapsed = row.get("elapsed_seconds", 0)
        if elapsed < ENTRY_SECOND_MIN or elapsed > ENTRY_SECOND_MAX:
            return None
        if entries_so_far >= MAX_ENTRIES_PER_MARKET:
            return None

        btc_ret = row.get("btc_return_from_start")
        leader_dur = row.get("leader_duration_ticks", 0)
        abs_spread = row.get("abs_spread", 0)

        has_binance = btc_ret is not None and not (isinstance(btc_ret, float) and np.isnan(btc_ret))

        if not has_binance:
            if leader_dur < 40 or abs_spread < 0.20:
                return None
        else:
            if abs(btc_ret) < self.min_btc_return:
                return None
            if leader_dur < self.min_leader_duration:
                return None
            if abs_spread < self.min_abs_spread:
                return None

            leader = row.get("leader", 0)
            if btc_ret > 0 and leader != 1:
                return None
            if btc_ret < 0 and leader != -1:
                return None

        leader = row.get("leader", 0)
        if leader == 1:
            price = row.get("up_price", 0)
            side = "UP"
        elif leader == -1:
            price = row.get("down_price", 0)
            side = "DOWN"
        else:
            return None

        if not (self.min_entry_price <= price <= self.max_entry_price):
            return None

        return {
            "side": side, "price": price,
            "sl": max(price - self.stop_loss_margin, 0.01),
            "tp": self.take_profit_price,
        }

    def should_exit(self, row: dict, pos: Position) -> Optional[Dict]:
        current = row.get("up_price" if pos.side == "UP" else "down_price", 0)
        elapsed = row.get("elapsed_seconds", 0)
        if current >= pos.take_profit:
            return {"price": current, "reason": "TAKE_PROFIT"}
        if current <= pos.stop_loss:
            return {"price": current, "reason": "STOP_LOSS"}
        if elapsed >= self.time_stop_second:
            return {"price": current, "reason": "TIME_STOP"}
        if elapsed >= 895:
            return {"price": current, "reason": "RESOLUTION"}
        return None


# ============================================================================
# Registro de todas las estrategias
# ============================================================================

ALL_STRATEGIES = [
    # NUEVAS (cheap token family)
    CheapTokenHoldStrategy(),
    CheapTokenReversalStopStrategy(),
    CheapTokenZScoreStrategy(),
    UltraCheapSniperStrategy(),
    # LEGACY (para comparacion)
    BinanceMomentumStrategy(),
]
