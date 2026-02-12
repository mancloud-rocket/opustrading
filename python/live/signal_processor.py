"""
Signal Processor avanzado.
Calcula features en real-time a partir de BTC price ticks.

Features:
  - btc_return_from_start: (btc_now - btc_start) / btc_start
  - btc_rolling_vol: volatilidad rolling de los ultimos N ticks
  - btc_zscore: btc_return / rolling_vol (señal normalizada por vol)
  - btc_momentum_5s: return en los ultimos 5 segundos
  - trend_strength: cuantos ticks consecutivos en la misma direccion
"""

import time
from collections import deque
from typing import Optional
from dataclasses import dataclass


@dataclass
class SignalState:
    """Estado actual de las señales."""
    btc_return: Optional[float] = None
    btc_zscore: Optional[float] = None
    btc_rolling_vol: Optional[float] = None
    btc_momentum_5s: Optional[float] = None
    trend_strength: int = 0
    trend_direction: str = "NONE"  # UP, DOWN, NONE
    signal_quality: str = "NONE"   # STRONG, MEDIUM, WEAK, NONE

    def should_enter(
        self,
        side: str,
        threshold: float = 0.0003,
        zscore_min: float = 1.0,
    ) -> bool:
        """Decide si la señal justifica entrada."""
        if self.btc_return is None:
            return False

        if side == "UP":
            basic = self.btc_return > threshold
        else:
            basic = self.btc_return < -threshold

        # Si tenemos z-score, usarlo como filtro adicional
        if self.btc_zscore is not None and abs(self.btc_zscore) < zscore_min:
            # Z-score bajo = señal podria ser ruido
            # Pero si el return es muy fuerte, entrar de todos modos
            if abs(self.btc_return) < threshold * 2:
                return False

        return basic


class SignalProcessor:
    """
    Procesa BTC price ticks y calcula señales en real-time.
    """

    def __init__(self, window_size: int = 30):
        self._btc_start: Optional[float] = None
        self._btc_ticks: deque = deque(maxlen=window_size * 2)
        self._returns: deque = deque(maxlen=window_size)
        self._last_price: Optional[float] = None
        self._consecutive_up = 0
        self._consecutive_down = 0
        self._window_size = window_size

    def reset(self, btc_start_price: Optional[float] = None):
        """Reset para nuevo mercado."""
        self._btc_start = btc_start_price
        self._btc_ticks.clear()
        self._returns.clear()
        self._last_price = None
        self._consecutive_up = 0
        self._consecutive_down = 0

        if btc_start_price is not None:
            self._btc_ticks.append((time.time(), btc_start_price))
            self._last_price = btc_start_price

    def update(self, btc_price: float) -> SignalState:
        """
        Procesa un nuevo tick de BTC y retorna el estado de señales actualizado.
        """
        now = time.time()
        state = SignalState()

        if btc_price <= 0:
            return state

        # Guardar tick
        self._btc_ticks.append((now, btc_price))

        # Return desde inicio del mercado
        if self._btc_start is not None and self._btc_start > 0:
            state.btc_return = (btc_price - self._btc_start) / self._btc_start

        # Return tick-to-tick
        if self._last_price is not None and self._last_price > 0:
            tick_return = (btc_price - self._last_price) / self._last_price
            self._returns.append(tick_return)

            # Trend tracking
            if btc_price > self._last_price:
                self._consecutive_up += 1
                self._consecutive_down = 0
            elif btc_price < self._last_price:
                self._consecutive_down += 1
                self._consecutive_up = 0

        self._last_price = btc_price

        # Rolling volatility
        if len(self._returns) >= 5:
            import numpy as np
            returns_arr = list(self._returns)
            vol = float(np.std(returns_arr))
            state.btc_rolling_vol = vol

            # Z-score: return normalizado por volatilidad
            if state.btc_return is not None and vol > 1e-10:
                state.btc_zscore = state.btc_return / vol

        # Momentum 5s: return en ultimos ~5 ticks
        if len(self._btc_ticks) >= 5:
            old_price = self._btc_ticks[-5][1]
            if old_price > 0:
                state.btc_momentum_5s = (btc_price - old_price) / old_price

        # Trend strength
        state.trend_strength = max(self._consecutive_up, self._consecutive_down)
        if self._consecutive_up > self._consecutive_down:
            state.trend_direction = "UP"
        elif self._consecutive_down > self._consecutive_up:
            state.trend_direction = "DOWN"
        else:
            state.trend_direction = "NONE"

        # Signal quality assessment
        state.signal_quality = self._assess_quality(state)

        return state

    def _assess_quality(self, state: SignalState) -> str:
        """
        Evalua la calidad de la señal actual.
        Umbrales ajustados para threshold de entrada de 0.065%.
        """
        if state.btc_return is None:
            return "NONE"

        ret = abs(state.btc_return)
        zscore = abs(state.btc_zscore) if state.btc_zscore is not None else 0

        # STRONG: >0.10% + z-score alto + trend consistente
        if ret > 0.0010 and zscore > 2.0 and state.trend_strength >= 3:
            return "STRONG"

        # MEDIUM: >= 0.065% (at threshold) + algo de confirmacion
        if ret > 0.00065 and (zscore > 1.0 or state.trend_strength >= 2):
            return "MEDIUM"

        # WEAK: por debajo del threshold pero no ruido total
        if ret > 0.0004:
            return "WEAK"

        return "NONE"

    @property
    def btc_start_price(self) -> Optional[float]:
        return self._btc_start

    @btc_start_price.setter
    def btc_start_price(self, value: float):
        self._btc_start = value

