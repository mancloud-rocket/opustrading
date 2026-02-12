"""
Health monitor: verifica que todos los componentes estan funcionando.
Auto-recovery para reconexiones y errores transitorios.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ComponentHealth:
    """Estado de salud de un componente."""
    name: str
    last_success: float = 0.0
    last_error: float = 0.0
    consecutive_errors: int = 0
    total_errors: int = 0
    total_success: int = 0
    last_error_msg: str = ""

    def record_success(self):
        self.last_success = time.time()
        self.consecutive_errors = 0
        self.total_success += 1

    def record_error(self, msg: str = ""):
        self.last_error = time.time()
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error_msg = msg

    @property
    def is_healthy(self) -> bool:
        if self.total_success == 0:
            return False
        if self.consecutive_errors >= 5:
            return False
        if self.last_success == 0:
            return False
        age = time.time() - self.last_success
        return age < 30.0

    @property
    def status_str(self) -> str:
        if self.is_healthy:
            return "OK"
        if self.consecutive_errors >= 5:
            return f"ERROR (x{self.consecutive_errors})"
        if self.total_success == 0:
            return "WAITING"
        return "DEGRADED"


class HealthMonitor:
    """
    Monitor de salud del sistema.
    Rastrea componentes y decide si hay que intervenir.
    """

    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {
            "binance": ComponentHealth(name="Binance Feed"),
            "polymarket_discovery": ComponentHealth(name="Polymarket Discovery"),
            "polymarket_prices": ComponentHealth(name="Polymarket Prices"),
            "trader": ComponentHealth(name="Trading Engine"),
        }
        self._start_time = time.time()

    def record(self, component: str, success: bool, error_msg: str = ""):
        """Registra resultado de una operacion."""
        if component not in self.components:
            self.components[component] = ComponentHealth(name=component)

        if success:
            self.components[component].record_success()
        else:
            self.components[component].record_error(error_msg)

    def is_system_healthy(self) -> bool:
        """True si todos los componentes criticos estan ok."""
        critical = ["binance", "polymarket_prices"]
        return all(
            self.components[c].is_healthy
            for c in critical
            if c in self.components
        )

    def should_restart_market_scan(self) -> bool:
        """True si debemos reintentar descubrimiento de mercado."""
        disc = self.components.get("polymarket_discovery")
        if disc is None:
            return True
        if not disc.is_healthy and disc.consecutive_errors >= 3:
            return True
        return False

    def get_status_summary(self) -> str:
        """Resumen de estado de todos los componentes."""
        lines = []
        for key, comp in self.components.items():
            age = ""
            if comp.last_success > 0:
                age = f" (last ok: {time.time() - comp.last_success:.0f}s ago)"
            lines.append(
                f"  {comp.name:25s} [{comp.status_str:10s}] "
                f"ok={comp.total_success} err={comp.total_errors}{age}"
            )

        uptime = time.time() - self._start_time
        lines.insert(0, f"  System uptime: {uptime/60:.1f} min")
        return "\n".join(lines)

    def get_wait_recommendation(self) -> float:
        """
        Recomienda cuanto esperar basado en errores acumulados.
        Backoff exponencial suave.
        """
        max_consec = max(
            (c.consecutive_errors for c in self.components.values()),
            default=0,
        )
        if max_consec == 0:
            return 1.0  # Normal
        if max_consec < 3:
            return 2.0
        if max_consec < 5:
            return 5.0
        if max_consec < 10:
            return 10.0
        return 30.0  # Problema serio, esperar mas

