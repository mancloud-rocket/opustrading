"""
Motor de trading: ML predictor + paper trading + risk management.

Estrategia v3.2 - ML Predictive (GradientBoosting):
  - Modelo GBM con ponderacion automatica de features
  - Solo features BTC (sin leakage de Polymarket)
  - Prediccion CONTINUA: se ejecuta en cada tick/iteracion
  - Entry: P > 0.65 con BTC threshold 0.08% como hard filter
  - Exit: Take profit ($0.97), Stop Loss (50%), o resolucion
  - Fallback a threshold si modelo no esta disponible
"""

import time
import csv
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

from live.polymarket_client import MarketInfo, PriceSnapshot
from live.signal_processor import SignalProcessor, SignalState
from live.settings import get_strategy_config

# Cargar configuracion desde settings.py
STRATEGY_CONFIG = get_strategy_config()

# Intentar cargar predictor ML
try:
    from ml.predictor import MarketPredictor
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Position:
    """Posicion abierta."""
    side: str               # 'UP' o 'DOWN'
    entry_price: float
    entry_second: float
    entry_time: datetime
    btc_at_entry: float     # BTC price al momento de entrar
    btc_ret_at_entry: float  # BTC return from start al entrar
    market_slug: str
    market_title: str

    @property
    def entry_minute(self) -> str:
        m = int(self.entry_second // 60)
        s = int(self.entry_second % 60)
        return f"{m}:{s:02d}"


@dataclass
class TradeRecord:
    """Registro de un trade completado."""
    timestamp: str
    market: str
    slug: str
    side: str
    entry_price: float
    exit_price: float
    entry_second: float
    exit_second: float
    reason: str              # RESOLUTION, BTC_REVERSAL, MARKET_CLOSE
    btc_at_entry: float
    btc_at_exit: float
    btc_ret_at_entry: float
    btc_ret_at_exit: float
    pnl_gross: float
    pnl_net: float
    fees: float


@dataclass
class DailyStats:
    """Estadisticas del dia."""
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    markets_seen: int = 0
    markets_traded: int = 0

    @property
    def win_rate(self) -> float:
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

    @property
    def is_loss_limit_hit(self) -> bool:
        return self.pnl_net <= -STRATEGY_CONFIG["max_daily_loss"]


# ============================================================================
# Trader principal
# ============================================================================

class Trader:
    """
    Motor de paper trading.
    Gestiona señales, posiciones, logs y riesgo.
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "live_trades.csv")
        self.log_file = os.path.join(data_dir, "live_log.csv")

        # Estado
        self.position: Optional[Position] = None
        self.btc_at_market_start: Optional[float] = None
        self.market_slug: Optional[str] = None
        self.entries_this_market: int = 0
        self.trade_history: List[TradeRecord] = []

        # Tick history for ML features (reset per market)
        self._btc_ticks: List[Tuple[float, float]] = []   # (elapsed_s, btc_price)
        self._poly_ticks: List[Tuple[float, float, float]] = []  # (elapsed_s, up, down)

        # Signal processor avanzado
        self.signals = SignalProcessor(window_size=30)
        self.last_signal_state: Optional[SignalState] = None

        # ML Predictor (prediccion continua cada tick)
        self.predictor: Optional[MarketPredictor] = None
        self._load_predictor()

        # Prediccion actual (actualizada cada tick)
        self.last_prediction = None  # PredictionResult o None
        self._prediction_history: List = []  # historial de predicciones del mercado

        # Stats
        self.daily_stats = DailyStats(
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        self.session_balance = 0.0

        # Init files
        self._init_files()

    def _load_predictor(self):
        """Intenta cargar el modelo ML entrenado."""
        if not _ML_AVAILABLE:
            return

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "ml", "model.pkl"
        )
        self.predictor = MarketPredictor()
        if self.predictor.load(model_path):
            stats = self.predictor.training_stats
            auc = stats.get("cv_auc", 0)
            n = stats.get("n_markets", 0)
            print(f"  ML Model cargado: AUC={auc:.3f}, "
                  f"entrenado con {n} mercados")
        else:
            print("  ML Model no encontrado. Usando fallback (thresholds).")
            print("  Para entrenar: python train_model.py")
            self.predictor = None

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _init_files(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if not os.path.exists(self.trades_file):
            with open(self.trades_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "market", "slug", "side",
                    "entry_price", "exit_price",
                    "entry_second", "exit_second", "reason",
                    "btc_at_entry", "btc_at_exit",
                    "btc_ret_at_entry", "btc_ret_at_exit",
                    "pnl_gross", "pnl_net", "fees", "balance",
                ])

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "market", "elapsed_s",
                    "up_price", "down_price", "btc_price",
                    "btc_return", "signal", "position",
                    "event", "ml_p_up", "ml_confidence",
                ])

    def _write_trade(self, trade: TradeRecord):
        """Escribe un trade completado al CSV."""
        with open(self.trades_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.timestamp, trade.market, trade.slug, trade.side,
                f"{trade.entry_price:.4f}", f"{trade.exit_price:.4f}",
                f"{trade.entry_second:.0f}", f"{trade.exit_second:.0f}",
                trade.reason,
                f"{trade.btc_at_entry:.2f}", f"{trade.btc_at_exit:.2f}",
                f"{trade.btc_ret_at_entry:.6f}", f"{trade.btc_ret_at_exit:.6f}",
                f"{trade.pnl_gross:.2f}", f"{trade.pnl_net:.2f}",
                f"{trade.fees:.2f}", f"{self.session_balance:.2f}",
            ])

    def _write_log(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
        btc_return: Optional[float],
        signal: str,
        event: str,
    ):
        """Escribe una linea de log (incluye prediccion ML)."""
        now = datetime.now(timezone.utc).isoformat()
        elapsed = market.elapsed_seconds if market else 0

        # ML prediction info (si disponible)
        ml_p_up = ""
        ml_conf = ""
        if self.last_prediction is not None:
            ml_p_up = f"{self.last_prediction.p_up:.4f}"
            ml_conf = f"{self.last_prediction.confidence:.4f}"

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now,
                market.title if market else "",
                f"{elapsed:.0f}",
                f"{prices.up:.4f}" if prices.up else "",
                f"{prices.down:.4f}" if prices.down else "",
                f"{btc_price:.2f}" if btc_price else "",
                f"{btc_return:.6f}" if btc_return is not None else "",
                signal,
                self.position.side if self.position else "NONE",
                event,
                ml_p_up,
                ml_conf,
            ])

    # ------------------------------------------------------------------
    # Market lifecycle
    # ------------------------------------------------------------------

    def on_new_market(self, market: MarketInfo, btc_price: Optional[float]):
        """Llamado cuando se descubre un nuevo mercado."""
        self.market_slug = market.slug
        self.btc_at_market_start = btc_price
        self.entries_this_market = 0

        # Reset tick history para ML features
        self._btc_ticks = []
        self._poly_ticks = []

        # Reset signal processor y predicciones para nuevo mercado
        self.signals.reset(btc_price)
        self.last_signal_state = None
        self.last_prediction = None
        self._prediction_history = []

        # Check daily stats reset
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.daily_stats.date:
            self._print_daily_summary()
            self.daily_stats = DailyStats(date=today)

        self.daily_stats.markets_seen += 1

    def on_market_close(
        self,
        market: MarketInfo,
        last_prices: PriceSnapshot,
        btc_price: Optional[float],
    ):
        """Llamado cuando un mercado termina (resolucion)."""
        if self.position is not None:
            # Cerrar posicion por resolucion
            exit_price = 0.0
            if last_prices.is_valid:
                if self.position.side == "UP":
                    exit_price = last_prices.up
                else:
                    exit_price = last_prices.down

            self._close_position(
                exit_price=exit_price,
                exit_second=market.elapsed_seconds,
                reason="RESOLUTION",
                btc_at_exit=btc_price or 0.0,
                btc_ret_at_exit=self._calc_btc_return(btc_price),
                market=market,
            )

    # ------------------------------------------------------------------
    # Signal engine
    # ------------------------------------------------------------------

    def process_tick(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
    ) -> str:
        """
        Procesa un tick. Retorna el evento generado:
        'ENTRY_UP', 'ENTRY_DOWN', 'EXIT_REVERSAL', 'HOLD', 'WAIT', 'SKIP'
        """
        if not prices.is_valid:
            return "INVALID_PRICES"

        # Update signal processor
        if btc_price is not None:
            self.last_signal_state = self.signals.update(btc_price)

        btc_return = self._calc_btc_return(btc_price)
        elapsed = market.elapsed_seconds
        event = "WAIT"
        signal = "NONE"

        # Collect tick data for ML features
        if btc_price is not None:
            self._btc_ticks.append((elapsed, btc_price))
        if prices.up is not None and prices.down is not None:
            self._poly_ticks.append((elapsed, prices.up, prices.down))

        # === PREDICCION CONTINUA (cada tick) ===
        if self.predictor and self.predictor.is_trained:
            features = self.predictor.compute_live_features(
                btc_ticks=self._btc_ticks,
                poly_ticks=self._poly_ticks,
                btc_start=self.btc_at_market_start or 0,
                elapsed=elapsed,
            )
            if features is not None:
                self.last_prediction = self.predictor.predict(features)
                if self.last_prediction is not None:
                    self._prediction_history.append({
                        "elapsed": elapsed,
                        "p_up": self.last_prediction.p_up,
                        "confidence": self.last_prediction.confidence,
                        "side": self.last_prediction.predicted_side,
                    })

        # === CHECK EXITS FIRST ===
        if self.position is not None:
            exit_event = self._check_exit(
                market, prices, btc_price, btc_return
            )
            if exit_event:
                event = exit_event
                signal = exit_event
            else:
                event = "HOLD"
                signal = "HOLD"

        # === CHECK ENTRY ===
        elif self.position is None:
            entry_event = self._check_entry(
                market, prices, btc_price, btc_return, elapsed
            )
            if entry_event:
                event = entry_event
                signal = entry_event
            else:
                # Describe signal state
                if self.last_signal_state:
                    signal = f"SIG:{self.last_signal_state.signal_quality}"
                elif btc_return is not None:
                    if abs(btc_return) > STRATEGY_CONFIG["btc_threshold"]:
                        signal = "BTC_SIGNAL"
                    else:
                        signal = "BTC_WEAK"
                else:
                    signal = "NO_BTC"

        # Log
        self._write_log(market, prices, btc_price, btc_return, signal, event)

        return event

    def _calc_btc_return(self, btc_price: Optional[float]) -> Optional[float]:
        """Calcula BTC return from market start."""
        if btc_price is None or self.btc_at_market_start is None:
            return None
        if self.btc_at_market_start == 0:
            return None
        return (btc_price - self.btc_at_market_start) / self.btc_at_market_start

    def _check_entry(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
        btc_return: Optional[float],
        elapsed: float,
    ) -> Optional[str]:
        """
        Verifica si debemos entrar.

        Usa ML predictor si esta disponible, fallback a thresholds.
        Retorna 'ENTRY_UP', 'ENTRY_DOWN' o None.
        """
        cfg = STRATEGY_CONFIG

        # Risk check
        if self.daily_stats.is_loss_limit_hit:
            return None

        # Max entries per market
        if self.entries_this_market >= cfg["max_trades_per_market"]:
            return None

        # Entry window
        if elapsed < cfg["entry_second_min"] or elapsed > cfg["entry_second_max"]:
            return None

        # BTC signal required (minimo basico)
        if btc_return is None:
            return None

        # ====== ML PREDICTION ======
        if self.predictor is not None and self.predictor.is_trained:
            return self._check_entry_ml(
                market, prices, btc_price, btc_return, elapsed
            )

        # ====== FALLBACK: Threshold-based ======
        return self._check_entry_threshold(
            market, prices, btc_price, btc_return, elapsed
        )

    def _check_entry_ml(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
        btc_return: Optional[float],
        elapsed: float,
    ) -> Optional[str]:
        """
        Entry decision basada en ML predictor.
        Usa self.last_prediction (actualizado cada tick) en vez de recomputar.
        """
        cfg = STRATEGY_CONFIG

        # HARD FILTER: BTC threshold minimo (independiente del ML).
        # Sin esto, el ML entra en senales de 0.05% que luego reversan.
        if btc_return is None or abs(btc_return) < cfg["btc_threshold"]:
            return None

        # Usar prediccion continua (ya calculada en process_tick)
        result = self.last_prediction
        if result is None:
            return None

        # Minimum confidence for entry
        min_confidence = cfg.get("ml_min_confidence", 0.58)
        if result.confidence < min_confidence:
            return None

        # Estabilidad: verificar que la prediccion es consistente
        # (al menos 3 ticks consecutivos prediciendo el mismo lado)
        if len(self._prediction_history) >= 3:
            recent = self._prediction_history[-3:]
            sides = [p["side"] for p in recent]
            if len(set(sides)) > 1:
                return None  # Prediccion inestable, no entrar

        # Max entry price desde settings
        is_strong = abs(btc_return) >= cfg.get("strong_signal_threshold", 0.0015)
        max_price = cfg.get("entry_price_max_strong", 0.65) if is_strong else cfg["entry_price_max"]
        min_price = cfg["entry_price_min"]

        # Select side and check price
        if result.predicted_side == "UP":
            price = prices.up
            if price is not None and min_price <= price <= max_price:
                self._open_position(
                    side="UP",
                    entry_price=price,
                    entry_second=elapsed,
                    btc_price=btc_price or 0.0,
                    btc_return=btc_return,
                    market=market,
                    ml_confidence=result.confidence,
                    ml_p_up=result.p_up,
                )
                return "ENTRY_UP"

        elif result.predicted_side == "DOWN":
            price = prices.down
            if price is not None and min_price <= price <= max_price:
                self._open_position(
                    side="DOWN",
                    entry_price=price,
                    entry_second=elapsed,
                    btc_price=btc_price or 0.0,
                    btc_return=btc_return,
                    market=market,
                    ml_confidence=result.confidence,
                    ml_p_up=result.p_up,
                )
                return "ENTRY_DOWN"

        return None

    def _check_entry_threshold(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
        btc_return: Optional[float],
        elapsed: float,
    ) -> Optional[str]:
        """Fallback: entry basada en thresholds fijos (sin ML)."""
        cfg = STRATEGY_CONFIG
        abs_ret = abs(btc_return)

        if abs_ret < cfg["btc_threshold"]:
            return None

        # Tiered entry: strong signal allows higher entry prices
        is_strong = abs_ret >= cfg.get("strong_signal_threshold", 0.001)
        max_price = cfg.get("entry_price_max_strong", 0.65) if is_strong else cfg["entry_price_max"]
        min_price = cfg["entry_price_min"]

        # BTC subiendo -> comprar UP
        if btc_return > 0:
            price = prices.up
            if price is not None and min_price <= price <= max_price:
                self._open_position(
                    side="UP",
                    entry_price=price,
                    entry_second=elapsed,
                    btc_price=btc_price or 0.0,
                    btc_return=btc_return,
                    market=market,
                )
                return "ENTRY_UP"

        # BTC bajando -> comprar DOWN
        if btc_return < 0:
            price = prices.down
            if price is not None and min_price <= price <= max_price:
                self._open_position(
                    side="DOWN",
                    entry_price=price,
                    entry_second=elapsed,
                    btc_price=btc_price or 0.0,
                    btc_return=btc_return,
                    market=market,
                )
                return "ENTRY_DOWN"

        return None

    def _check_exit(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
        btc_return: Optional[float],
    ) -> Optional[str]:
        """Verifica si debemos salir. Retorna razon o None."""
        if self.position is None:
            return None

        cfg = STRATEGY_CONFIG
        elapsed = market.elapsed_seconds

        # Resolucion (minuto 14:55+)
        if elapsed >= 895:
            our = prices.up if self.position.side == "UP" else prices.down
            opp = prices.down if self.position.side == "UP" else prices.up

            # Si hay resolucion limpia (1.00 / 0.00), usar ese precio
            if our is not None and (our >= 0.99 or our <= 0.01):
                exit_price = our
            elif our is not None and opp is not None:
                # Sin resolucion limpia: el token mas alto gana
                if our > opp:
                    # Ganamos - usar nuestro precio real
                    exit_price = our
                else:
                    # Perdimos - usar nuestro precio real
                    exit_price = our
            else:
                exit_price = our or 0.0

            self._close_position(
                exit_price=exit_price,
                exit_second=elapsed,
                reason="RESOLUTION",
                btc_at_exit=btc_price or 0.0,
                btc_ret_at_exit=btc_return,
                market=market,
            )
            return "EXIT_RESOLUTION"

        # Precios actuales del token propio y del contrario
        current_token = (
            prices.up if self.position.side == "UP" else prices.down
        )
        opposite_token = (
            prices.down if self.position.side == "UP" else prices.up
        )

        # Take Profit (token >= 0.97)
        tp_price = cfg.get("take_profit_price", 0.97)
        if current_token is not None and current_token >= tp_price:
            self._close_position(
                exit_price=current_token,
                exit_second=elapsed,
                reason="TAKE_PROFIT",
                btc_at_exit=btc_price or 0.0,
                btc_ret_at_exit=btc_return,
                market=market,
            )
            return "EXIT_TAKE_PROFIT"

        # Token Stop Loss: si el token cae X% desde entry, cerrar.
        # VALIDACION (85 trades reales):
        #   Sin SL: 19 trades (22%) fueron a $0.00 = -$394 total
        #   SL 50%: esos trades pierden ~$10.80 cada uno = +$190 de mejora
        sl_pct = cfg.get("token_stop_loss_pct")
        if sl_pct and current_token is not None and self.position.entry_price > 0:
            sl_price = self.position.entry_price * (1 - sl_pct)
            if current_token <= sl_price:
                self._close_position(
                    exit_price=current_token,
                    exit_second=elapsed,
                    reason="TOKEN_STOP_LOSS",
                    btc_at_exit=btc_price or 0.0,
                    btc_ret_at_exit=btc_return,
                    market=market,
                )
                return "EXIT_TOKEN_STOP_LOSS"

        return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _open_position(
        self,
        side: str,
        entry_price: float,
        entry_second: float,
        btc_price: float,
        btc_return: float,
        market: MarketInfo,
        ml_confidence: Optional[float] = None,
        ml_p_up: Optional[float] = None,
    ):
        """Abre una posicion (paper)."""
        self.position = Position(
            side=side,
            entry_price=entry_price,
            entry_second=entry_second,
            entry_time=datetime.now(timezone.utc),
            btc_at_entry=btc_price,
            btc_ret_at_entry=btc_return,
            market_slug=market.slug,
            market_title=market.title,
        )
        self.entries_this_market += 1
        self.daily_stats.markets_traded += 1

        # Log ML info
        if ml_confidence is not None:
            conf_str = f"{ml_confidence*100:.1f}%"
            pup_str = f"{ml_p_up:.3f}" if ml_p_up is not None else "N/A"
            print(f"  [ML] P(UP)={pup_str} Confidence={conf_str} -> {side}")

    def _close_position(
        self,
        exit_price: float,
        exit_second: float,
        reason: str,
        btc_at_exit: float,
        btc_ret_at_exit: Optional[float],
        market: MarketInfo,
    ):
        """Cierra posicion y registra trade."""
        if self.position is None:
            return

        pos = self.position
        cfg = STRATEGY_CONFIG

        # PnL
        entry = pos.entry_price
        if entry > 0:
            gross = cfg["bet_size"] * (exit_price - entry) / entry
        else:
            gross = 0.0
        fees = cfg["bet_size"] * cfg["fee_rate"] * 2
        net = gross - fees

        # Stats
        self.session_balance += net
        self.daily_stats.trades += 1
        self.daily_stats.pnl_gross += gross
        self.daily_stats.pnl_net += net
        if net > 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

        # Record
        trade = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market=pos.market_title,
            slug=pos.market_slug,
            side=pos.side,
            entry_price=entry,
            exit_price=exit_price,
            entry_second=pos.entry_second,
            exit_second=exit_second,
            reason=reason,
            btc_at_entry=pos.btc_at_entry,
            btc_at_exit=btc_at_exit,
            btc_ret_at_entry=pos.btc_ret_at_entry,
            btc_ret_at_exit=btc_ret_at_exit if btc_ret_at_exit is not None else 0.0,
            pnl_gross=gross,
            pnl_net=net,
            fees=fees,
        )
        self.trade_history.append(trade)
        self._write_trade(trade)

        # Clear
        self.position = None

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def get_status_line(
        self,
        market: MarketInfo,
        prices: PriceSnapshot,
        btc_price: Optional[float],
    ) -> str:
        """Genera linea de status para consola."""
        elapsed = market.elapsed_seconds
        minute = int(elapsed // 60)
        second = int(elapsed % 60)
        btc_return = self._calc_btc_return(btc_price)

        parts = [
            f"[{minute}:{second:02d}]",
            f"UP:{prices.up:.2f}" if prices.up else "UP:--",
            f"DN:{prices.down:.2f}" if prices.down else "DN:--",
        ]

        if btc_price:
            parts.append(f"BTC:${btc_price:,.0f}")

        if btc_return is not None:
            sign = "+" if btc_return >= 0 else ""
            parts.append(f"ret:{sign}{btc_return*100:.3f}%")

        # ML prediction continua (actualizada cada tick)
        if self.last_prediction is not None and not self.position:
            result = self.last_prediction
            parts.append(
                f"ML:{result.predicted_side}"
                f" P(UP):{result.p_up:.2f}"
                f"[{result.confidence:.0%}]"
            )

        # Z-score y calidad de señal (fallback/complemento)
        if self.last_signal_state:
            ss = self.last_signal_state
            if ss.btc_zscore is not None:
                parts.append(f"z:{ss.btc_zscore:+.2f}")

        if self.position:
            current = prices.up if self.position.side == "UP" else prices.down
            if current and self.position.entry_price > 0:
                pnl_pct = (current - self.position.entry_price) / self.position.entry_price * 100
                parts.append(
                    f"| POS:{self.position.side}@{self.position.entry_price:.2f}"
                    f" now={current:.2f} PnL:{pnl_pct:+.1f}%"
                )
        else:
            cfg = STRATEGY_CONFIG
            in_w = cfg["entry_second_min"] <= elapsed <= cfg["entry_second_max"]
            window = "IN_WINDOW" if in_w else "WAITING"
            parts.append(f"| {window}")

        return " ".join(parts)

    def _print_daily_summary(self):
        """Imprime resumen diario."""
        s = self.daily_stats
        wr = f"{s.win_rate * 100:.1f}%" if s.trades > 0 else "N/A"
        print(f"\n{'='*60}")
        print(f"  RESUMEN DIARIO: {s.date}")
        print(f"  Mercados vistos:  {s.markets_seen}")
        print(f"  Mercados operados: {s.markets_traded}")
        print(f"  Trades:           {s.trades} ({s.wins}W / {s.losses}L)")
        print(f"  Win Rate:         {wr}")
        print(f"  PnL (net):        ${s.pnl_net:.2f}")
        print(f"  Balance sesion:   ${self.session_balance:.2f}")
        print(f"{'='*60}\n")

    def print_session_summary(self):
        """Imprime resumen de toda la sesion."""
        self._print_daily_summary()
        if self.trade_history:
            print(f"  Trades totales esta sesion: {len(self.trade_history)}")
            total = sum(t.pnl_net for t in self.trade_history)
            print(f"  PnL total sesion: ${total:.2f}")

