"""
Dashboard de sesion para consola.
Lee datos de live_trades.csv y live_log.csv para estadisticas en tiempo real.
"""

import os
import time
import csv
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from live.polymarket_client import MarketInfo, PriceSnapshot


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TradeRow:
    """Un trade leido desde CSV."""
    timestamp: str
    market: str
    slug: str
    side: str
    entry_price: float
    exit_price: float
    entry_second: float
    exit_second: float
    reason: str
    btc_at_entry: float
    btc_at_exit: float
    btc_ret_at_entry: float
    btc_ret_at_exit: float
    pnl_gross: float
    pnl_net: float
    fees: float
    balance: float


@dataclass
class StatsBlock:
    """Bloque de estadisticas calculadas."""
    label: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_net: float = 0.0
    pnl_gross: float = 0.0
    fees: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    current_streak: str = ""
    win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades * 100 if self.trades > 0 else 0.0


# ============================================================================
# Dashboard
# ============================================================================

class Dashboard:
    """Dashboard de estadisticas para consola con datos de CSV."""

    def __init__(self, trader, data_dir: str = "./data"):
        self.trader = trader
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, "live_trades.csv")
        self.log_file = os.path.join(data_dir, "live_log.csv")
        self._start_time = time.time()
        self._last_full_render = 0.0
        self._cached_trades: List[TradeRow] = []
        self._cache_time = 0.0
        self._cache_ttl = 10.0  # refresh every 10s

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_trades(self) -> List[TradeRow]:
        """Lee live_trades.csv con cache."""
        now = time.time()
        if now - self._cache_time < self._cache_ttl and self._cached_trades:
            return self._cached_trades

        trades = []
        if not os.path.exists(self.trades_file):
            return trades

        try:
            with open(self.trades_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        trades.append(TradeRow(
                            timestamp=row["timestamp"],
                            market=row["market"],
                            slug=row.get("slug", ""),
                            side=row["side"],
                            entry_price=float(row["entry_price"]),
                            exit_price=float(row["exit_price"]),
                            entry_second=float(row["entry_second"]),
                            exit_second=float(row["exit_second"]),
                            reason=row["reason"],
                            btc_at_entry=float(row.get("btc_at_entry", 0)),
                            btc_at_exit=float(row.get("btc_at_exit", 0)),
                            btc_ret_at_entry=float(row.get("btc_ret_at_entry", 0)),
                            btc_ret_at_exit=float(row.get("btc_ret_at_exit", 0)),
                            pnl_gross=float(row["pnl_gross"]),
                            pnl_net=float(row["pnl_net"]),
                            fees=float(row["fees"]),
                            balance=float(row["balance"]),
                        ))
                    except (ValueError, KeyError):
                        continue
        except Exception:
            return self._cached_trades  # return old cache on error

        self._cached_trades = trades
        self._cache_time = now
        return trades

    def _detect_sessions(self, trades: List[TradeRow]) -> List[List[TradeRow]]:
        """
        Detecta sesiones separadas. Una sesion nueva empieza cuando:
        - El balance se resetea (balance ≈ pnl_net del trade)
        - Hay un gap de mas de 2 horas entre trades
        """
        if not trades:
            return []

        sessions = []
        current_session = [trades[0]]

        for i in range(1, len(trades)):
            t = trades[i]
            prev = trades[i - 1]

            # Check balance reset
            is_reset = abs(t.balance - t.pnl_net) < 0.02

            # Check time gap
            try:
                t_time = datetime.fromisoformat(t.timestamp)
                p_time = datetime.fromisoformat(prev.timestamp)
                gap = (t_time - p_time).total_seconds()
                is_gap = gap > 7200  # 2 hours
            except Exception:
                is_gap = False

            if is_reset or is_gap:
                sessions.append(current_session)
                current_session = [t]
            else:
                current_session.append(t)

        sessions.append(current_session)
        return sessions

    def _calc_stats(self, trades: List[TradeRow], label: str) -> StatsBlock:
        """Calcula estadisticas de una lista de trades."""
        stats = StatsBlock(label=label)
        if not trades:
            return stats

        stats.trades = len(trades)
        balance = 0.0
        peak = 0.0
        max_dd = 0.0
        streak_char = ""
        streak_count = 0

        for t in trades:
            if t.pnl_net > 0:
                stats.wins += 1
                stats.win_pnls.append(t.pnl_net)
                new_char = "W"
            else:
                stats.losses += 1
                stats.loss_pnls.append(t.pnl_net)
                new_char = "L"

            stats.pnl_net += t.pnl_net
            stats.pnl_gross += t.pnl_gross
            stats.fees += t.fees
            balance += t.pnl_net

            if balance > peak:
                peak = balance
            dd = peak - balance
            if dd > max_dd:
                max_dd = dd

            # Streak
            if new_char == streak_char:
                streak_count += 1
            else:
                streak_char = new_char
                streak_count = 1

        stats.max_drawdown = max_dd
        stats.current_streak = f"{streak_count}{streak_char}" if streak_char else ""

        if stats.win_pnls:
            stats.avg_win = np.mean(stats.win_pnls)
            stats.best_trade = max(stats.win_pnls)
        if stats.loss_pnls:
            stats.avg_loss = np.mean(stats.loss_pnls)
            stats.worst_trade = min(stats.loss_pnls)

        total_wins = sum(stats.win_pnls) if stats.win_pnls else 0
        total_losses = abs(sum(stats.loss_pnls)) if stats.loss_pnls else 0
        stats.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return stats

    def _get_hourly_breakdown(self, trades: List[TradeRow]) -> Dict[int, StatsBlock]:
        """Agrupa trades por hora ET."""
        hourly = {}
        for t in trades:
            try:
                ts = datetime.fromisoformat(t.timestamp)
                # Convert UTC to ET (UTC-5)
                et_hour = (ts.hour - 5) % 24
                if et_hour not in hourly:
                    hourly[et_hour] = []
                hourly[et_hour].append(t)
            except Exception:
                continue

        result = {}
        for hour, hour_trades in sorted(hourly.items()):
            result[hour] = self._calc_stats(hour_trades, f"{hour}:00")
        return result

    def _get_reason_breakdown(self, trades: List[TradeRow]) -> Dict[str, StatsBlock]:
        """Agrupa trades por razon de salida."""
        by_reason = {}
        for t in trades:
            if t.reason not in by_reason:
                by_reason[t.reason] = []
            by_reason[t.reason].append(t)

        result = {}
        for reason, reason_trades in sorted(by_reason.items()):
            result[reason] = self._calc_stats(reason_trades, reason)
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_header(
        self,
        market: Optional[MarketInfo],
        btc_price: Optional[float],
    ) -> str:
        """Render header compacto para cada mercado nuevo."""
        lines = []
        lines.append("")
        lines.append("=" * 70)

        if market:
            elapsed = market.elapsed_seconds
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            remaining = max(0, 900 - elapsed)
            rm = int(remaining // 60)
            rs = int(remaining % 60)

            lines.append(f"  Mercado: {market.title}")
            lines.append(f"  Slug:    {market.slug}")
            lines.append(
                f"  Tiempo:  {m}:{s:02d} elapsed | {rm}:{rs:02d} remaining"
            )

            btc_start = self.trader.btc_at_market_start
            if btc_start and btc_price:
                ret = (btc_price - btc_start) / btc_start * 100
                lines.append(
                    f"  BTC:     ${btc_start:,.2f} -> ${btc_price:,.2f} "
                    f"({ret:+.3f}%)"
                )
        else:
            lines.append("  Esperando mercado...")

        # Quick session stats from in-memory
        s = self.trader.daily_stats
        wr = f"{s.win_rate*100:.1f}%" if s.trades > 0 else "N/A"
        lines.append(f"  Session: {s.trades}T ({s.wins}W/{s.losses}L) "
                      f"WR:{wr} PnL:${s.pnl_net:+.2f}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def render_trade_alert(self, trade, is_entry: bool) -> str:
        """Render alerta de trade."""
        lines = []

        if is_entry:
            lines.append("")
            lines.append(f"  >>> ENTRY {trade.side} @ {trade.entry_price:.2f}")
            lines.append(f"      BTC ret: {trade.btc_ret_at_entry*100:+.4f}%")
            lines.append(f"      Market sec: {trade.entry_second:.0f}")
            lines.append("")
        else:
            pnl_symbol = "+" if trade.pnl_net > 0 else ""
            result = "WIN" if trade.pnl_net > 0 else "LOSS"
            lines.append("")
            lines.append(
                f"  <<< EXIT {trade.side} | {trade.reason} | {result}"
            )
            lines.append(
                f"      Entry: {trade.entry_price:.2f} -> "
                f"Exit: {trade.exit_price:.2f}"
            )
            lines.append(f"      PnL: {pnl_symbol}${trade.pnl_net:.2f}")
            lines.append(
                f"      Balance: ${self.trader.session_balance:+.2f}"
            )
            lines.append("")

        return "\n".join(lines)

    def render_full_stats(self) -> str:
        """
        Render dashboard completo con datos de CSV.
        Se llama cada FULL_STATS_INTERVAL segundos.
        """
        all_trades = self._load_trades()
        sessions = self._detect_sessions(all_trades)

        lines = []
        lines.append("")
        lines.append("\033[96m" + "+" + "=" * 68 + "+")
        lines.append("|" + "  OPUS TRADING DASHBOARD".center(68) + "|")
        lines.append("|" + f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}".center(68) + "|")
        lines.append("+" + "=" * 68 + "+" + "\033[0m")

        # --- Current session stats (from CSV) ---
        if sessions:
            current = sessions[-1]
            cs = self._calc_stats(current, "Sesion actual")
            lines.append("")
            lines.append(f"  \033[1mSESION ACTUAL\033[0m ({len(current)} trades)")
            lines.append(f"  {'─' * 60}")

            wr_color = "\033[92m" if cs.win_rate >= 65 else (
                "\033[93m" if cs.win_rate >= 50 else "\033[91m")
            pnl_color = "\033[92m" if cs.pnl_net >= 0 else "\033[91m"

            lines.append(
                f"  Trades: {cs.trades} "
                f"({cs.wins}W / {cs.losses}L)  |  "
                f"WR: {wr_color}{cs.win_rate:.1f}%\033[0m  |  "
                f"PnL: {pnl_color}${cs.pnl_net:+.2f}\033[0m"
            )
            lines.append(
                f"  Avg Win: \033[92m${cs.avg_win:+.2f}\033[0m  |  "
                f"Avg Loss: \033[91m${cs.avg_loss:+.2f}\033[0m  |  "
                f"PF: {cs.profit_factor:.2f}"
            )
            lines.append(
                f"  Max DD: \033[91m${cs.max_drawdown:.2f}\033[0m  |  "
                f"Best: \033[92m${cs.best_trade:+.2f}\033[0m  |  "
                f"Worst: \033[91m${cs.worst_trade:+.2f}\033[0m  |  "
                f"Streak: {cs.current_streak}"
            )
            lines.append(
                f"  Fees pagadas: ${cs.fees:.2f}  |  "
                f"PnL bruto: ${cs.pnl_gross:+.2f}"
            )

        # --- All-time stats ---
        if len(all_trades) > 0:
            alltime = self._calc_stats(all_trades, "All-time")
            lines.append("")
            lines.append(f"  \033[1mALL-TIME\033[0m ({alltime.trades} trades, "
                          f"{len(sessions)} sesiones)")
            lines.append(f"  {'─' * 60}")

            pnl_color = "\033[92m" if alltime.pnl_net >= 0 else "\033[91m"
            wr_color = "\033[92m" if alltime.win_rate >= 65 else (
                "\033[93m" if alltime.win_rate >= 50 else "\033[91m")
            lines.append(
                f"  PnL total: {pnl_color}${alltime.pnl_net:+.2f}\033[0m  |  "
                f"WR: {wr_color}{alltime.win_rate:.1f}%\033[0m  |  "
                f"PF: {alltime.profit_factor:.2f}  |  "
                f"MaxDD: ${alltime.max_drawdown:.2f}"
            )

        # --- Exit reason breakdown ---
        if sessions:
            current = sessions[-1]
            reasons = self._get_reason_breakdown(current)
            if reasons:
                lines.append("")
                lines.append(f"  \033[1mPOR TIPO DE SALIDA\033[0m")
                lines.append(f"  {'─' * 60}")
                lines.append(
                    f"  {'Razon':<18} {'Trades':>6} {'WR':>7} "
                    f"{'PnL':>10} {'AvgPnL':>9}"
                )
                for reason, rs in reasons.items():
                    wr_color = "\033[92m" if rs.win_rate >= 65 else (
                        "\033[93m" if rs.win_rate >= 50 else "\033[91m")
                    pnl_color = "\033[92m" if rs.pnl_net >= 0 else "\033[91m"
                    avg_pnl = rs.pnl_net / rs.trades if rs.trades > 0 else 0
                    lines.append(
                        f"  {reason:<18} {rs.trades:>6} "
                        f"{wr_color}{rs.win_rate:>6.1f}%\033[0m "
                        f"{pnl_color}${rs.pnl_net:>+9.2f}\033[0m "
                        f"${avg_pnl:>+8.2f}"
                    )

        # --- Hourly performance ---
        if sessions:
            current = sessions[-1]
            hourly = self._get_hourly_breakdown(current)
            if hourly:
                lines.append("")
                lines.append(f"  \033[1mPOR HORA (ET)\033[0m")
                lines.append(f"  {'─' * 60}")
                lines.append(
                    f"  {'Hora':>6} {'Trades':>6} {'W':>3} {'L':>3} "
                    f"{'WR':>7} {'PnL':>10} {'Barra'}"
                )
                for hour, hs in hourly.items():
                    # AM/PM format
                    if hour == 0:
                        h_str = "12AM"
                    elif hour < 12:
                        h_str = f"{hour}AM"
                    elif hour == 12:
                        h_str = "12PM"
                    else:
                        h_str = f"{hour-12}PM"

                    pnl_color = "\033[92m" if hs.pnl_net >= 0 else "\033[91m"
                    bar_len = min(20, int(abs(hs.pnl_net) / 3))
                    bar = ("█" * bar_len) if hs.pnl_net >= 0 else ("░" * bar_len)
                    bar_color = "\033[92m" if hs.pnl_net >= 0 else "\033[91m"

                    lines.append(
                        f"  {h_str:>6} {hs.trades:>6} {hs.wins:>3} {hs.losses:>3} "
                        f"{hs.win_rate:>6.1f}% "
                        f"{pnl_color}${hs.pnl_net:>+9.2f}\033[0m "
                        f"{bar_color}{bar}\033[0m"
                    )

        # --- Last 10 trades ---
        if sessions and sessions[-1]:
            current = sessions[-1]
            recent = current[-10:]
            lines.append("")
            lines.append(f"  \033[1mULTIMOS {len(recent)} TRADES\033[0m")
            lines.append(f"  {'─' * 60}")
            lines.append(
                f"  {'#':>3} {'Market':>18} {'Side':>4} {'Entry':>6} "
                f"{'Exit':>6} {'Reason':>14} {'PnL':>9} {'Bal':>9}"
            )

            running_bal = 0.0
            # Calculate running balance up to the point before recent trades
            offset = len(current) - len(recent)
            if offset > 0:
                running_bal = sum(t.pnl_net for t in current[:offset])

            for i, t in enumerate(recent):
                running_bal += t.pnl_net
                # Extract short market name (e.g., "8:00AM-8:15AM")
                market_short = ""
                if "," in t.market:
                    market_short = t.market.split(",")[1].strip()
                    market_short = market_short.replace(" ET", "")
                else:
                    market_short = t.market[-20:]

                pnl_color = "\033[92m" if t.pnl_net > 0 else "\033[91m"
                result = "W" if t.pnl_net > 0 else "L"
                btc_pct = abs(t.btc_ret_at_entry) * 100

                lines.append(
                    f"  {offset + i + 1:>3} {market_short:>18} {t.side:>4} "
                    f"{t.entry_price:>6.2f} {t.exit_price:>6.2f} "
                    f"{t.reason:>14} "
                    f"{pnl_color}${t.pnl_net:>+8.2f}\033[0m "
                    f"${running_bal:>+8.2f} {result}"
                )

        # --- BTC Threshold analysis (from current session) ---
        if sessions and len(sessions[-1]) >= 5:
            current = sessions[-1]
            lines.append("")
            lines.append(f"  \033[1mANALISIS POR BTC THRESHOLD\033[0m")
            lines.append(f"  {'─' * 60}")
            lines.append(
                f"  {'Threshold':>10} {'Trades':>6} {'WR':>7} "
                f"{'PnL':>10} {'AvgPnL':>9}"
            )
            for thr_pct in [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080]:
                thr = thr_pct / 100.0
                filtered = [
                    t for t in current
                    if abs(t.btc_ret_at_entry) >= thr
                ]
                if not filtered:
                    continue
                fs = self._calc_stats(filtered, f"{thr_pct}%")
                pnl_color = "\033[92m" if fs.pnl_net >= 0 else "\033[91m"
                wr_color = "\033[92m" if fs.win_rate >= 65 else "\033[91m"
                avg_pnl = fs.pnl_net / fs.trades if fs.trades > 0 else 0
                marker = " <-- actual" if abs(thr_pct - 0.065) < 0.001 else ""
                lines.append(
                    f"  {thr_pct:>9.3f}% {fs.trades:>6} "
                    f"{wr_color}{fs.win_rate:>6.1f}%\033[0m "
                    f"{pnl_color}${fs.pnl_net:>+9.2f}\033[0m "
                    f"${avg_pnl:>+8.2f}{marker}"
                )

        # --- System info ---
        uptime = time.time() - self._start_time
        uptime_min = int(uptime // 60)
        uptime_hr = uptime_min // 60
        uptime_m = uptime_min % 60
        lines.append("")
        lines.append(f"  \033[90mUptime: {uptime_hr}h {uptime_m}m | "
                      f"Trades file: {self.trades_file} | "
                      f"Cache: {len(self._cached_trades)} trades\033[0m")
        lines.append("\033[96m" + "+" + "=" * 68 + "+" + "\033[0m")
        lines.append("")

        return "\n".join(lines)

    def should_show_full_stats(self, interval: float = 300.0) -> bool:
        """True si debemos mostrar stats completas (cada N seg)."""
        now = time.time()
        if now - self._last_full_render >= interval:
            self._last_full_render = now
            return True
        return False
