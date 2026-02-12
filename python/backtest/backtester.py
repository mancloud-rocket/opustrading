"""
Motor de backtesting.
Simula estrategias sobre datos historicos tick-by-tick.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from strategy.strategies import Position, TradeResult, compute_pnl
from config import BET_SIZE


@dataclass
class MarketResult:
    market_name: str
    resolution: Optional[str]
    trades: List[TradeResult] = field(default_factory=list)
    num_ticks: int = 0


@dataclass
class BacktestResult:
    strategy_name: str
    market_results: List[MarketResult] = field(default_factory=list)

    @property
    def all_trades(self) -> List[TradeResult]:
        trades = []
        for mr in self.market_results:
            trades.extend(mr.trades)
        return trades

    @property
    def total_trades(self) -> int:
        return len(self.all_trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.all_trades if t.pnl_net > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.all_trades if t.pnl_net <= 0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_net for t in self.all_trades)

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def max_drawdown(self) -> float:
        if not self.all_trades:
            return 0.0
        equity = np.cumsum([t.pnl_net for t in self.all_trades])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    @property
    def sharpe(self) -> float:
        pnls = [t.pnl_net for t in self.all_trades]
        if len(pnls) < 2:
            return 0.0
        mean = np.mean(pnls)
        std = np.std(pnls)
        if std < 1e-9:
            return 0.0
        return float(mean / std)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.pnl_net for t in self.all_trades if t.pnl_net > 0)
        gross_losses = abs(sum(t.pnl_net for t in self.all_trades if t.pnl_net <= 0))
        if gross_losses < 1e-9:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def avg_win(self) -> float:
        wins = [t.pnl_net for t in self.all_trades if t.pnl_net > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_net for t in self.all_trades if t.pnl_net <= 0]
        return float(np.mean(losses)) if losses else 0.0


def run_backtest_on_market(
    strategy,
    market_df: pd.DataFrame,
    market_name: str,
    resolution: Optional[str] = None,
) -> MarketResult:
    """
    Ejecuta una estrategia sobre un mercado individual.
    market_df debe tener features ya computados.
    """
    result = MarketResult(
        market_name=market_name,
        resolution=resolution,
        num_ticks=len(market_df),
    )

    # Pre-convertir a lista de dicts para velocidad
    cols = market_df.columns.tolist()
    rows = market_df.to_numpy()

    position: Optional[Position] = None
    entries = 0

    for idx in range(len(rows)):
        row = {cols[j]: rows[idx, j] for j in range(len(cols))}

        # Si tenemos posicion, verificar salidas
        if position is not None:
            exit_signal = strategy.should_exit(row, position)
            if exit_signal is not None:
                exit_price = exit_signal["price"]
                reason = exit_signal["reason"]

                # Para RESOLUTION, usar el precio real segun resolucion
                if reason == "RESOLUTION" and resolution is not None:
                    if position.side == resolution:
                        exit_price = 0.99  # gano
                    else:
                        exit_price = 0.01  # perdio

                gross, net = compute_pnl(position.entry_price, exit_price)
                trade = TradeResult(
                    side=position.side,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    entry_second=position.entry_second,
                    exit_second=row.get("elapsed_seconds", 0),
                    reason=reason,
                    pnl_gross=gross,
                    pnl_net=net,
                )
                result.trades.append(trade)
                position = None
                continue

        # Si no tenemos posicion, buscar entrada
        if position is None:
            entry_signal = strategy.should_enter(row, entries)
            if entry_signal is not None:
                position = Position(
                    side=entry_signal["side"],
                    entry_price=entry_signal["price"],
                    entry_second=row.get("elapsed_seconds", 0),
                    entry_idx=idx,
                    stop_loss=entry_signal.get("sl", 0.0),
                    take_profit=entry_signal.get("tp", 99.0),
                    btc_ret_at_entry=entry_signal.get("btc_ret_at_entry", 0.0),
                )
                entries += 1

    # Si quedo posicion abierta al final, cerrar a resolucion
    if position is not None:
        last_row = {cols[j]: rows[-1, j] for j in range(len(cols))}
        if resolution is not None:
            if position.side == resolution:
                exit_price = 0.99
            else:
                exit_price = 0.01
        else:
            if position.side == "UP":
                exit_price = last_row.get("up_price", 0.5)
            else:
                exit_price = last_row.get("down_price", 0.5)

        gross, net = compute_pnl(position.entry_price, exit_price)
        trade = TradeResult(
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_second=position.entry_second,
            exit_second=last_row.get("elapsed_seconds", 900),
            reason="END_OF_DATA",
            pnl_gross=gross,
            pnl_net=net,
        )
        result.trades.append(trade)

    return result


def run_full_backtest(
    strategy,
    markets_with_features: Dict[str, pd.DataFrame],
    resolutions: Dict[str, Optional[str]],
) -> BacktestResult:
    """
    Ejecuta backtest completo sobre todos los mercados.
    """
    result = BacktestResult(strategy_name=strategy.name)

    for name, mdf in markets_with_features.items():
        res = resolutions.get(name)
        mr = run_backtest_on_market(strategy, mdf, name, res)
        result.market_results.append(mr)

    return result
