"""
OPUS Trading Bot v3.0 - Backtester Principal

Ejecuta el pipeline completo:
  1. Parsear datos historicos de Polymarket (prices.csv)
  2. Descargar datos historicos de Binance (BTC/USDT)
  3. Computar features
  4. Analisis Lead-Lag (Binance vs Polymarket)
  5. Backtest de multiples estrategias
  6. Reporte de resultados

Uso:
  cd python
  pip install -r requirements.txt
  python run_backtest.py
"""

import sys
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Agregar directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PRICES_CSV, CACHE_DIR, BET_SIZE, FEE_RATE,
    BINANCE_SYMBOL, MARKET_DURATION_SECONDS,
)
from data.parser import (
    load_markets, determine_resolution, parse_market_times,
    build_market_summary,
)
from data.binance_client import (
    download_klines, determine_resolution_binance,
    get_btc_returns_series, get_btc_price_at,
)
from features.engine import (
    compute_polymarket_features, compute_binance_features,
    merge_features, get_available_features,
)
from backtest.backtester import run_full_backtest, BacktestResult
from backtest.lead_lag import run_lead_lag_analysis
from strategy.strategies import (
    ALL_STRATEGIES,
    BinanceMomentumStrategy,
    SmartTimingStrategy,
    CalibrationEdgeStrategy,
    CombinedStrategy,
    CheapTokenHoldStrategy,
    CheapTokenReversalStopStrategy,
    CheapTokenZScoreStrategy,
    UltraCheapSniperStrategy,
)


# ---------------------------------------------------------------------------
# Utilidades de impresion
# ---------------------------------------------------------------------------

SEP = "=" * 72
THIN = "-" * 72


def header(text: str):
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)


def subheader(text: str):
    print(f"\n{THIN}")
    print(f"  {text}")
    print(THIN)


def print_backtest_result(result: BacktestResult, show_trades: int = 10):
    """Imprime resultados de backtest de forma legible."""
    subheader(f"ESTRATEGIA: {result.strategy_name}")

    if result.total_trades == 0:
        print("  Sin trades generados.")
        return

    operated = sum(1 for mr in result.market_results if mr.trades)
    total_mkt = len(result.market_results)

    print(f"  Mercados operados : {operated} / {total_mkt}")
    print(f"  Total trades      : {result.total_trades}")
    print(f"  Wins / Losses     : {result.wins} / {result.losses}")
    print(f"  Win Rate          : {result.win_rate * 100:.1f}%")
    print(f"  PnL Total (net)   : ${result.total_pnl:.2f}")
    print(f"  PnL Promedio      : ${result.avg_pnl:.2f}")
    print(f"  Avg Win           : ${result.avg_win:.2f}")
    print(f"  Avg Loss          : ${result.avg_loss:.2f}")
    print(f"  Profit Factor     : {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio      : {result.sharpe:.3f}")
    print(f"  Max Drawdown      : ${result.max_drawdown:.2f}")

    # Desglose por razon de salida
    reasons = {}
    for t in result.all_trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1
    print(f"  Razones de salida :")
    for reason, count in sorted(reasons.items()):
        print(f"    {reason:20s}: {count}")

    # Distribucion de entry prices
    entries = [t.entry_price for t in result.all_trades]
    if entries:
        print(f"  Entry prices dist : min={min(entries):.2f} "
              f"Q25={np.percentile(entries, 25):.2f} "
              f"median={np.median(entries):.2f} "
              f"Q75={np.percentile(entries, 75):.2f} "
              f"max={max(entries):.2f}")

    # Distribucion de entry times (en segundos)
    entry_secs = [t.entry_second for t in result.all_trades]
    if entry_secs:
        print(f"  Entry time dist   : min={min(entry_secs):.0f}s "
              f"median={np.median(entry_secs):.0f}s "
              f"max={max(entry_secs):.0f}s")

    # Distribucion de duracion
    durations = [t.exit_second - t.entry_second for t in result.all_trades]
    if durations:
        print(f"  Duracion trade    : min={min(durations):.0f}s "
              f"median={np.median(durations):.0f}s "
              f"max={max(durations):.0f}s")

    # Top 5 mejores y peores trades
    trades_sorted = sorted(result.all_trades, key=lambda t: t.pnl_net)
    n_show = min(5, len(trades_sorted))
    if n_show >= 2:
        print(f"  Top {n_show} peores      :")
        for t in trades_sorted[:n_show]:
            dur = t.exit_second - t.entry_second
            print(f"    {t.side} @{t.entry_second:.0f}s "
                  f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                  f"dur={dur:.0f}s -> ${t.pnl_net:.2f} ({t.reason})")
        print(f"  Top {n_show} mejores     :")
        for t in trades_sorted[-n_show:]:
            dur = t.exit_second - t.entry_second
            print(f"    {t.side} @{t.entry_second:.0f}s "
                  f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                  f"dur={dur:.0f}s -> ${t.pnl_net:.2f} ({t.reason})")

    # Primeros N trades cronologicos (para verificar logica)
    if show_trades > 0:
        all_t = result.all_trades[:show_trades]
        print(f"\n  Primeros {len(all_t)} trades (cronologico):")
        for i, t in enumerate(all_t):
            dur = t.exit_second - t.entry_second
            print(f"    #{i+1:3d}: {t.side:4s} entry={t.entry_price:.2f} "
                  f"exit={t.exit_price:.2f} @{t.entry_second:.0f}s "
                  f"dur={dur:.0f}s ${t.pnl_net:+.2f} ({t.reason})")

    # PnL por mercado
    mkt_pnls = []
    for mr in result.market_results:
        if mr.trades:
            mpnl = sum(t.pnl_net for t in mr.trades)
            mkt_pnls.append((mr.market_name, mpnl, len(mr.trades)))

    if mkt_pnls:
        mkt_pnls.sort(key=lambda x: x[1])
        print(f"\n  PnL por mercado (top 5 peor / mejor):")
        for name, pnl, n_trades in mkt_pnls[:5]:
            short = name[:55]
            print(f"    ${pnl:+7.2f} ({n_trades}t) {short}")
        print(f"    ...")
        for name, pnl, n_trades in mkt_pnls[-5:]:
            short = name[:55]
            print(f"    ${pnl:+7.2f} ({n_trades}t) {short}")


def print_lead_lag_results(ll_results: dict):
    """Imprime resultados del analisis lead-lag."""
    subheader("ANALISIS LEAD-LAG: Binance vs Polymarket")

    if ll_results["status"] == "NO_DATA":
        print(f"  {ll_results['message']}")
        print("  (Esto es normal si Binance no estaba accesible)")
        print("  => Se procedera con estrategias basadas solo en Polymarket.")
        return

    print(f"  Mercados analizados       : {ll_results['total_markets_analyzed']}")
    print(f"  Mercados saltados         : {ll_results['skipped_markets']}")
    print(f"  Con lead-lag significativo : {ll_results['significant_markets']}"
          f" ({ll_results['significant_pct']:.1f}%)")
    print(f"  Binance lidera            : {ll_results['btc_leads_count']}")
    print(f"  Polymarket lidera         : {ll_results['poly_leads_count']}")
    print(f"  Lag promedio (minutos)    : {ll_results['avg_optimal_lag']:.2f}")
    print(f"  Lag mediana               : {ll_results['median_optimal_lag']:.1f}")
    print(f"  Correlacion promedio      : {ll_results['avg_max_correlation']:.4f}")
    print(f"  Corr contemporanea prom   : {ll_results['avg_contemporaneous_corr']:.4f}")
    print(f"  Muestras prom/mercado     : {ll_results['avg_samples_per_market']:.1f}")
    print(f"  Minutos alineados prom    : {ll_results['avg_aligned_minutes']:.1f}")
    print(f"\n  CONCLUSION:")
    for line in ll_results["conclusion"].split("\n"):
        print(f"    {line}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    header("OPUS TRADING BOT v3.0 - BACKTESTER")
    print(f"  Fecha:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Bet size:    ${BET_SIZE:.2f}")
    print(f"  Fee rate:    {FEE_RATE * 100:.1f}% por lado")
    print(f"  Archivo:     {PRICES_CSV}")

    # ==================================================================
    # PASO 1: Parsear datos de Polymarket
    # ==================================================================
    header("[1/5] Parseando datos de Polymarket...")

    csv_path = str(PRICES_CSV)
    if not PRICES_CSV.exists():
        print(f"  ERROR: No se encuentra {csv_path}")
        sys.exit(1)

    markets = load_markets(csv_path)
    print(f"  Mercados encontrados: {len(markets)}")

    # Resumen
    summary = build_market_summary(markets)
    resolved = summary[summary["resolution_poly"].notna()]
    print(f"  Con resolucion determinable (Polymarket): {len(resolved)}")

    # Rango de fechas
    all_min = summary["start_utc"].dropna().min()
    all_max = summary["end_utc"].dropna().max()
    if all_min is not None:
        print(f"  Rango: {all_min} -> {all_max}")

    # Distribucion de resoluciones
    if len(resolved) > 0:
        up_count = (resolved["resolution_poly"] == "UP").sum()
        down_count = (resolved["resolution_poly"] == "DOWN").sum()
        print(f"  Resoluciones: UP={up_count} DOWN={down_count} "
              f"({up_count/(up_count+down_count)*100:.0f}% UP)")

    # Estadisticas de ticks por mercado
    tick_counts = summary["num_ticks"]
    print(f"  Ticks/mercado: min={tick_counts.min():.0f} "
          f"median={tick_counts.median():.0f} max={tick_counts.max():.0f}")

    # Muestra de mercados
    print(f"\n  Muestra de 5 mercados:")
    for _, row in summary.head(5).iterrows():
        name = row["market"][:55]
        print(f"    {name}")
        print(f"      ticks={int(row['num_ticks'])} "
              f"range=[{row['min_second']:.0f}s-{row['max_second']:.0f}s] "
              f"res={row['resolution_poly']}")

    # ==================================================================
    # PASO 2: Descargar datos de Binance
    # ==================================================================
    header("[2/5] Descargando datos historicos de Binance (BTC/USDT)...")

    btc_df = pd.DataFrame()
    btc_features_cache = {}
    market_times = {}
    resolutions = {}

    # Parsear tiempos de cada mercado
    parsed_ok = 0
    parsed_fail = 0
    for name in markets:
        times = parse_market_times(name)
        if times:
            market_times[name] = times
            parsed_ok += 1
        else:
            parsed_fail += 1

    print(f"  Tiempos parseados OK: {parsed_ok} / Fallidos: {parsed_fail}")

    if parsed_ok > 0:
        # Muestra de tiempos parseados
        sample_times = list(market_times.items())[:3]
        for name, times in sample_times:
            print(f"    {name[:55]}")
            print(f"      start_utc={times['start_utc']}  end_utc={times['end_utc']}")

    if all_min is not None and all_max is not None:
        try:
            # Descargar klines de 1 minuto para todo el periodo
            dl_start = all_min - timedelta(minutes=30)
            dl_end = all_max + timedelta(minutes=30)

            print(f"\n  Descargando klines 1m: {dl_start} -> {dl_end}")
            btc_df = download_klines(
                start_utc=dl_start,
                end_utc=dl_end,
                symbol=BINANCE_SYMBOL,
                interval="1m",
                cache_dir=CACHE_DIR,
            )
            print(f"  Velas descargadas: {len(btc_df)}")

            if not btc_df.empty:
                print(f"  BTC open_time rango: {btc_df['open_time'].min()} -> "
                      f"{btc_df['open_time'].max()}")
                print(f"  BTC close rango: ${btc_df['close'].min():.2f} -> "
                      f"${btc_df['close'].max():.2f}")
                print(f"  BTC tz info: {btc_df['open_time'].dt.tz}")

                btc_returns = get_btc_returns_series(btc_df)
                print(f"  Retornos calculados: {len(btc_returns)} velas")

                # Determinar resolucion via Binance (ground truth)
                binance_resolved = 0
                binance_errors = 0
                binance_flat = 0

                # Mostrar detalle para los primeros 5
                detail_count = 0

                for name, times in market_times.items():
                    show = detail_count < 5
                    try:
                        start_ts = pd.Timestamp(times["start_utc"])
                        end_ts = pd.Timestamp(times["end_utc"])

                        # Asegurar UTC
                        if start_ts.tzinfo is None:
                            start_ts = start_ts.tz_localize("UTC")
                        else:
                            start_ts = start_ts.tz_convert("UTC")
                        if end_ts.tzinfo is None:
                            end_ts = end_ts.tz_localize("UTC")
                        else:
                            end_ts = end_ts.tz_convert("UTC")

                        res = determine_resolution_binance(
                            btc_df, start_ts, end_ts, verbose=show
                        )
                    except Exception as e:
                        if show:
                            print(f"      [BnRes] EXCEPTION: {e}")
                        res = None
                        binance_errors += 1

                    if res:
                        resolutions[name] = res
                        binance_resolved += 1
                    elif res is None:
                        binance_flat += 1
                        # Fallback a resolucion de Polymarket
                        poly_res = determine_resolution(markets[name])
                        if poly_res:
                            resolutions[name] = poly_res

                    detail_count += 1

                print(f"\n  Resoluciones via Binance: {binance_resolved}")
                print(f"  Binance flat/error: {binance_flat} flat, {binance_errors} errors")
                print(f"  Total resoluciones: {len(resolutions)}")

                # Comparar resoluciones Binance vs Polymarket
                if binance_resolved > 0:
                    agree = 0
                    disagree = 0
                    for name in market_times:
                        poly_res = determine_resolution(markets[name])
                        bn_res = resolutions.get(name)
                        if poly_res and bn_res:
                            if poly_res == bn_res:
                                agree += 1
                            else:
                                disagree += 1
                    print(f"  Binance vs Polymarket: {agree} acuerdo, "
                          f"{disagree} desacuerdo")
            else:
                print("  Sin datos de Binance (DataFrame vacio)")

        except Exception as e:
            print(f"  Error descargando datos de Binance: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuando solo con datos de Polymarket...")

    # Si no hay resoluciones, usar Polymarket
    if not resolutions:
        print("  Usando resoluciones estimadas desde Polymarket...")
        for name, mdf in markets.items():
            res = determine_resolution(mdf)
            if res:
                resolutions[name] = res
        print(f"  Resoluciones desde Polymarket: {len(resolutions)}")

    # Distribucion de resoluciones finales
    up_res = sum(1 for v in resolutions.values() if v == "UP")
    down_res = sum(1 for v in resolutions.values() if v == "DOWN")
    print(f"  Resoluciones finales: UP={up_res} DOWN={down_res}")

    # ==================================================================
    # PASO 3: Computar features
    # ==================================================================
    header("[3/5] Computando features...")

    markets_featured = {}
    has_binance = not btc_df.empty
    btc_merge_success = 0
    btc_merge_fail = 0

    for name, mdf in markets.items():
        # Features de Polymarket
        featured = compute_polymarket_features(mdf)

        # Features de Binance (si disponible)
        if has_binance and name in market_times:
            times = market_times[name]
            start_utc = pd.Timestamp(times["start_utc"])
            end_utc = pd.Timestamp(times["end_utc"])

            if start_utc.tzinfo is None:
                start_utc = start_utc.tz_localize("UTC")
            else:
                start_utc = start_utc.tz_convert("UTC")
            if end_utc.tzinfo is None:
                end_utc = end_utc.tz_localize("UTC")
            else:
                end_utc = end_utc.tz_convert("UTC")

            btc_ret_series = get_btc_returns_series(btc_df)
            btc_feat = compute_binance_features(btc_ret_series, start_utc, end_utc)

            # Merge
            old_cols = set(featured.columns)
            featured = merge_features(featured, btc_feat)
            new_cols = set(featured.columns) - old_cols

            if "btc_return_from_start" in featured.columns:
                valid = featured["btc_return_from_start"].notna().sum()
                if valid > 0:
                    btc_merge_success += 1
                else:
                    btc_merge_fail += 1
            else:
                btc_merge_fail += 1

        markets_featured[name] = featured

    # Contar features disponibles
    sample_df = next(iter(markets_featured.values()))
    avail = get_available_features(sample_df)
    print(f"  Features de Polymarket: {sum(1 for f in avail if 'btc' not in f)}")
    print(f"  Features de Binance:    {sum(1 for f in avail if 'btc' in f)}")
    print(f"  Total features:         {len(avail)}")
    if has_binance:
        print(f"  Merge BTC exitoso: {btc_merge_success} / Fallido: {btc_merge_fail}")

    # Validar: mostrar sample de datos mergeados
    if has_binance:
        print(f"\n  Muestra de datos mergeados (primer mercado con BTC):")
        for name, mdf in list(markets_featured.items())[:1]:
            if "btc_return_from_start" in mdf.columns:
                valid_btc = mdf[mdf["btc_return_from_start"].notna()]
                if len(valid_btc) > 0:
                    # Mostrar primeros 5 ticks con datos BTC
                    sample = valid_btc.head(5)
                    for _, r in sample.iterrows():
                        print(f"    t={r.get('elapsed_seconds', '?'):.0f}s "
                              f"UP={r['up_price']:.2f} DOWN={r['down_price']:.2f} "
                              f"btc_ret={r['btc_return_from_start']:.6f}")
                    # Y ultimos 3
                    print(f"    ...")
                    sample = valid_btc.tail(3)
                    for _, r in sample.iterrows():
                        print(f"    t={r.get('elapsed_seconds', '?'):.0f}s "
                              f"UP={r['up_price']:.2f} DOWN={r['down_price']:.2f} "
                              f"btc_ret={r['btc_return_from_start']:.6f}")

    # ==================================================================
    # PASO 4: Analisis Lead-Lag
    # ==================================================================
    header("[4/5] Analisis Lead-Lag...")

    if has_binance:
        ll_results = run_lead_lag_analysis(
            markets, btc_returns, market_times,
            verbose=True, show_per_market=5,
        )
        print_lead_lag_results(ll_results)
    else:
        print("  Sin datos de Binance. Saltando analisis lead-lag.")
        print("  => Las estrategias se evaluaran solo con datos de Polymarket.")
        ll_results = {"status": "NO_DATA"}

    # ==================================================================
    # PASO 5: Backtest de estrategias
    # ==================================================================
    header("[5/5] Ejecutando backtests...")

    # Filtrar mercados con resolucion conocida y suficientes ticks
    valid_markets = {
        name: mdf
        for name, mdf in markets_featured.items()
        if name in resolutions and len(mdf) >= 100
    }
    print(f"  Mercados validos para backtest: {len(valid_markets)}")

    # Log: cuantos tienen datos de Binance
    with_btc = sum(
        1 for mdf in valid_markets.values()
        if "btc_return_from_start" in mdf.columns
        and mdf["btc_return_from_start"].notna().any()
    )
    print(f"  Con datos de Binance: {with_btc}")

    # Seleccionar estrategias segun datos disponibles
    if has_binance:
        strategies = ALL_STRATEGIES
    else:
        # Sin Binance, excluir estrategia que depende de BTC
        strategies = [
            s for s in ALL_STRATEGIES
            if not isinstance(s, BinanceMomentumStrategy)
        ]
        print("  (Sin Binance: excluyendo estrategia BinanceMomentum)")

    all_results = []
    for strategy in strategies:
        t_strat = time.time()
        result = run_full_backtest(strategy, valid_markets, resolutions)
        dt = time.time() - t_strat
        all_results.append(result)
        print_backtest_result(result, show_trades=10)
        print(f"\n  (Backtest completado en {dt:.1f}s)")

    # ==================================================================
    # RESUMEN COMPARATIVO
    # ==================================================================
    header("RESUMEN COMPARATIVO")

    if all_results:
        rows = []
        for r in all_results:
            rows.append({
                "Estrategia": r.strategy_name,
                "Trades": r.total_trades,
                "WR%": f"{r.win_rate*100:.1f}",
                "PnL $": f"{r.total_pnl:.2f}",
                "Avg PnL $": f"{r.avg_pnl:.2f}",
                "Sharpe": f"{r.sharpe:.3f}",
                "MaxDD $": f"{r.max_drawdown:.2f}",
                "PF": f"{r.profit_factor:.2f}",
            })
        comp_df = pd.DataFrame(rows)
        print(comp_df.to_string(index=False))

        # Mejor estrategia
        best = max(all_results, key=lambda r: r.total_pnl)
        print(f"\n  >>> MEJOR ESTRATEGIA: {best.strategy_name}")
        print(f"      PnL: ${best.total_pnl:.2f} | "
              f"WR: {best.win_rate*100:.1f}% | "
              f"Sharpe: {best.sharpe:.3f}")

    # ==================================================================
    # VALIDACION CRITICA
    # ==================================================================
    header("VALIDACION DE RESULTADOS")

    print("  Checks criticos:")

    # Check 1: resolucion Binance vs Polymarket
    if has_binance:
        bn_up = sum(1 for n, r in resolutions.items()
                     if r == "UP" and n in market_times)
        bn_down = sum(1 for n, r in resolutions.items()
                       if r == "DOWN" and n in market_times)
        print(f"  [1] Resoluciones usadas: UP={bn_up} DOWN={bn_down} "
              f"({bn_up/(bn_up+bn_down)*100:.0f}% UP)" if bn_up + bn_down > 0
              else "  [1] Sin resoluciones")

    # Check 2: verificar que BTC features tienen varianza
    if has_binance:
        btc_rets = []
        for mdf in valid_markets.values():
            if "btc_return_from_start" in mdf.columns:
                vals = mdf["btc_return_from_start"].dropna()
                if len(vals) > 0:
                    btc_rets.extend(vals.tolist())
        if btc_rets:
            arr = np.array(btc_rets)
            print(f"  [2] btc_return_from_start stats: "
                  f"mean={np.mean(arr):.6f} std={np.std(arr):.6f} "
                  f"min={np.min(arr):.6f} max={np.max(arr):.6f} "
                  f"n={len(arr)}")
            # Cuantos superan el threshold de 0.0008
            above = np.sum(np.abs(arr) > 0.0008)
            print(f"      Ticks con |btc_ret| > 0.08%: {above} "
                  f"({above/len(arr)*100:.1f}%)")
        else:
            print(f"  [2] ALERTA: Sin datos de btc_return_from_start!")

    # Check 3: verificar que las resoluciones no tienen look-ahead bias
    # (las resoluciones de Polymarket usan los ultimos ticks)
    print(f"  [3] Resolucion method: {'Binance (ground truth)' if binance_resolved > 0 else 'Polymarket heuristic (posible sesgo)'}"
          if has_binance else "  [3] Resolucion method: Polymarket heuristic")

    # Check 4: cuantos trades salen por RESOLUTION vs otros
    for r in all_results:
        reasons = {}
        for t in r.all_trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        res_exits = reasons.get("RESOLUTION", 0) + reasons.get("END_OF_DATA", 0)
        total = r.total_trades
        if total > 0:
            print(f"  [4] {r.strategy_name}: {res_exits}/{total} "
                  f"({res_exits/total*100:.0f}%) exits por RESOLUTION/END_OF_DATA")

    # ==================================================================
    # RESPUESTA A: Que pasa si el lead-lag no existe?
    # ==================================================================
    header("ANALISIS: Y SI EL LEAD-LAG NO EXISTE?")

    # Separar estrategias que usan BTC vs las que no
    BINANCE_STRATEGY_TYPES = (
        BinanceMomentumStrategy,
        CheapTokenHoldStrategy,
        CheapTokenReversalStopStrategy,
        CheapTokenZScoreStrategy,
        UltraCheapSniperStrategy,
    )

    btc_results = []
    poly_only_results = []
    for r in all_results:
        matched = [s for s in strategies if s.name == r.strategy_name]
        if matched and isinstance(matched[0], BINANCE_STRATEGY_TYPES):
            btc_results.append(r)
        else:
            poly_only_results.append(r)

    if btc_results:
        best_btc = max(btc_results, key=lambda r: r.total_pnl)
        print(f"  Mejor estrategia BTC: {best_btc.strategy_name}")
        print(f"    PnL: ${best_btc.total_pnl:.2f} | "
              f"WR: {best_btc.win_rate*100:.1f}% | "
              f"Trades: {best_btc.total_trades}")

    if poly_only_results:
        best_poly = max(poly_only_results, key=lambda r: r.total_pnl)
        print(f"  Mejor estrategia Poly-only: {best_poly.strategy_name}")
        print(f"    PnL: ${best_poly.total_pnl:.2f} | "
              f"WR: {best_poly.win_rate*100:.1f}%")

    if btc_results and poly_only_results:
        delta = best_btc.total_pnl - best_poly.total_pnl
        print(f"  Delta (BTC vs Poly-only): ${delta:.2f}")
        if delta > 0:
            print("  => Datos de Binance agregan valor.")
        else:
            print("  => Datos de Binance NO agregan valor.")

    if not btc_results and not poly_only_results:
        print("  Sin resultados para comparar.")

    # ==================================================================
    # THRESHOLD SWEEP: sensibilidad al BTC return minimo
    # ==================================================================
    header("THRESHOLD SWEEP: btc_return minimo")

    if has_binance:
        thresholds = [0.0005, 0.0008, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]
        price_ranges = [
            ("0.25-0.40", 0.25, 0.40),
            ("0.25-0.50", 0.25, 0.50),
            ("0.30-0.50", 0.30, 0.50),
            ("0.30-0.60", 0.30, 0.60),
        ]

        print(f"  {'Threshold':>10s} {'PriceRange':>12s} {'Trades':>7s} "
              f"{'WR%':>6s} {'PnL':>10s} {'AvgPnL':>8s} {'PF':>6s} {'MaxDD':>8s}")
        print(f"  {'-'*10} {'-'*12} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")

        best_sweep_pnl = -float("inf")
        best_sweep_params = ""

        for pr_label, pr_min, pr_max in price_ranges:
            for thr in thresholds:
                strat = CheapTokenHoldStrategy(
                    min_btc_abs_return=thr,
                    min_entry_price=pr_min,
                    max_entry_price=pr_max,
                )
                r = run_full_backtest(strat, valid_markets, resolutions)
                if r.total_trades > 0:
                    wr = r.win_rate * 100
                    pf = r.profit_factor
                    print(f"  {thr:>10.4f} {pr_label:>12s} {r.total_trades:>7d} "
                          f"{wr:>5.1f}% ${r.total_pnl:>9.2f} "
                          f"${r.avg_pnl:>7.2f} {pf:>5.2f} ${r.max_drawdown:>7.2f}")
                    if r.total_pnl > best_sweep_pnl:
                        best_sweep_pnl = r.total_pnl
                        best_sweep_params = f"thr={thr}, price={pr_label}"

        if best_sweep_params:
            print(f"\n  Mejor combinacion: {best_sweep_params} -> ${best_sweep_pnl:.2f}")
            print(f"  (Cuidado: esto es in-sample. Walk-forward abajo valida)")

    # ==================================================================
    # VALIDACION: Walk-Forward Split
    # ==================================================================
    header("WALK-FORWARD VALIDATION")

    # Dividir mercados por fecha: train = Feb 3-5, test = Feb 6-7
    train_markets = {}
    test_markets = {}
    train_res = {}
    test_res = {}

    for name, mdf in valid_markets.items():
        times = market_times.get(name)
        if times is None:
            continue
        start = pd.Timestamp(times["start_utc"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        # Feb 6 00:00 UTC = cutoff
        cutoff = pd.Timestamp("2026-02-06", tz="UTC")
        if start < cutoff:
            train_markets[name] = mdf
            if name in resolutions:
                train_res[name] = resolutions[name]
        else:
            test_markets[name] = mdf
            if name in resolutions:
                test_res[name] = resolutions[name]

    print(f"  Train (Feb 3-5): {len(train_markets)} mercados")
    print(f"  Test  (Feb 6-7): {len(test_markets)} mercados")

    if train_markets and test_markets:
        for strategy in strategies:
            train_result = run_full_backtest(strategy, train_markets, train_res)
            test_result = run_full_backtest(strategy, test_markets, test_res)

            t_trades = train_result.total_trades
            t_wr = train_result.win_rate * 100 if t_trades > 0 else 0
            t_pnl = train_result.total_pnl
            s_trades = test_result.total_trades
            s_wr = test_result.win_rate * 100 if s_trades > 0 else 0
            s_pnl = test_result.total_pnl

            status = "OK" if (t_pnl > 0 and s_pnl > 0) else "OVERFIT?" if (t_pnl > 0 and s_pnl <= 0) else "BAD"
            print(f"\n  {strategy.name}:")
            print(f"    TRAIN: {t_trades} trades | WR {t_wr:.1f}% | PnL ${t_pnl:.2f}")
            print(f"    TEST:  {s_trades} trades | WR {s_wr:.1f}% | PnL ${s_pnl:.2f}")
            print(f"    => {status}")

    # ==================================================================
    # VALIDACION: Permutation Test (señal BTC es real?)
    # ==================================================================
    header("PERMUTATION TEST (BTC señal real o ruido?)")

    # Solo para la mejor estrategia cheap token
    cheap_strategies = [s for s in strategies
                        if isinstance(s, (CheapTokenHoldStrategy,
                                          CheapTokenReversalStopStrategy,
                                          UltraCheapSniperStrategy))]

    if cheap_strategies and has_binance:
        best_cheap = None
        best_cheap_pnl = -float("inf")
        for s in cheap_strategies:
            r = [x for x in all_results if x.strategy_name == s.name]
            if r and r[0].total_pnl > best_cheap_pnl:
                best_cheap = s
                best_cheap_pnl = r[0].total_pnl

        if best_cheap is not None:
            print(f"  Testeando: {best_cheap.name} (PnL real: ${best_cheap_pnl:.2f})")
            print(f"  Permutando btc_return_from_start 200 veces...")

            rng = np.random.RandomState(42)
            perm_pnls = []

            for perm_i in range(200):
                # Crear copia con BTC returns permutados
                perm_markets = {}
                for name, mdf in valid_markets.items():
                    perm_df = mdf.copy()
                    if "btc_return_from_start" in perm_df.columns:
                        vals = perm_df["btc_return_from_start"].values.copy()
                        rng.shuffle(vals)
                        perm_df["btc_return_from_start"] = vals
                    perm_markets[name] = perm_df

                perm_result = run_full_backtest(best_cheap, perm_markets, resolutions)
                perm_pnls.append(perm_result.total_pnl)

            perm_arr = np.array(perm_pnls)
            p_value = np.mean(perm_arr >= best_cheap_pnl)
            print(f"\n  Resultados permutacion (n=200):")
            print(f"    PnL real:         ${best_cheap_pnl:.2f}")
            print(f"    PnL perm mean:    ${np.mean(perm_arr):.2f}")
            print(f"    PnL perm median:  ${np.median(perm_arr):.2f}")
            print(f"    PnL perm std:     ${np.std(perm_arr):.2f}")
            print(f"    PnL perm min/max: ${np.min(perm_arr):.2f} / ${np.max(perm_arr):.2f}")
            print(f"    p-value:          {p_value:.4f}")
            if p_value < 0.05:
                print(f"    => SEÑAL SIGNIFICATIVA (p < 0.05). "
                      f"El BTC momentum NO es ruido.")
            else:
                print(f"    => SEÑAL NO SIGNIFICATIVA. "
                      f"El PnL podria ser azar.")
    else:
        print("  Sin estrategias cheap token o sin datos Binance.")

    # Tiempo total
    elapsed = time.time() - t_start
    print(f"\n  Tiempo total: {elapsed:.1f} segundos")

    header("FIN DEL BACKTEST")


if __name__ == "__main__":
    main()
