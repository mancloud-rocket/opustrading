"""
Analisis de Lead-Lag entre Binance y Polymarket.

Pregunta clave: Los movimientos de BTC en Binance predicen
los cambios de precios UP/DOWN en Polymarket con retraso?

Si la respuesta es si: tenemos alpha estructural.
Si la respuesta es no: necesitamos estrategias alternativas.

NOTA: Para que el analisis sea valido, ambas series deben
estar en la misma frecuencia temporal.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple


def compute_cross_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 10,
) -> Dict[int, float]:
    """
    Calcula la correlacion cruzada normalizada.

    lag > 0: series_a en t predice series_b en t+lag (A lidera)
    lag < 0: series_b en t predice series_a en t+|lag| (B lidera)
    """
    n = len(series_a)
    if n < 10:
        return {}

    a_std = np.std(series_a)
    b_std = np.std(series_b)
    if a_std < 1e-12 or b_std < 1e-12:
        return {}

    a_norm = (series_a - np.mean(series_a)) / a_std
    b_norm = (series_b - np.mean(series_b)) / b_std

    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_slice = a_norm[:n - lag] if lag > 0 else a_norm
            b_slice = b_norm[lag:] if lag > 0 else b_norm
        else:
            a_slice = a_norm[-lag:]
            b_slice = b_norm[:n + lag]

        k = len(a_slice)
        if k < 5:
            continue

        corr = np.sum(a_slice * b_slice) / k
        if not np.isnan(corr):
            correlations[lag] = float(corr)

    return correlations


def analyze_lead_lag_per_market(
    poly_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    market_start_utc: pd.Timestamp,
    market_end_utc: pd.Timestamp,
    market_name: str = "",
    verbose: bool = False,
) -> Optional[Dict]:
    """
    Analiza lead-lag para un mercado individual.

    CLAVE: Ambas series se resamplean a la MISMA frecuencia (1 minuto)
    para que la comparacion sea valida.
    """
    # Filtrar BTC al rango del mercado (con 5min de buffer)
    buf = pd.Timedelta(minutes=5)
    mask = (btc_df["open_time"] >= (market_start_utc - buf)) & (
        btc_df["open_time"] <= (market_end_utc + buf)
    )
    btc_window = btc_df[mask].copy()

    if verbose:
        print(f"    [LL] {market_name[:60]}")
        print(f"         Rango BTC: {btc_window['open_time'].min()} -> {btc_window['open_time'].max()}"
              if len(btc_window) > 0 else "         Sin datos BTC en rango")
        print(f"         Velas BTC en rango: {len(btc_window)}")
        print(f"         Ticks Poly: {len(poly_df)}")

    if len(btc_window) < 5:
        if verbose:
            print(f"         SKIP: < 5 velas BTC")
        return None

    if len(poly_df) < 50:
        if verbose:
            print(f"         SKIP: < 50 ticks Polymarket")
        return None

    # Resamplear Polymarket a 1 MINUTO (para igualar frecuencia de BTC)
    poly = poly_df.copy()
    poly = poly.set_index("timestamp")
    poly_1m = poly[["up_price"]].resample("1min").last().dropna()

    if verbose:
        print(f"         Poly resampleado 1min: {len(poly_1m)} puntos")

    if len(poly_1m) < 5:
        if verbose:
            print(f"         SKIP: < 5 puntos Poly 1min")
        return None

    # Alinear por timestamp: merge_asof BTC -> Poly
    btc_for_merge = btc_window[["open_time", "btc_close"]].copy()
    btc_for_merge = btc_for_merge.sort_values("open_time")
    btc_for_merge = btc_for_merge.set_index("open_time")

    # Resamplear BTC a 1 minuto (ya lo esta, pero asegurar)
    btc_1m = btc_for_merge.resample("1min").last().dropna()

    if verbose:
        print(f"         BTC resampleado 1min: {len(btc_1m)} puntos")

    # Alinear ambos por indice temporal comun
    common_idx = poly_1m.index.intersection(btc_1m.index)
    if len(common_idx) < 5:
        # Si no hay interseccion exacta, usar merge_asof
        poly_reset = poly_1m.reset_index()
        poly_reset.columns = ["ts", "up_price"]
        btc_reset = btc_1m.reset_index()
        btc_reset.columns = ["ts", "btc_close"]

        merged = pd.merge_asof(
            poly_reset.sort_values("ts"),
            btc_reset.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=90),
        )
        merged = merged.dropna()

        if verbose:
            print(f"         Merge asof: {len(merged)} puntos alineados")

        if len(merged) < 5:
            if verbose:
                print(f"         SKIP: < 5 puntos alineados")
            return None

        poly_prices = merged["up_price"].values
        btc_prices = merged["btc_close"].values
    else:
        if verbose:
            print(f"         Interseccion exacta: {len(common_idx)} puntos")
        poly_prices = poly_1m.loc[common_idx, "up_price"].values
        btc_prices = btc_1m.loc[common_idx, "btc_close"].values

    n = len(poly_prices)

    # Calcular cambios/retornos
    poly_changes = np.diff(poly_prices)
    btc_returns = np.diff(btc_prices) / btc_prices[:-1]

    if verbose:
        print(f"         Series para cross-corr: n={len(poly_changes)}")
        if len(poly_changes) > 0:
            print(f"         Poly changes: mean={np.mean(poly_changes):.6f} "
                  f"std={np.std(poly_changes):.6f}")
            print(f"         BTC returns:  mean={np.mean(btc_returns):.6f} "
                  f"std={np.std(btc_returns):.6f}")

    if len(poly_changes) < 5:
        if verbose:
            print(f"         SKIP: < 5 cambios")
        return None

    # Cross-correlation
    max_lag = min(5, len(poly_changes) // 3)
    corrs = compute_cross_correlation(btc_returns, poly_changes, max_lag=max_lag)

    if not corrs:
        if verbose:
            print(f"         SKIP: sin correlaciones validas")
        return None

    # Lag optimo
    optimal_lag = max(corrs, key=lambda k: abs(corrs[k]))
    max_corr = corrs[optimal_lag]

    # Correlacion contemporanea (lag=0) como referencia
    contemp_corr = corrs.get(0, 0.0)

    # Test de significancia
    if optimal_lag >= 0:
        a = btc_returns[:len(btc_returns) - optimal_lag] if optimal_lag > 0 else btc_returns
        b = poly_changes[optimal_lag:] if optimal_lag > 0 else poly_changes
    else:
        a = btc_returns[-optimal_lag:]
        b = poly_changes[:len(poly_changes) + optimal_lag]

    if len(a) < 3:
        return None

    try:
        r, p_value = stats.pearsonr(a, b)
    except Exception:
        return None

    if verbose:
        print(f"         Lag optimo: {optimal_lag} | corr={max_corr:.4f} | "
              f"p={p_value:.4f} | sig={'SI' if p_value < 0.05 else 'NO'}")
        print(f"         Corr contemporanea (lag=0): {contemp_corr:.4f}")
        # Mostrar todas las correlaciones
        lags_str = " ".join(f"L{k}={v:.3f}" for k, v in sorted(corrs.items()))
        print(f"         Correlaciones: {lags_str}")

    return {
        "optimal_lag": optimal_lag,
        "max_correlation": max_corr,
        "contemporaneous_corr": contemp_corr,
        "pearson_r": r,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_samples": len(a),
        "n_aligned_minutes": n,
        "all_correlations": corrs,
    }


def run_lead_lag_analysis(
    markets: Dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    market_times: Dict[str, Dict],
    verbose: bool = True,
    show_per_market: int = 10,
) -> Dict:
    """
    Ejecuta analisis de lead-lag sobre todos los mercados.
    Retorna resumen agregado.
    """
    results = []
    significant_count = 0
    btc_leads_count = 0
    poly_leads_count = 0
    skipped = 0

    for name, poly_df in markets.items():
        times = market_times.get(name)
        if times is None:
            skipped += 1
            continue

        start_utc = pd.Timestamp(times["start_utc"])
        end_utc = pd.Timestamp(times["end_utc"])

        # Solo verbose para los primeros N mercados
        show_detail = verbose and len(results) < show_per_market

        analysis = analyze_lead_lag_per_market(
            poly_df, btc_df, start_utc, end_utc,
            market_name=name,
            verbose=show_detail,
        )
        if analysis is None:
            skipped += 1
            continue

        results.append({
            "market": name,
            **{k: v for k, v in analysis.items() if k != "all_correlations"},
        })

        if analysis["significant"]:
            significant_count += 1
            if analysis["optimal_lag"] > 0:
                btc_leads_count += 1
            elif analysis["optimal_lag"] < 0:
                poly_leads_count += 1

    if verbose:
        print(f"\n    Mercados saltados (sin datos suficientes): {skipped}")

    if not results:
        return {
            "status": "NO_DATA",
            "message": "No se pudo analizar lead-lag (sin datos alineados suficientes)",
        }

    results_df = pd.DataFrame(results)

    # Estadisticas
    avg_lag = results_df["optimal_lag"].mean()
    avg_corr = results_df["max_correlation"].mean()
    median_lag = results_df["optimal_lag"].median()
    avg_contemp = results_df["contemporaneous_corr"].mean()
    avg_samples = results_df["n_samples"].mean()
    avg_minutes = results_df["n_aligned_minutes"].mean()

    if verbose:
        # Distribucion de lags
        print(f"\n    Distribucion de lags optimos:")
        lag_counts = results_df["optimal_lag"].value_counts().sort_index()
        for lag_val, cnt in lag_counts.items():
            bar = "#" * cnt
            print(f"      lag={int(lag_val):+3d}: {cnt:3d} {bar}")

        # Distribucion de correlaciones
        print(f"\n    Distribucion de correlaciones maximas:")
        print(f"      min={results_df['max_correlation'].min():.4f}  "
              f"Q25={results_df['max_correlation'].quantile(0.25):.4f}  "
              f"median={results_df['max_correlation'].median():.4f}  "
              f"Q75={results_df['max_correlation'].quantile(0.75):.4f}  "
              f"max={results_df['max_correlation'].max():.4f}")

        # Top 5 mercados con mayor correlacion
        top5 = results_df.nlargest(5, "max_correlation")
        print(f"\n    Top 5 mercados con mayor correlacion:")
        for _, row in top5.iterrows():
            mname = row["market"][:50]
            print(f"      {mname}: lag={int(row['optimal_lag'])} "
                  f"corr={row['max_correlation']:.4f} "
                  f"p={row['p_value']:.4f} n={int(row['n_samples'])}")

    return {
        "status": "OK",
        "total_markets_analyzed": len(results),
        "skipped_markets": skipped,
        "significant_markets": significant_count,
        "significant_pct": significant_count / len(results) * 100,
        "btc_leads_count": btc_leads_count,
        "poly_leads_count": poly_leads_count,
        "avg_optimal_lag": avg_lag,
        "median_optimal_lag": median_lag,
        "avg_max_correlation": avg_corr,
        "avg_contemporaneous_corr": avg_contemp,
        "avg_samples_per_market": avg_samples,
        "avg_aligned_minutes": avg_minutes,
        "conclusion": _interpret_results(
            len(results), significant_count, btc_leads_count,
            avg_lag, avg_corr, avg_contemp, avg_samples,
        ),
        "details_df": results_df,
    }


def _interpret_results(
    total: int,
    significant: int,
    btc_leads: int,
    avg_lag: float,
    avg_corr: float,
    avg_contemp: float,
    avg_samples: float,
) -> str:
    """Interpreta los resultados del analisis lead-lag."""
    if total == 0:
        return "Sin datos suficientes para analisis."

    sig_pct = significant / total * 100
    btc_pct = btc_leads / max(significant, 1) * 100
    # Con N tests, esperamos ~5% significativos por azar
    expected_by_chance = total * 0.05

    lines = []

    lines.append(f"Muestras promedio por mercado: {avg_samples:.0f}")
    lines.append(
        f"Significativos esperados por azar (p<0.05): ~{expected_by_chance:.0f} "
        f"| Observados: {significant}"
    )

    if significant <= expected_by_chance * 1.5:
        lines.append(
            "ALERTA: La cantidad de mercados significativos NO supera lo "
            "esperado por azar. El lead-lag probablemente es RUIDO."
        )
    elif sig_pct > 60 and btc_pct > 60 and abs(avg_corr) > 0.15:
        lines.append(
            "LEAD-LAG CONFIRMADO: Binance lidera Polymarket "
            f"en {sig_pct:.0f}% de mercados."
        )
        lines.append(
            f"Lag promedio: {avg_lag:.1f} min | "
            f"Correlacion promedio: {avg_corr:.4f}"
        )
        lines.append("=> La estrategia de lead-lag ES viable.")
    elif sig_pct > 30 and abs(avg_corr) > 0.08:
        lines.append(
            f"LEAD-LAG PARCIAL: significativo en {sig_pct:.0f}% de mercados."
        )
        lines.append(
            f"Correlacion promedio: {avg_corr:.4f} (contemporanea: {avg_contemp:.4f})"
        )
        lines.append(
            "=> Usable como factor complementario, no como estrategia unica."
        )
    else:
        lines.append(
            f"LEAD-LAG DEBIL/INEXISTENTE: {sig_pct:.0f}% significativo, "
            f"corr promedio: {avg_corr:.4f}"
        )
        lines.append(
            "=> Enfocarse en estrategias basadas solo en Polymarket."
        )

    return "\n".join(lines)
