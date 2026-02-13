"""
Genera datos de prediccion del modelo ML como JSON y JS para el frontend.

Genera dos archivos:
  - chart_data.json: para uso via fetch (HTTP server)
  - chart_data.js: para uso directo via script tag (file://)

Incluye DOS tipos de predicciones:
  1. Model (in-sample): predicciones del modelo reentrenado sobre datos historicos
  2. Live: predicciones REALES que el bot hizo en tiempo real (del live_log.csv)

Uso: python generate_chart_data.py
"""
import sys
import os
import csv
import json
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from ml.predictor import MarketPredictor, FEATURE_NAMES


def extract_live_predictions(log_path: str) -> dict:
    """
    Extrae las predicciones REALES del bot (ml_p_up) desde live_log.csv.

    Retorna dict con:
      - live_predictions_by_minute: { "5": [...], "6": [...], ... }
      - live_stats: resumen de accuracy live
    """
    # Leer CSV con csv.reader para manejar 10 y 12 columnas
    markets = defaultdict(list)  # market_name -> list of row dicts

    with open(log_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) < 12:
                continue  # Solo filas con ml_p_up
            try:
                ml_p_up_str = row[10].strip()
                ml_conf_str = row[11].strip()
                if not ml_p_up_str or not ml_conf_str:
                    continue
                ml_p_up = float(ml_p_up_str)
                ml_conf = float(ml_conf_str)
            except (ValueError, IndexError):
                continue

            market = row[1]
            try:
                elapsed = float(row[2])
                up_price = float(row[3]) if row[3] else None
                down_price = float(row[4]) if row[4] else None
                btc_price = float(row[5]) if row[5] else None
                btc_return = float(row[6]) if row[6] else None
            except ValueError:
                continue

            if up_price is None or down_price is None:
                continue

            markets[market].append({
                "elapsed": elapsed,
                "up_price": up_price,
                "down_price": down_price,
                "btc_price": btc_price,
                "btc_return": btc_return,
                "ml_p_up": ml_p_up,
                "ml_confidence": ml_conf,
            })

    print(f"\n=== LIVE PREDICTIONS ===")
    print(f"Mercados con ML data: {len(markets)}")

    # Para cada mercado, determinar resolucion y extraer puntos por minuto
    live_by_minute = defaultdict(list)
    total_correct = 0
    total_pts = 0

    for market_name, ticks in markets.items():
        if len(ticks) < 20:
            continue

        # Determinar resolucion desde los ultimos ticks
        ticks_sorted = sorted(ticks, key=lambda t: t["elapsed"])
        last_ticks = [t for t in ticks_sorted if t["elapsed"] >= ticks_sorted[-1]["elapsed"] - 60]
        if not last_ticks:
            last_ticks = ticks_sorted[-10:]

        final_up = np.median([t["up_price"] for t in last_ticks])
        final_down = np.median([t["down_price"] for t in last_ticks])

        if abs(final_up - final_down) < 0.10:
            continue  # Resolucion ambigua

        up_won = 1 if final_up > final_down else 0

        # Agrupar por minuto y tomar mediana de predicciones por minuto
        minute_groups = defaultdict(list)
        for t in ticks_sorted:
            minute = int(t["elapsed"] // 60)
            if 2 <= minute <= 12:
                minute_groups[minute].append(t)

        for minute, m_ticks in minute_groups.items():
            p_ups = [t["ml_p_up"] for t in m_ticks]
            confs = [t["ml_confidence"] for t in m_ticks]
            up_prices_m = [t["up_price"] for t in m_ticks]
            btc_rets = [t["btc_return"] for t in m_ticks if t["btc_return"] is not None]

            # Usar mediana del minuto (mas estable que un tick individual)
            median_p_up = float(np.median(p_ups))
            median_conf = float(np.median(confs))
            median_up_price = float(np.median(up_prices_m))
            median_btc_ret = float(np.median(btc_rets)) if btc_rets else 0.0

            # Accuracy check
            pred_up = median_p_up > 0.5
            is_correct = pred_up == (up_won == 1)
            total_correct += int(is_correct)
            total_pts += 1

            point = {
                "market": market_name[-30:],  # Nombre corto
                "p_up": round(median_p_up, 4),
                "confidence": round(median_conf, 4),
                "actual": up_won,
                "btc_ret": round(median_btc_ret * 100, 4),
                "up_price": round(median_up_price, 3),
                "minute": minute,
                "n_ticks": len(m_ticks),
            }

            live_by_minute[str(minute)].append(point)

    live_acc = (total_correct / total_pts * 100) if total_pts > 0 else 0
    print(f"Puntos con prediccion live: {total_pts}")
    print(f"Accuracy live: {live_acc:.1f}%")

    # Stats por minuto
    live_minute_stats = []
    for m in range(2, 13):
        pts = live_by_minute.get(str(m), [])
        if not pts:
            continue
        correct = sum(1 for p in pts if (p["p_up"] > 0.5) == (p["actual"] == 1))
        acc = correct / len(pts) * 100
        avg_conf = np.mean([p["confidence"] for p in pts]) * 100
        live_minute_stats.append({
            "minute": m,
            "accuracy": round(acc, 1),
            "avg_confidence": round(avg_conf, 1),
            "count": len(pts),
        })
        print(f"  Min {m}: {len(pts)} pts, Acc: {acc:.1f}%, Conf: {avg_conf:.1f}%")

    return {
        "live_predictions_by_minute": dict(live_by_minute),
        "live_stats": {
            "total_points": total_pts,
            "total_markets": len(markets),
            "accuracy": round(live_acc, 1),
        },
        "live_minute_stats": live_minute_stats,
    }


def main():
    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "live_log.csv")
    output_json = os.path.join(os.path.dirname(__file__), "..", "frontend", "chart_data.json")
    output_js = os.path.join(os.path.dirname(__file__), "..", "frontend", "chart_data.js")

    predictor = MarketPredictor()
    model_path = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")
    if not predictor.load(model_path):
        print("ERROR: No se encontro modelo ML. Ejecuta: python train_model.py")
        sys.exit(1)

    print("Cargando live_log.csv...")
    if not os.path.exists(log_path):
        print(f"ERROR: No se encontro {log_path}")
        sys.exit(1)

    # ============================================================
    # 1. LIVE PREDICTIONS (reales, del bot en tiempo real)
    # ============================================================
    live_data = extract_live_predictions(log_path)

    # ============================================================
    # 2. MODEL PREDICTIONS (in-sample, del modelo reentrenado)
    # ============================================================
    # Build dataset (4 return values en v2)
    X, y, groups, up_prices = predictor.build_dataset(log_path)
    X_scaled = predictor.scaler.transform(X)
    probs = predictor.model.predict_proba(X_scaled)[:, 1]

    # --- Per-market predictions at EVERY minute (for dropdown selector) ---
    # Estructura: { minute: [ {market_id, p_up, actual, ...}, ... ] }
    market_points_by_minute = {}
    unique_groups = np.unique(groups)

    for g in unique_groups:
        mask = groups == g
        X_g = X[mask]
        y_g = y[mask]
        p_g = probs[mask]
        up_g = up_prices[mask]
        minutes = X_g[:, FEATURE_NAMES.index("minute")]
        up_won = int(y_g[0])

        for idx in range(len(X_g)):
            minute = int(minutes[idx])
            p_up = float(p_g[idx])
            btc_ret = float(X_g[idx, FEATURE_NAMES.index("btc_ret")])
            up_price = float(up_g[idx])
            btc_vol = float(X_g[idx, FEATURE_NAMES.index("btc_vol")])

            point = {
                "market_id": int(g),
                "p_up": round(p_up, 4),
                "actual": up_won,
                "btc_ret": round(btc_ret * 100, 4),
                "up_price": round(up_price, 3),
                "btc_vol": round(btc_vol, 6),
                "minute": minute,
            }

            key = str(minute)
            if key not in market_points_by_minute:
                market_points_by_minute[key] = []
            market_points_by_minute[key].append(point)

    # Compatibilidad: market_predictions = minuto 7 (default)
    market_points = market_points_by_minute.get("7", [])

    # --- Calibration data ---
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    calibration = []
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        calibration.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "bin_mid": round((lo + hi) / 2, 2),
            "predicted": round(float(probs[mask].mean()), 4),
            "actual": round(float(y[mask].mean()), 4),
            "count": int(mask.sum()),
        })

    # --- Minute-by-minute accuracy ---
    minute_stats = []
    for m in range(2, 13):
        mask = X[:, FEATURE_NAMES.index("minute")] == m
        if mask.sum() == 0:
            continue
        p_m = probs[mask]
        y_m = y[mask]
        preds = (p_m > 0.5).astype(int)
        acc = float((preds == y_m).mean())
        avg_conf = float(np.maximum(p_m, 1 - p_m).mean())
        minute_stats.append({
            "minute": int(m),
            "accuracy": round(acc * 100, 1),
            "avg_confidence": round(avg_conf * 100, 1),
            "count": int(mask.sum()),
        })

    # --- PnL simulation by confidence threshold ---
    BET = 20.0
    FEES = 0.80
    pnl_curves = {}
    for min_conf in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        cumulative = []
        running_pnl = 0.0
        trade_num = 0
        for i in range(len(probs)):
            p = probs[i]
            conf = max(p, 1 - p)
            if conf < min_conf:
                continue
            pred_up = p > 0.5
            actual_up = y[i] == 1

            # Token price from auxiliary data (not features)
            token = up_prices[i] if pred_up else (1.0 - up_prices[i])
            if token < 0.10 or token > 0.75:
                continue

            trade_num += 1
            if pred_up == actual_up:
                gross = BET * (0.97 - token) / token
                running_pnl += gross - FEES
            else:
                running_pnl += -BET - FEES
            cumulative.append({
                "trade": trade_num,
                "pnl": round(running_pnl, 2),
            })
        pnl_curves[f"{int(min_conf*100)}%"] = cumulative

    # --- Feature importance (ponderacion GBM) ---
    importances = predictor.model.feature_importances_
    feature_importance = []
    for i, name in enumerate(FEATURE_NAMES):
        feature_importance.append({
            "feature": name,
            "coefficient": round(float(importances[i]), 4),
            "abs_coefficient": round(float(importances[i]), 4),
        })
    feature_importance.sort(key=lambda x: x["abs_coefficient"], reverse=True)

    # --- Distribution of P(UP) ---
    dist_up_won = [round(float(x), 3) for x in probs[y == 1]]
    dist_down_won = [round(float(x), 3) for x in probs[y == 0]]

    # --- Training stats ---
    stats = predictor.training_stats

    # Assemble output
    output = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_type": "GradientBoosting",
        "model_stats": {
            "cv_accuracy": round(stats.get("cv_accuracy", 0) * 100, 1),
            "cv_auc": round(stats.get("cv_auc", 0), 3),
            "cv_brier": round(stats.get("cv_brier", 0), 4),
            "n_markets": stats.get("n_markets", 0),
            "n_samples": stats.get("n_samples", 0),
        },
        "market_predictions": market_points,
        "market_predictions_by_minute": market_points_by_minute,
        # LIVE predictions (reales del bot)
        "live_predictions_by_minute": live_data["live_predictions_by_minute"],
        "live_stats": live_data["live_stats"],
        "live_minute_stats": live_data["live_minute_stats"],
        # Model analytics
        "calibration": calibration,
        "minute_accuracy": minute_stats,
        "pnl_curves": pnl_curves,
        "feature_importance": feature_importance,
        "distribution_up_won": dist_up_won,
        "distribution_down_won": dist_down_won,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # --- Write JSON (para fetch) ---
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    # --- Write JS (para script tag, funciona con file://) ---
    with open(output_js, "w") as f:
        f.write("// Auto-generated by generate_chart_data.py\n")
        f.write("var CHART_DATA = ")
        json.dump(output, f, indent=2)
        f.write(";\n")

    print(f"Datos generados:")
    print(f"  JSON: {output_json}")
    print(f"  JS:   {output_js}")
    print(f"  Markets: {len(market_points)}")
    print(f"  Samples: {len(probs)}")
    print(f"  Modelo: GradientBoosting (features solo BTC)")


if __name__ == "__main__":
    main()
