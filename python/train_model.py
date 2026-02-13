"""
Script de entrenamiento del modelo predictivo v2.

Lee live_log.csv, extrae features BTC (sin leakage Polymarket),
entrena GradientBoosting con ponderacion automatica de features
y sample weights, cross-validation por mercado.

Uso:
    cd python
    python train_model.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ml.predictor import MarketPredictor, FEATURE_NAMES


def main():
    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "live_log.csv")
    model_path = os.path.join(os.path.dirname(__file__), "ml", "model.pkl")

    if not os.path.exists(log_path):
        print(f"ERROR: No se encontro {log_path}")
        print("Ejecuta el bot live primero para generar datos.")
        sys.exit(1)

    print("=" * 70)
    print("  ENTRENAMIENTO DE MODELO PREDICTIVO v2")
    print("  GradientBoosting + Ponderacion de Features")
    print("  Features: SOLO BTC (sin leakage Polymarket)")
    print("=" * 70)

    # ---- Build dataset ----
    predictor = MarketPredictor()
    X, y, groups, up_prices = predictor.build_dataset(log_path)

    n_markets = len(np.unique(groups))
    if n_markets < 10:
        print(f"\nWARN: Solo {n_markets} mercados. Minimo recomendado: 20+.")
        print("Los resultados pueden no ser confiables.")

    # ---- Feature stats ----
    print(f"\n{'='*70}")
    print("ESTADISTICAS DE FEATURES (solo BTC)")
    print(f"{'='*70}")
    print(f"\n{'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 65)
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        print(f"{name:<20} {col.mean():>10.6f} {col.std():>10.6f} "
              f"{col.min():>10.6f} {col.max():>10.6f}")

    # ---- Train ----
    stats = predictor.train(X, y, groups, up_prices)

    # ---- Results ----
    print(f"\n{'='*70}")
    print("RESULTADOS DE CROSS-VALIDATION")
    print(f"{'='*70}")

    print(f"\n  Accuracy (CV):   {stats['cv_accuracy']*100:.1f}%")
    print(f"  AUC (CV):        {stats['cv_auc']:.3f}")
    print(f"  Brier Score:     {stats['cv_brier']:.4f}  (menor = mejor, 0.25 = azar)")
    print(f"  Log Loss:        {stats['cv_logloss']:.4f}")

    # Per-fold breakdown
    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':>5} {'Acc':>7} {'AUC':>7} {'Brier':>7} {'Markets':>8} {'Samples':>8}")
    print(f"  {'-'*45}")
    for fm in stats["fold_metrics"]:
        print(f"  {fm['fold']:>5} {fm['accuracy']*100:>6.1f}% "
              f"{fm['auc']:>6.3f} {fm['brier']:>6.4f} "
              f"{fm['n_test_markets']:>8} {fm['n_test_samples']:>8}")

    # Calibration
    print(f"\n  Calibracion (predicted vs actual):")
    print(f"  {'Bin':>10} {'Count':>6} {'Predicted':>10} {'Actual':>10} {'Diff':>8}")
    print(f"  {'-'*48}")
    for c in stats["calibration"]:
        diff = c["actual"] - c["predicted"]
        print(f"  {c['bin']:>10} {c['count']:>6} "
              f"{c['predicted']:>10.3f} {c['actual']:>10.3f} "
              f"{diff:>+7.3f}")

    # Trading simulation
    print(f"\n  Simulacion de trading (con precios reales de tokens):")
    print(f"  {'Confianza':>10} {'Trades':>7} {'Wins':>5} {'WR':>7} {'PnL':>10}")
    print(f"  {'-'*42}")
    for key, sim in sorted(stats["trading_sim"].items()):
        conf = key.replace("conf_", "")
        pnl_str = f"${sim['pnl']:+.2f}"
        print(f"  >={conf:>8} {sim['trades']:>7} {sim['wins']:>5} "
              f"{sim['wr']:>6.1f}% {pnl_str:>10}")

    # Feature importance (PONDERACION)
    predictor.print_feature_importance()

    # ---- Feature weights detail ----
    if "feature_importances" in stats:
        print(f"\n  PESOS DEL MODELO (dict para referencia):")
        for name, weight in sorted(
            stats["feature_importances"].items(),
            key=lambda x: -x[1],
        ):
            print(f"    {name:<20} = {weight:.4f}")

    # ---- Sample predictions ----
    print(f"\n{'='*70}")
    print("EJEMPLOS DE PREDICCION (primeros 5 mercados)")
    print(f"{'='*70}")

    unique_groups = np.unique(groups)
    for g in unique_groups[:5]:
        mask = groups == g
        X_market = X[mask]
        y_market = y[mask]

        # Predict
        X_scaled = predictor.scaler.transform(X_market)
        probs = predictor.model.predict_proba(X_scaled)[:, 1]

        actual = "UP" if y_market[0] == 1 else "DOWN"
        print(f"\n  Market #{g} (resolvio: {actual})")
        print(f"  {'Min':>5} {'P(UP)':>7} {'Conf':>6} {'Pred':>5} {'btc_ret':>10} "
              f"{'btc_vol':>9} {'accel':>9}")
        for i in range(len(X_market)):
            p_up = probs[i]
            conf = max(p_up, 1 - p_up)
            pred = "UP" if p_up > 0.5 else "DN"
            minute = X_market[i, FEATURE_NAMES.index("minute")]
            btc_ret = X_market[i, FEATURE_NAMES.index("btc_ret")]
            btc_vol = X_market[i, FEATURE_NAMES.index("btc_vol")]
            accel = X_market[i, FEATURE_NAMES.index("btc_acceleration")]
            correct = "ok" if (p_up > 0.5) == (y_market[i] == 1) else "XX"
            print(f"  {minute:>5.0f} {p_up:>7.3f} {conf:>5.1%} {pred:>5} "
                  f"{btc_ret*100:>+9.3f}% {btc_vol:>9.6f} {accel*100:>+8.4f}%  {correct}")

    # ---- Save ----
    predictor.save(model_path)

    print(f"\n{'='*70}")
    print(f"  MODELO GUARDADO: {model_path}")
    print(f"  Tipo: GradientBoosting (ponderacion automatica de features)")
    print(f"  Features: {len(FEATURE_NAMES)} (solo BTC, sin leakage)")
    print(f"  Mercados de entrenamiento: {n_markets}")
    print(f"  CV Accuracy: {stats['cv_accuracy']*100:.1f}%")
    print(f"  CV AUC: {stats['cv_auc']:.3f}")
    print(f"{'='*70}")

    # ---- Recomendaciones ----
    print(f"\n  RECOMENDACIONES:")
    if stats["cv_auc"] >= 0.80:
        print(f"  -> AUC {stats['cv_auc']:.3f} es EXCELENTE. Modelo listo para uso.")
    elif stats["cv_auc"] >= 0.70:
        print(f"  -> AUC {stats['cv_auc']:.3f} es BUENO. Usar con confianza >= 0.65.")
    elif stats["cv_auc"] >= 0.60:
        print(f"  -> AUC {stats['cv_auc']:.3f} es MODERADO. Usar con confianza >= 0.70.")
    else:
        print(f"  -> AUC {stats['cv_auc']:.3f} es BAJO. Necesita mas datos.")
        print(f"  -> SIN features leaky, el AUC real deberia ser mas honesto.")
        print(f"  -> Esto refleja la verdadera capacidad predictiva del BTC.")

    # Best trading sim
    best_sim = None
    best_pnl = -999
    for key, sim in stats["trading_sim"].items():
        if sim["pnl"] > best_pnl and sim["trades"] >= 3:
            best_pnl = sim["pnl"]
            best_sim = key

    if best_sim:
        sim = stats["trading_sim"][best_sim]
        conf = best_sim.replace("conf_", "")
        print(f"  -> Mejor config de trading: confianza >= {conf} "
              f"({sim['trades']}T, {sim['wr']:.0f}% WR, ${sim['pnl']:+.2f})")


if __name__ == "__main__":
    main()
