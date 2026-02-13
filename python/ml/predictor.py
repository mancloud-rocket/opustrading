"""
MarketPredictor v2: modelo predictivo con ponderacion de features.

Cambios respecto a v1:
  - Features: SOLO BTC (eliminadas up_price, spread, price_confidence por leakage)
  - Modelo: GradientBoostingClassifier (pondera features automaticamente)
  - Sample weights: recencia + ventana de entrada
  - Prediccion continua: se ejecuta en cada tick/iteracion

PONDERACION DE VARIABLES:
  GradientBoosting asigna peso a cada feature automaticamente segun su
  poder predictivo. A diferencia de Logistic Regression (pesos lineales),
  GBM captura interacciones no lineales entre features y pondera
  naturalmente las mas importantes via tree splits.

  Ademas, se aplican sample weights:
    - Mercados recientes pesan mas (x1.5) -- el mercado evoluciona
    - Minutos en ventana de entrada (5-10) pesan mas (x1.3) -- son los
      que realmente usamos para operar

Output: P(UP wins) -- score continuo entre 0 y 1.
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Optional, Dict, List, Tuple
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
)
from dataclasses import dataclass


# ============================================================================
# Feature names -- SOLO BTC (sin Polymarket leaky features)
#
# Las features up_price, spread, price_confidence fueron ELIMINADAS porque
# reflejan informacion que el mercado YA decidio, no prediccion genuina.
# El modelo debe predecir desde BTC, no leer lo que Polymarket ya sabe.
# ============================================================================

FEATURE_NAMES = [
    "btc_ret",            # BTC return from market start (signed)
    "btc_ret_1m",         # BTC return last 1 minute (signed)
    "btc_ret_3m",         # BTC return last 3 minutes (signed)
    "btc_vol",            # BTC realized vol (std of tick-to-tick returns)
    "btc_trend_r2",       # R-squared of linear fit to BTC prices (0-1)
    "btc_trend_slope",    # Normalized slope of BTC trend (signed)
    "btc_acceleration",   # Cambio en momentum: ret_1m - ret_prev_1m
    "btc_range_pct",      # (max - min) / start -- rango de precios
    "minute",             # Minuto actual en el mercado
]


@dataclass
class PredictionResult:
    """Resultado de una prediccion."""
    p_up: float              # P(UP wins), 0-1
    confidence: float        # max(p_up, 1-p_up), 0.5-1.0
    predicted_side: str      # "UP" or "DOWN"
    features: Dict           # features usadas
    max_entry_price: float   # precio maximo aceptable para el lado predicho
    feature_weights: Optional[Dict] = None  # ponderacion de cada feature


# ============================================================================
# MarketPredictor v2
# ============================================================================

class MarketPredictor:
    """
    Modelo predictivo v2 con ponderacion de features.

    Usa GradientBoostingClassifier que pondera features automaticamente
    via tree splits. Solo features de BTC (sin leakage de Polymarket).
    Predice P(UP wins) en cada tick/iteracion.

    Usage:
        # Training
        predictor = MarketPredictor()
        stats = predictor.train_from_csv("data/live_log.csv")
        predictor.save("ml/model.pkl")

        # Prediction (cada tick)
        predictor = MarketPredictor()
        predictor.load("ml/model.pkl")
        result = predictor.predict(features_dict)
    """

    def __init__(self):
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained: bool = False
        self.training_stats: Dict = {}
        self._feature_importances: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Feature extraction from historical data
    # ------------------------------------------------------------------

    def _extract_market_features(
        self,
        market_df: pd.DataFrame,
        market_name: str,
    ) -> List[Tuple[int, Dict, int, float]]:
        """
        Extrae features minuto a minuto para un mercado.

        Returns:
            Lista de (minute, features_dict, up_won, up_price_aux)
            up_price_aux se guarda aparte para simulacion (NO es feature)
        """
        df = market_df.sort_values("elapsed_s").copy()

        if len(df) < 30:
            return []

        df["btc_price"] = pd.to_numeric(df["btc_price"], errors="coerce")
        df["up_price"] = pd.to_numeric(df["up_price"], errors="coerce")
        df["down_price"] = pd.to_numeric(df["down_price"], errors="coerce")
        df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
        df = df.dropna(subset=["btc_price", "up_price", "down_price", "elapsed_s"])

        if len(df) < 20:
            return []

        # Determinar resolucion desde los ultimos ticks
        last_ticks = df[df["elapsed_s"] >= df["elapsed_s"].max() - 30]
        if last_ticks.empty:
            last_ticks = df.tail(10)

        final_up = last_ticks["up_price"].median()
        final_down = last_ticks["down_price"].median()

        if abs(final_up - final_down) < 0.05:
            return []  # Resolucion ambigua

        up_won = 1 if final_up > final_down else 0

        # BTC price al inicio del mercado
        early_ticks = df[df["elapsed_s"] <= 30]
        if early_ticks.empty:
            early_ticks = df.head(5)
        btc_start = early_ticks["btc_price"].median()

        if btc_start <= 0:
            return []

        # Extraer features a cada minuto
        results = []
        for minute in range(2, 13):  # minutos 2-12
            target_s = minute * 60
            window = df[df["elapsed_s"] <= target_s]

            if len(window) < 10:
                continue

            features = self._compute_features_from_window(
                window, btc_start, minute
            )
            if features is not None:
                # up_price auxiliar para simulacion de trading (NO es feature)
                up_price_aux = float(window["up_price"].iloc[-1])
                results.append((minute, features, up_won, up_price_aux))

        return results

    def _compute_features_from_window(
        self,
        window: pd.DataFrame,
        btc_start: float,
        minute: int,
    ) -> Optional[Dict]:
        """
        Calcula features SOLO de BTC (sin Polymarket).

        GradientBoosting pondera cada feature automaticamente segun
        su contribucion a la prediccion via tree splits.
        """
        btc_prices = window["btc_price"].values
        elapsed = window["elapsed_s"].values

        if len(btc_prices) < 5:
            return None

        btc_now = btc_prices[-1]

        # --- btc_ret: Return desde inicio del mercado ---
        btc_ret = (btc_now - btc_start) / btc_start

        # --- btc_ret_1m: Return ultimo minuto ---
        cutoff_1m = elapsed[-1] - 60
        mask_1m = elapsed > cutoff_1m
        if mask_1m.sum() > 1:
            btc_1m_start = btc_prices[mask_1m][0]
            btc_ret_1m = (btc_now - btc_1m_start) / btc_1m_start
        else:
            btc_ret_1m = 0.0

        # --- btc_ret_3m: Return ultimos 3 minutos ---
        cutoff_3m = elapsed[-1] - 180
        mask_3m = elapsed > cutoff_3m
        if mask_3m.sum() > 1:
            btc_3m_start = btc_prices[mask_3m][0]
            btc_ret_3m = (btc_now - btc_3m_start) / btc_3m_start
        else:
            btc_ret_3m = btc_ret  # Fallback a return total

        # --- btc_vol: Volatilidad tick-to-tick ---
        if len(btc_prices) > 3:
            tick_returns = np.diff(btc_prices) / btc_prices[:-1]
            btc_vol = float(np.std(tick_returns))
        else:
            btc_vol = 0.0

        # --- btc_trend: Regresion lineal sobre BTC prices ---
        if len(btc_prices) > 5:
            x = np.arange(len(btc_prices))
            try:
                slope, _, r_value, _, _ = linregress(x, btc_prices)
                btc_trend_r2 = r_value ** 2
                btc_trend_slope = slope / btc_start  # normalizado
            except Exception:
                btc_trend_r2 = 0.0
                btc_trend_slope = 0.0
        else:
            btc_trend_r2 = 0.0
            btc_trend_slope = 0.0

        # --- btc_acceleration: Cambio en momentum ---
        # ret_1m_actual - ret_1m_anterior = aceleracion
        cutoff_2m = elapsed[-1] - 120
        mask_prev_1m = (elapsed > cutoff_2m) & (elapsed <= cutoff_1m)
        if mask_1m.sum() > 1 and mask_prev_1m.sum() > 1:
            btc_prev_start = btc_prices[mask_prev_1m][0]
            btc_prev_end = btc_prices[mask_prev_1m][-1]
            if btc_prev_start > 0:
                ret_prev_1m = (btc_prev_end - btc_prev_start) / btc_prev_start
                btc_acceleration = btc_ret_1m - ret_prev_1m
            else:
                btc_acceleration = 0.0
        else:
            btc_acceleration = 0.0

        # --- btc_range_pct: Rango de precios como % ---
        btc_high = float(np.max(btc_prices))
        btc_low = float(np.min(btc_prices))
        btc_range_pct = (btc_high - btc_low) / btc_start if btc_start > 0 else 0.0

        return {
            "btc_ret": btc_ret,
            "btc_ret_1m": btc_ret_1m,
            "btc_ret_3m": btc_ret_3m,
            "btc_vol": btc_vol,
            "btc_trend_r2": btc_trend_r2,
            "btc_trend_slope": btc_trend_slope,
            "btc_acceleration": btc_acceleration,
            "btc_range_pct": btc_range_pct,
            "minute": minute,
        }

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

    def build_dataset(
        self, log_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye dataset de entrenamiento desde live_log.csv.

        Returns:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,) -- 1=UP won, 0=DOWN won
            groups: market IDs (n_samples,) -- para GroupKFold
            up_prices: precios UP auxiliares para simulacion (NO features)
        """
        print("Cargando live_log.csv...")
        # Header puede tener 10 o 12 columnas (ml_p_up, ml_confidence
        # se agregaron en v3.3). Usar engine python para tolerar filas
        # con diferente numero de campos.
        df = pd.read_csv(
            log_path,
            engine="python",
            on_bad_lines="skip",
        )

        markets = df["market"].unique()
        print(f"Mercados encontrados: {len(markets)}")

        all_features = []
        all_labels = []
        all_groups = []
        all_up_prices = []
        markets_used = 0
        markets_skipped = 0

        for i, market_name in enumerate(markets):
            market_df = df[df["market"] == market_name].copy()

            max_elapsed = market_df["elapsed_s"].max() if not market_df.empty else 0
            if max_elapsed < 600:  # Menos de 10 min de datos
                markets_skipped += 1
                continue

            results = self._extract_market_features(market_df, market_name)
            if not results:
                markets_skipped += 1
                continue

            for minute, features, up_won, up_price_aux in results:
                feature_vec = [features[name] for name in FEATURE_NAMES]
                all_features.append(feature_vec)
                all_labels.append(up_won)
                all_groups.append(i)
                all_up_prices.append(up_price_aux)

            markets_used += 1

        print(f"Mercados usados: {markets_used}, skipped: {markets_skipped}")
        if markets_used > 0:
            print(f"Samples totales: {len(all_features)} "
                  f"({len(all_features) // markets_used:.0f} per market)")

        X = np.array(all_features)
        y = np.array(all_labels)
        groups = np.array(all_groups)
        up_prices = np.array(all_up_prices)

        # Sanity check
        up_pct = y.mean() * 100
        print(f"Balance: {up_pct:.1f}% UP / {100-up_pct:.1f}% DOWN")

        return X, y, groups, up_prices

    # ------------------------------------------------------------------
    # Sample weights (ponderacion de muestras)
    # ------------------------------------------------------------------

    def _compute_sample_weights(
        self,
        X: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula pesos por sample para el entrenamiento.

        Ponderacion dual:
          1. Recencia: mercados recientes pesan mas (0.7x a 1.3x)
             -- el mercado evoluciona, datos recientes son mas relevantes
          2. Ventana de entrada: minutos 5-10 pesan 1.3x
             -- es donde realmente operamos, importa mas acertar ahi
        """
        weights = np.ones(len(X))

        # --- Peso por recencia ---
        unique_groups = np.unique(groups)
        n_markets = len(unique_groups)
        for i, g in enumerate(unique_groups):
            mask = groups == g
            # Linealmente de 0.7 (mas viejo) a 1.3 (mas reciente)
            recency_weight = 0.7 + 0.6 * (i / max(n_markets - 1, 1))
            weights[mask] *= recency_weight

        # --- Peso por ventana de entrada ---
        minute_idx = FEATURE_NAMES.index("minute")
        minutes = X[:, minute_idx]
        entry_window = (minutes >= 5) & (minutes <= 10)
        weights[entry_window] *= 1.3  # Entry window pesa mas

        return weights

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        up_prices: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Entrena GradientBoosting con GroupKFold + sample weights.

        Returns:
            Dict con metricas de cross-validation y ponderacion de features
        """
        print(f"\nEntrenando modelo GradientBoosting...")
        print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
        print(f"  Markets: {len(np.unique(groups))}")

        # Escalar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Sample weights (ponderacion de muestras)
        sample_weights = self._compute_sample_weights(X, groups)
        print(f"  Sample weights: min={sample_weights.min():.2f}, "
              f"max={sample_weights.max():.2f}, mean={sample_weights.mean():.2f}")

        # Cross-validation por mercado
        n_unique = len(np.unique(groups))
        n_splits = min(5, n_unique)

        if n_splits < 3:
            print("  WARN: Menos de 3 mercados, no se puede hacer CV.")
            self.model = self._create_model()
            self.model.fit(X_scaled, y, sample_weight=sample_weights)
            self.is_trained = True
            self._feature_importances = dict(
                zip(FEATURE_NAMES, self.model.feature_importances_)
            )
            return {"warning": "insufficient data for CV"}

        gkf = GroupKFold(n_splits=n_splits)

        fold_metrics = []
        all_y_true = []
        all_y_prob = []
        all_indices = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]

            model = self._create_model()
            model.fit(X_train, y_train, sample_weight=w_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5
            brier = brier_score_loss(y_test, y_prob)

            n_test_markets = len(np.unique(groups[test_idx]))
            fold_metrics.append({
                "fold": fold,
                "accuracy": acc,
                "auc": auc,
                "brier": brier,
                "n_test_samples": len(y_test),
                "n_test_markets": n_test_markets,
            })

            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob)
            all_indices.extend(test_idx)

        # Overall CV metrics
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_indices_arr = np.array(all_indices)

        cv_acc = accuracy_score(all_y_true, (all_y_prob > 0.5).astype(int))
        cv_auc = roc_auc_score(all_y_true, all_y_prob)
        cv_brier = brier_score_loss(all_y_true, all_y_prob)
        cv_logloss = log_loss(all_y_true, all_y_prob)

        # Calibration check
        calibration = self._calibration_check(all_y_true, all_y_prob)

        # Trading simulation (using auxiliary up_prices)
        sim_up_prices = up_prices[all_indices_arr] if up_prices is not None else None
        trading_sim = self._simulate_trading(
            all_y_true, all_y_prob, X[all_indices_arr],
            up_prices=sim_up_prices,
        )

        stats = {
            "cv_accuracy": cv_acc,
            "cv_auc": cv_auc,
            "cv_brier": cv_brier,
            "cv_logloss": cv_logloss,
            "fold_metrics": fold_metrics,
            "calibration": calibration,
            "trading_sim": trading_sim,
            "n_samples": len(X),
            "n_markets": len(np.unique(groups)),
            "n_features": X.shape[1],
        }

        # Train final model on ALL data
        self.model = self._create_model()
        self.model.fit(X_scaled, y, sample_weight=sample_weights)
        self.is_trained = True

        # Feature importances (ponderacion natural del GBM)
        self._feature_importances = dict(
            zip(FEATURE_NAMES, self.model.feature_importances_)
        )
        stats["feature_importances"] = {
            k: round(v, 4) for k, v in self._feature_importances.items()
        }

        self.training_stats = stats
        return stats

    def _create_model(self) -> GradientBoostingClassifier:
        """
        Crea instancia del modelo con hiperparametros conservadores.

        GradientBoosting pondera features automaticamente:
        - Arboles profundos = captura interacciones entre features
        - Subsample < 1.0 = reduce overfitting (stochastic GB)
        - min_samples_leaf alto = evita aprender ruido
        """
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

    def _calibration_check(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> List[Dict]:
        """Verifica calibracion: predicted P vs actual frequency."""
        bins = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
                (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
        results = []
        for lo, hi in bins:
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            actual = y_true[mask].mean()
            predicted = y_prob[mask].mean()
            results.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": int(mask.sum()),
                "predicted": round(predicted, 3),
                "actual": round(actual, 3),
            })
        return results

    def _simulate_trading(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X: np.ndarray,
        up_prices: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Simula trades basados en las predicciones del modelo.
        Usa up_prices auxiliares (separados de features).
        """
        BET = 20.0
        FEES = 0.80  # round trip

        results = {}
        for min_conf in [0.55, 0.60, 0.65, 0.70, 0.75]:
            trades = 0
            pnl = 0.0
            wins = 0

            for i in range(len(y_prob)):
                p_up = y_prob[i]
                confidence = max(p_up, 1 - p_up)

                if confidence < min_conf:
                    continue

                pred_up = p_up > 0.5
                actual_up = y_true[i] == 1

                # Token price from auxiliary data (no es feature)
                if up_prices is not None:
                    up_price = up_prices[i]
                else:
                    up_price = 0.50  # Default si no hay datos

                if pred_up:
                    entry_price = up_price
                else:
                    entry_price = 1.0 - up_price

                if entry_price < 0.10 or entry_price > 0.90:
                    continue

                trades += 1

                if pred_up == actual_up:
                    # Win: token goes to ~1.00
                    gross = BET * (0.97 - entry_price) / entry_price
                    pnl += gross - FEES
                    wins += 1
                else:
                    # Loss: token goes to ~0.00
                    pnl += -BET - FEES

            wr = wins / trades * 100 if trades > 0 else 0
            results[f"conf_{min_conf}"] = {
                "trades": trades,
                "wins": wins,
                "wr": round(wr, 1),
                "pnl": round(pnl, 2),
            }

        return results

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------

    def train_from_csv(self, log_path: str) -> Dict:
        """Pipeline completo: cargar datos -> extraer features -> entrenar."""
        X, y, groups, up_prices = self.build_dataset(log_path)
        stats = self.train(X, y, groups, up_prices)
        return stats

    # ------------------------------------------------------------------
    # Prediction (cada tick/iteracion)
    # ------------------------------------------------------------------

    def predict(self, features: Dict) -> Optional[PredictionResult]:
        """
        Predice P(UP wins) dado features del momento actual.
        Diseñado para ejecutarse en CADA tick/iteracion.

        Args:
            features: dict con las mismas keys que FEATURE_NAMES

        Returns:
            PredictionResult o None si el modelo no esta entrenado
        """
        if not self.is_trained or self.model is None or self.scaler is None:
            return None

        # Build feature vector
        try:
            vec = np.array([[features[name] for name in FEATURE_NAMES]])
        except KeyError:
            return None

        # Scale and predict
        vec_scaled = self.scaler.transform(vec)
        p_up = float(self.model.predict_proba(vec_scaled)[0, 1])

        confidence = max(p_up, 1 - p_up)
        predicted_side = "UP" if p_up > 0.5 else "DOWN"

        # Max entry price: conservador, controlado por settings
        max_price = 0.60  # Default, Trader lo sobreescribe desde settings

        return PredictionResult(
            p_up=p_up,
            confidence=confidence,
            predicted_side=predicted_side,
            features=features,
            max_entry_price=max_price,
            feature_weights=self._feature_importances,
        )

    def compute_live_features(
        self,
        btc_ticks: List[Tuple[float, float]],
        poly_ticks: List[Tuple[float, float, float]],
        btc_start: float,
        elapsed: float,
    ) -> Optional[Dict]:
        """
        Calcula features para prediccion en vivo.
        Ejecutado en CADA tick para prediccion continua.

        Solo usa BTC data -- Polymarket features eliminadas del modelo.

        Args:
            btc_ticks: lista de (elapsed_s, btc_price)
            poly_ticks: lista de (elapsed_s, up_price, down_price) [no usados]
            btc_start: BTC price al inicio del mercado
            elapsed: segundos transcurridos

        Returns:
            Dict de features o None si datos insuficientes
        """
        if len(btc_ticks) < 5:
            return None
        if btc_start <= 0:
            return None

        minute = elapsed / 60.0

        # BTC prices array
        btc_times = np.array([t[0] for t in btc_ticks])
        btc_prices = np.array([t[1] for t in btc_ticks])
        btc_now = btc_prices[-1]

        # --- btc_ret: Return desde inicio ---
        btc_ret = (btc_now - btc_start) / btc_start

        # --- btc_ret_1m: Return ultimo minuto ---
        cutoff_1m = elapsed - 60
        mask_1m = btc_times > cutoff_1m
        if mask_1m.sum() > 1:
            btc_ret_1m = (btc_now - btc_prices[mask_1m][0]) / btc_prices[mask_1m][0]
        else:
            btc_ret_1m = 0.0

        # --- btc_ret_3m: Return ultimos 3 minutos ---
        cutoff_3m = elapsed - 180
        mask_3m = btc_times > cutoff_3m
        if mask_3m.sum() > 1:
            btc_ret_3m = (btc_now - btc_prices[mask_3m][0]) / btc_prices[mask_3m][0]
        else:
            btc_ret_3m = btc_ret

        # --- btc_vol: Volatilidad ---
        if len(btc_prices) > 3:
            tick_returns = np.diff(btc_prices) / btc_prices[:-1]
            btc_vol = float(np.std(tick_returns))
        else:
            btc_vol = 0.0

        # --- btc_trend: Regresion lineal ---
        if len(btc_prices) > 5:
            x = np.arange(len(btc_prices))
            try:
                slope, _, r_value, _, _ = linregress(x, btc_prices)
                btc_trend_r2 = r_value ** 2
                btc_trend_slope = slope / btc_start
            except Exception:
                btc_trend_r2 = 0.0
                btc_trend_slope = 0.0
        else:
            btc_trend_r2 = 0.0
            btc_trend_slope = 0.0

        # --- btc_acceleration: Cambio en momentum ---
        cutoff_2m = elapsed - 120
        mask_prev_1m = (btc_times > cutoff_2m) & (btc_times <= cutoff_1m)
        if mask_1m.sum() > 1 and mask_prev_1m.sum() > 1:
            btc_prev_start = btc_prices[mask_prev_1m][0]
            btc_prev_end = btc_prices[mask_prev_1m][-1]
            if btc_prev_start > 0:
                ret_prev_1m = (btc_prev_end - btc_prev_start) / btc_prev_start
                btc_acceleration = btc_ret_1m - ret_prev_1m
            else:
                btc_acceleration = 0.0
        else:
            btc_acceleration = 0.0

        # --- btc_range_pct: Rango de precios ---
        btc_high = float(np.max(btc_prices))
        btc_low = float(np.min(btc_prices))
        btc_range_pct = (btc_high - btc_low) / btc_start if btc_start > 0 else 0.0

        return {
            "btc_ret": btc_ret,
            "btc_ret_1m": btc_ret_1m,
            "btc_ret_3m": btc_ret_3m,
            "btc_vol": btc_vol,
            "btc_trend_r2": btc_trend_r2,
            "btc_trend_slope": btc_trend_slope,
            "btc_acceleration": btc_acceleration,
            "btc_range_pct": btc_range_pct,
            "minute": minute,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Guarda modelo entrenado a disco."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": FEATURE_NAMES,
            "training_stats": self.training_stats,
            "is_trained": self.is_trained,
            "feature_importances": self._feature_importances,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Modelo guardado en {path}")

    def load(self, path: str) -> bool:
        """Carga modelo entrenado desde disco."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.is_trained = data["is_trained"]
            self.training_stats = data.get("training_stats", {})
            self._feature_importances = data.get("feature_importances")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_feature_importance(self):
        """Imprime ponderacion/importancia de cada feature (GBM)."""
        if not self.is_trained or self.model is None:
            print("Modelo no entrenado.")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nPONDERACION DE FEATURES (importancia GradientBoosting):")
        print(f"  GBM asigna peso automaticamente segun poder predictivo.")
        print(f"  Mayor % = feature mas influyente en la prediccion.\n")
        print(f"{'#':>3} {'Feature':<20} {'Peso':>8} {'Barra'}")
        print("-" * 60)

        max_imp = importances[indices[0]] if len(indices) > 0 else 1.0
        for rank, idx in enumerate(indices):
            name = FEATURE_NAMES[idx]
            imp = importances[idx]
            bar_len = int(imp / max_imp * 30) if max_imp > 0 else 0
            bar = "#" * bar_len
            pct = imp * 100
            print(f"{rank+1:>3} {name:<20} {pct:>7.1f}%  {bar}")

        print(f"\n  Total features: {len(FEATURE_NAMES)}")
        print(f"  Modelo: GradientBoosting ({self.model.n_estimators} arboles, "
              f"depth {self.model.max_depth})")
