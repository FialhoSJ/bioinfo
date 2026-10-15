"""Prespecified forecasting families and compact tuning grids."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .data import make_feature_row, make_supervised


MODEL_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "lag_12",
    "media_3", "media_6", "media_12", "tendencia_3m", "mes_sin", "mes_cos",
]
COUNT_FEATURES = ["lag_1", "lag_3", "lag_12", "mes_sin", "mes_cos"]


def candidate_configurations() -> dict[str, list[dict[str, Any]]]:
    """Return small, prespecified grids for selection on the training half only."""
    return {
        "naive_ultimo_mes": [{}],
        "media_movel_3m": [{}],
        "naive_sazonal_12m": [{}],
        "ETS": [
            {"trend": None, "seasonal": None, "damped_trend": False},
            {"trend": "add", "seasonal": None, "damped_trend": True},
            {"trend": "add", "seasonal": "add", "damped_trend": True},
        ],
        "ARIMA": [
            {"order": (1, 0, 0)},
            {"order": (0, 1, 1)},
            {"order": (1, 1, 1)},
        ],
        "SARIMA": [
            {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 12)},
            {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 12)},
        ],
        "Poisson_lags": [
            {"ridge_alpha": 0.001, "link": "log"},
            {"ridge_alpha": 0.05, "link": "log"},
            {"ridge_alpha": 0.001, "link": "identity"},
            {"ridge_alpha": 0.05, "link": "identity"},
        ],
        "NegBin_lags": [
            {"dispersion_alpha": 0.1},
            {"dispersion_alpha": 0.5},
            {"dispersion_alpha": 1.0},
        ],
        "Random_Forest": [
            {"max_depth": 3, "min_samples_leaf": 3},
            {"max_depth": 5, "min_samples_leaf": 3},
            {"max_depth": None, "min_samples_leaf": 5},
        ],
        "Boosting_Poisson": [
            {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 3},
            {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 5},
        ],
        "SVR": [
            {"C": 1.0, "epsilon": 0.1},
            {"C": 10.0, "epsilon": 0.1},
            {"C": 50.0, "epsilon": 1.0},
        ],
    }


def forecast_one_step(
    method: str,
    config: dict[str, Any],
    observed_counts: pd.Series,
    supervised: pd.DataFrame,
    target_month: pd.Timestamp,
    seed: int = 42,
) -> float:
    """Fit a selected method on information available by an origin and forecast 1 month."""
    history = observed_counts.loc[observed_counts.index < target_month].astype(float)
    if history.empty:
        raise ValueError(f"Histórico vazio antes de {target_month:%Y-%m}.")

    if method == "naive_ultimo_mes":
        return float(history.iloc[-1])
    if method == "media_movel_3m":
        return float(history.iloc[-min(3, len(history)):].mean())
    if method == "naive_sazonal_12m":
        seasonal_month = target_month - pd.DateOffset(months=12)
        if seasonal_month not in history.index:
            return float(history.iloc[-1])
        return float(history.loc[seasonal_month])
    if method in {"ETS", "ARIMA", "SARIMA"}:
        return _forecast_statsmodels(method, config, history)

    rows = supervised.loc[supervised.index < target_month].dropna(subset=["target"])
    if rows.empty:
        raise ValueError(f"Sem linhas de treinamento para {method} antes de {target_month:%Y-%m}.")
    x_train = rows[MODEL_FEATURES].to_numpy(dtype=float)
    y_train = rows["target"].to_numpy(dtype=float)
    x_next = make_feature_row(history, target_month)[MODEL_FEATURES].to_numpy(dtype=float)

    if method in {"Poisson_lags", "NegBin_lags"}:
        return _forecast_count_glm(method, config, rows, x_next)
    if method == "Random_Forest":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=1.0,
            random_state=seed,
            n_jobs=-1,
        )
    elif method == "Boosting_Poisson":
        estimator = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=config["learning_rate"],
            max_iter=config["max_iter"],
            max_leaf_nodes=config["max_leaf_nodes"],
            l2_regularization=1.0,
            random_state=seed,
        )
    elif method == "SVR":
        estimator = make_pipeline(
            StandardScaler(),
            SVR(C=config["C"], epsilon=config["epsilon"], kernel="rbf"),
        )
    else:
        raise ValueError(f"Método desconhecido: {method}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        estimator.fit(x_train, y_train)
    prediction = float(estimator.predict(x_next)[0])
    return max(0.0, prediction)


def _forecast_statsmodels(method: str, config: dict[str, Any], history: pd.Series) -> float:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError(
            "Os modelos ETS/ARIMA/SARIMA requerem statsmodels. "
            "Instale as dependências com: python -m pip install -r requirements.txt"
        ) from exc

    values = history.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "ETS":
            seasonal = config["seasonal"]
            model = ExponentialSmoothing(
                values,
                trend=config["trend"],
                damped_trend=bool(config["damped_trend"]),
                seasonal=seasonal,
                seasonal_periods=12 if seasonal else None,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            prediction = float(np.asarray(fitted.forecast(1))[0])
        else:
            order = tuple(config["order"])
            seasonal_order = (
                tuple(config["seasonal_order"])
                if method == "SARIMA" else (0, 0, 0, 0)
            )
            trend = "c" if order[1] == 0 else "n"
            fitted = SARIMAX(
                values,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            prediction = float(np.asarray(fitted.forecast(1))[0])
    return max(0.0, prediction)


def _forecast_count_glm(
    method: str,
    config: dict[str, Any],
    training_rows: pd.DataFrame,
    x_next: np.ndarray,
) -> float:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError(
            "Os modelos de contagem requerem statsmodels. "
            "Instale as dependências com: python -m pip install -r requirements.txt"
        ) from exc

    x_columns = COUNT_FEATURES
    x_train = training_rows[x_columns].copy()
    x_test = pd.DataFrame([{
        column: x_next[0, MODEL_FEATURES.index(column)]
        for column in x_columns
    }])
    # Scaling keeps the count lags on a useful numerical range for the log link.
    for column in ("lag_1", "lag_3", "lag_12"):
        x_train[column] = x_train[column] / 100.0
        x_test[column] = x_test[column] / 100.0
    x_train = sm.add_constant(x_train, has_constant="add")
    x_test = sm.add_constant(x_test, has_constant="add")
    y_train = training_rows["target"].to_numpy(dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "Poisson_lags":
            link = (
                sm.families.links.Identity()
                if config["link"] == "identity"
                else sm.families.links.Log()
            )
            model = sm.GLM(y_train, x_train, family=sm.families.Poisson(link=link))
            if config["ridge_alpha"] > 0:
                fitted = model.fit_regularized(
                    alpha=config["ridge_alpha"], L1_wt=0.0, maxiter=200
                )
            else:
                fitted = model.fit(maxiter=200)
        else:
            family = sm.families.NegativeBinomial(
                alpha=float(config["dispersion_alpha"])
            )
            model = sm.GLM(y_train, x_train, family=family)
            fitted = model.fit(maxiter=200)
        prediction = float(np.asarray(fitted.predict(x_test))[0])
    return max(0.0, prediction)


def tune_model(
    method: str,
    configurations: list[dict[str, Any]],
    development: pd.DataFrame,
    observed_counts: pd.Series,
    all_supervised: pd.DataFrame,
    cv_splits: int = 3,
    cv_horizon: int = 12,
    seed: int = 42,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    """Select configuration by expanding, one-step validation within the first half."""
    if len(development) < (cv_splits * cv_horizon + 24):
        raise ValueError(
            "A metade de treinamento é curta para a validação interna configurada. "
            "Reduza --cv-splits/--cv-horizon ou use uma série maior."
        )
    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=cv_splits, test_size=cv_horizon)
    x_placeholder = np.arange(len(development))
    folds = list(splitter.split(x_placeholder))
    score_rows: list[dict[str, Any]] = []

    for config_index, config in enumerate(configurations):
        fold_errors: list[float] = []
        failed = False
        for _, validation_indices in folds:
            predictions: list[float] = []
            actuals: list[float] = []
            for validation_index in validation_indices:
                target_month = development.index[validation_index]
                history = observed_counts.loc[observed_counts.index < target_month]
                target = float(development.iloc[validation_index]["target"])
                try:
                    prediction = forecast_one_step(
                        method, config, history, all_supervised, target_month,
                        seed=seed + config_index,
                    )
                except Exception as exc:  # Candidate-level failures must not stop other methods.
                    score_rows.append({
                        "modelo": method,
                        "configuracao": repr(config),
                        "fold": len(fold_errors) + 1,
                        "RMSE": np.nan,
                        "falha": f"{type(exc).__name__}: {exc}",
                    })
                    failed = True
                    break
                predictions.append(max(0.0, float(prediction)))
                actuals.append(target)
            if failed:
                break
            fold_rmse = float(np.sqrt(np.mean((np.asarray(actuals) - np.asarray(predictions)) ** 2)))
            fold_errors.append(fold_rmse)
            score_rows.append({
                "modelo": method,
                "configuracao": repr(config),
                "fold": len(fold_errors),
                "RMSE": fold_rmse,
                "falha": "",
            })
        if not failed and fold_errors:
            score_rows.append({
                "modelo": method,
                "configuracao": repr(config),
                "fold": "media",
                "RMSE": float(np.mean(fold_errors)),
                "falha": "",
            })

    score_frame = pd.DataFrame(score_rows)
    means = score_frame.loc[score_frame["fold"].eq("media")].copy()
    if means.empty:
        raise RuntimeError(f"Nenhuma configuração pôde ser validada para {method}.")
    best_index = means["RMSE"].idxmin()
    best_config_repr = means.loc[best_index, "configuracao"]
    best_config = next(config for config in configurations if repr(config) == best_config_repr)
    best_score = float(means.loc[best_index, "RMSE"])
    return best_config, best_score, score_frame


def model_feature_frame(counts: pd.Series) -> pd.DataFrame:
    """Public helper retained for CLI and notebooks."""
    return make_supervised(counts)
