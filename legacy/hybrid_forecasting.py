"""Leak-aware comparisons of generic methods for monthly malaria forecasting.

GA, PSO, and Bat Algorithm optimize the same RF/SVR/PCA-ANN ensemble. They are
compared with standalone model families and temporal-naive baselines using
expanding-window, rolling-origin evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("once", category=ConvergenceWarning)


@dataclass
class ForecastResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_features: dict[str, list[str]]
    convergence: pd.DataFrame
    parameters: dict[str, dict]
    elapsed_seconds: float
    nonconverged_mlp_fits: int


@dataclass
class ForecastEvaluation:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    run_metrics: pd.DataFrame
    run_predictions: pd.DataFrame
    paired_comparisons: pd.DataFrame
    selections: pd.DataFrame
    convergence: pd.DataFrame
    outer_folds: int
    repetitions: int
    nonconverged_mlp_fits: int


def readable_results_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return model scores with user-facing Portuguese labels and rounded values."""
    labels = {
        "GA": "Ensemble otimizado por GA",
        "PSO": "Ensemble otimizado por PSO",
        "BAT": "Ensemble otimizado por Bat Algorithm",
        "ANN_media": "Rede neural média (3 redes)",
        "P2CA_ANN": "Rede neural PCA (1 rede)",
        "SVR": "SVR",
        "RF": "Random Forest",
        "naive_ultimo_mes": "Baseline: último mês",
        "media_movel_3m": "Baseline: média de 3 meses",
        "naive_sazonal_12m": "Baseline: mesmo mês do ano anterior",
    }
    table = metrics.copy()
    table["modelo"] = table["modelo"].map(labels).fillna(table["modelo"])
    table = table.rename(columns={
        "modelo": "Modelo",
        "MAE": "MAE médio (casos)",
        "RMSE": "RMSE médio (casos)",
        "sMAPE_pct": "sMAPE médio (%)",
        "MAPE_pct_sem_zeros": "MAPE médio (%)¹",
        "Bias_casos": "Viés médio (casos)²",
        "MASE": "MASE médio³",
        "Correlacao": "Correlação média⁴",
        "MAE_dp": "DP do MAE",
        "RMSE_dp": "DP do RMSE",
        "sMAPE_pct_dp": "DP do sMAPE",
        "MAPE_pct_sem_zeros_dp": "DP do MAPE",
        "Bias_casos_dp": "DP do viés",
        "MASE_dp": "DP do MASE",
        "Correlacao_dp": "DP da correlação",
    })
    sort_column = "RMSE médio (casos)" if "RMSE médio (casos)" in table else "RMSE"
    return table.sort_values(sort_column).round(3).reset_index(drop=True)


def plot_forecast_diagnostics(
    result: ForecastEvaluation,
    save_path: str | None = None,
    show: bool = True,
    observed_series: pd.Series | None = None,
):
    """Plot generic model forecasts and outer-fold error summaries."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    display_names = {
        "GA": "Ensemble otimizado por GA",
        "PSO": "Ensemble otimizado por PSO",
        "BAT": "Ensemble otimizado por BAT",
        "RF": "Random Forest",
        "SVR": "SVR",
        "P2CA_ANN": "Rede neural PCA",
        "ANN_media": "Rede neural média",
        "naive_ultimo_mes": "Baseline: último mês",
        "media_movel_3m": "Baseline: média 3 meses",
        "naive_sazonal_12m": "Baseline sazonal de 12 meses",
    }
    predictions = result.predictions

    if observed_series is not None:
        observed_series = observed_series.sort_index()
        axes[0, 0].plot(
            observed_series.index, observed_series.values,
            color="#9aa0a6", alpha=0.48, linewidth=1.0,
            label="Histórico observado", zorder=1,
        )
    axes[0, 0].plot(
        predictions.index, predictions["real"],
        color="#202124", linewidth=2.1, label="Real avaliado", zorder=5,
    )
    for column in ["GA", "PSO", "BAT", "RF"]:
        if column in predictions:
            axes[0, 0].plot(
                predictions.index, predictions[column],
                label=display_names[column], linewidth=1.4, alpha=0.9,
            )
    axes[0, 0].set_title("Previsões de um mês à frente")
    axes[0, 0].set_ylabel("Casos notificados")
    axes[0, 0].legend(fontsize=7, ncol=2)

    if not result.convergence.empty:
        for algorithm, trace in result.convergence.groupby("algorithm", sort=False):
            mean_trace = trace.groupby("iteration", as_index=False)["cv_RMSE"].mean()
            axes[0, 1].plot(
                mean_trace["iteration"], mean_trace["cv_RMSE"],
                marker="o", ms=3, label=algorithm,
            )
    axes[0, 1].set_title("RMSE interno médio por iteração")
    axes[0, 1].set_xlabel("Iteração")
    axes[0, 1].set_ylabel("RMSE de validação")
    axes[0, 1].legend()

    rmse_winner = result.metrics.loc[result.metrics["RMSE"].idxmin()]
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05, 0.92,
        "Resumo exploratório\n\n"
        f"Menor RMSE médio: {display_names.get(rmse_winner['modelo'], rmse_winner['modelo'])} "
        f"({rmse_winner['RMSE']:.1f} casos)\n\n"
        f"{result.outer_folds} blocos temporais × {result.repetitions} sementes\n"
        f"{len(predictions)} meses avaliados\n"
        f"{predictions.index.min():%m/%Y} a {predictions.index.max():%m/%Y}\n\n"
        "Ranking pontual; consulte os intervalos pareados antes de afirmar superioridade.",
        va="top", fontsize=12, transform=axes[0, 2].transAxes,
    )

    metric_order = result.metrics.sort_values("RMSE")
    axes[1, 0].barh(
        [display_names.get(name, name) for name in metric_order["modelo"]],
        metric_order["RMSE"], color="#177e73",
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("RMSE médio por método")
    axes[1, 0].set_xlabel("RMSE (menor é melhor)")

    if not result.paired_comparisons.empty:
        comparisons = result.paired_comparisons.sort_values("diferenca_MAE_casos")
        center = comparisons["diferenca_MAE_casos"].to_numpy(dtype=float)
        lower_error = np.maximum(
            0.0, center - comparisons["IC95_inf"].to_numpy(dtype=float)
        )
        upper_error = np.maximum(
            0.0, comparisons["IC95_sup"].to_numpy(dtype=float) - center
        )
        axes[1, 1].errorbar(
            center,
            range(len(comparisons)),
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o", color="#0077b6", capsize=3,
        )
        axes[1, 1].axvline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 1].set_yticks(range(len(comparisons)))
        axes[1, 1].set_yticklabels(comparisons["modelo"])
        axes[1, 1].set_title("Diferença pareada de MAE vs. último mês")
        axes[1, 1].set_xlabel("Método − baseline (casos; negativo favorece método)")
    else:
        axes[1, 1].axis("off")

    for column in ["GA", "PSO", "BAT", "RF", "naive_ultimo_mes"]:
        if column not in predictions:
            continue
        residual = predictions[column] - predictions["real"]
        axes[1, 2].plot(
            residual.index, residual, marker="o", ms=2.5, linewidth=0.9,
            label=display_names[column],
        )
    axes[1, 2].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1, 2].set_title("Resíduo mensal (previsto − real)")
    axes[1, 2].set_ylabel("Erro em casos")
    axes[1, 2].tick_params(axis="x", rotation=30)
    axes[1, 2].legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_forecast_overlay(
    predictions: pd.DataFrame,
    observed_series: pd.Series,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot historical counts and predictions from generic model families."""
    fig, ax = plt.subplots(figsize=(16, 7))
    observed_series = observed_series.sort_index()
    ax.plot(
        observed_series.index, observed_series.values,
        color="#9aa0a6", alpha=0.58, linewidth=1.25,
        label="Histórico real", zorder=1,
    )
    ax.plot(
        predictions.index, predictions["real"],
        color="#202124", linewidth=2.7, marker="o", markersize=3.5,
        label="Real — avaliação", zorder=6,
    )
    display_names = {
        "GA": "Ensemble otimizado por GA",
        "PSO": "Ensemble otimizado por PSO",
        "BAT": "Ensemble otimizado por BAT",
        "RF": "Random Forest",
        "SVR": "SVR",
        "P2CA_ANN": "Rede neural PCA",
        "ANN_media": "Rede neural média",
        "naive_ultimo_mes": "Baseline: último mês",
        "media_movel_3m": "Baseline: média de 3 meses",
        "naive_sazonal_12m": "Baseline sazonal de 12 meses",
    }
    for column, label in display_names.items():
        if column not in predictions:
            continue
        is_baseline = column.startswith(("naive_", "media_movel"))
        ax.plot(
            predictions.index, predictions[column], linewidth=1.65, alpha=0.9,
            linestyle="--" if is_baseline else "-", label=label, zorder=3,
        )
    ax.set_title("Casos notificados e previsões — Breves-PA")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Casos notificados")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=3, fontsize=9, framealpha=0.95)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_forecast_comparison_panels(
    predictions: pd.DataFrame,
    observed_series: pd.Series,
    output_prefix: str | Path,
    show: bool = False,
) -> list[Path]:
    """Save panels comparing generic optimizers, models, and temporal baselines."""
    display_names = {
        "GA": "Ensemble otimizado por GA",
        "PSO": "Ensemble otimizado por PSO",
        "BAT": "Ensemble otimizado por BAT",
        "RF": "Random Forest",
        "SVR": "SVR",
        "P2CA_ANN": "Rede neural PCA",
        "ANN_media": "Rede neural média (3 redes)",
        "naive_ultimo_mes": "Baseline: último mês",
        "media_movel_3m": "Baseline: média de 3 meses",
        "naive_sazonal_12m": "Baseline sazonal de 12 meses",
    }
    groups = [
        [name for name in ["GA", "PSO", "BAT", "RF"] if name in predictions],
        [name for name in ["SVR", "P2CA_ANN", "ANN_media", "naive_ultimo_mes"] if name in predictions],
        [name for name in ["media_movel_3m", "naive_sazonal_12m"] if name in predictions],
    ]
    colors = {
        "GA": "#d1495b", "PSO": "#0077b6", "BAT": "#2a9d8f",
        "RF": "#7251b5", "SVR": "#e76f00", "P2CA_ANN": "#2a9d8f",
        "ANN_media": "#d1495b", "naive_ultimo_mes": "#0077b6",
        "media_movel_3m": "#e76f00", "naive_sazonal_12m": "#8e6cba",
    }
    observed_series = observed_series.sort_index().astype(float)
    predictions = predictions.sort_index()
    output_prefix = Path(output_prefix)
    output_paths = []
    group_number = 0

    for model_group in groups:
        if not model_group:
            continue
        group_number += 1
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, model_group):
            ax.plot(
                observed_series.index, observed_series.values,
                color="#aeb4ba", linewidth=1.15, alpha=0.55,
                label="Histórico real", zorder=1,
            )
            ax.plot(
                predictions.index, predictions["real"],
                color="#202124", linewidth=2.2, marker="o", markersize=3.1,
                label="Real avaliado", zorder=4,
            )
            is_baseline = model.startswith(("naive_", "media_movel"))
            ax.plot(
                predictions.index, predictions[model], color=colors[model],
                linewidth=2.0, marker="o", markersize=2.8,
                linestyle="--" if is_baseline else "-",
                label="Previsão média entre sementes", zorder=5,
            )
            mae = mean_absolute_error(predictions["real"], predictions[model])
            rmse = mean_squared_error(predictions["real"], predictions[model]) ** 0.5
            ax.set_title(f"{display_names[model]} | MAE {mae:.1f} · RMSE {rmse:.1f}")
            ax.grid(True, alpha=0.25)
            ax.set_ylabel("Casos notificados")
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax.tick_params(axis="x", rotation=25)
        for ax in axes.flat[len(model_group):]:
            ax.axis("off")
        fig.suptitle(
            "Previsões temporais por método — Breves-PA\n"
            f"Avaliação em {predictions.index.min():%m/%Y} a {predictions.index.max():%m/%Y}",
            fontsize=15, y=1.02,
        )
        fig.tight_layout()
        path = output_prefix.with_name(f"{output_prefix.stem}_{group_number}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        output_paths.append(path)
        if show:
            plt.show()
        plt.close(fig)
    return output_paths


def plot_optimizer_comparison(
    metrics: pd.DataFrame,
    save_path: str | None = None,
    show: bool = True,
):
    """Compare standard GA, PSO, and BAT optimizer scores."""
    names = {"GA": "GA", "PSO": "PSO", "BAT": "BAT"}
    rows = metrics.loc[metrics["modelo"].isin(names)].copy()
    if rows.empty:
        raise ValueError("A tabela não contém resultados dos otimizadores genéricos.")
    metric_specs = [("MAE", "MAE médio (casos)"), ("RMSE", "RMSE médio (casos)"),
                    ("sMAPE_pct", "sMAPE médio (%)")]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(14, 4.8))
    for ax, (column, title) in zip(axes, metric_specs):
        ranked = rows.dropna(subset=[column]).sort_values(column)
        if ranked.empty:
            ax.axis("off")
            continue
        best = ranked.iloc[0]
        values = ranked[column].to_numpy(dtype=float)
        colors = ["#e9a23b" if name == best["modelo"] else "#2a9d8f"
                  for name in ranked["modelo"]]
        std_column = {
            "MAE": "MAE_dp", "RMSE": "RMSE_dp", "sMAPE_pct": "sMAPE_pct_dp"
        }[column]
        errors = (
            ranked[std_column].fillna(0).to_numpy(dtype=float)
            if std_column in ranked else None
        )
        bars = ax.barh(
            [names[name] for name in ranked["modelo"]], values,
            xerr=errors, capsize=3, color=colors,
        )
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 1)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.025, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}", va="center", fontsize=9)
    rmse_winner = rows.loc[rows["RMSE"].idxmin()]
    fig.suptitle(
        "Otimizadores genéricos — médias da validação temporal\n"
        f"Menor RMSE médio pontual: {rmse_winner['modelo']} "
        f"({rmse_winner['RMSE']:.1f} casos); avalie o IC pareado antes de concluir superioridade.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def add_demographic_features(features: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Add monthly species/sex/age counts observable at forecast origin t."""
    enriched = features.copy()
    groups = {
        "especie": "especie",
        "sexo": "sexo",
        "faixa_etaria": "faixa_etaria",
    }
    for source, prefix in groups.items():
        values = records[source].fillna("Não informado").astype(str).str.strip()
        values = values.replace("", "Não informado")
        temp = records[["mes", "peso"]].copy()
        temp["categoria"] = values.to_numpy()
        monthly = temp.pivot_table(
            index="mes", columns="categoria", values="peso", aggfunc="sum", fill_value=0
        )
        monthly.columns = [f"{prefix}_{str(value)[:40]}" for value in monthly.columns]
        monthly = monthly.reindex(enriched.index, fill_value=0)
        # Small, fixed lag set keeps the expanded search space manageable.
        for lag in (0, 1, 3, 12):
            lagged = monthly.shift(lag).fillna(0)
            lagged.columns = [f"{column}_lag_{lag}" for column in monthly.columns]
            enriched = enriched.join(lagged)
    # Keep warm-up NaNs so the notebook drops rows without enough history.
    return enriched.replace([np.inf, -np.inf], np.nan)


def _halton_like_population(size: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    """Low-discrepancy Kronecker sequence inspired by Good Point Set starts."""
    primes = []
    candidate = 2
    while len(primes) < dimensions:
        if all(candidate % p for p in range(2, int(candidate**0.5) + 1)):
            primes.append(candidate)
        candidate += 1
    offsets = rng.random(dimensions)
    indices = np.arange(1, size + 1, dtype=float)[:, None]
    return np.mod(indices * np.sqrt(np.asarray(primes))[None, :] + offsets, 1.0)


def _make_pca_ann(params: dict, seed: int):
    return make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95, svd_solver="full"),
            MLPRegressor(
                hidden_layer_sizes=(params["hidden"],),
                alpha=params["alpha"],
                learning_rate_init=0.01,
                max_iter=200,
                early_stopping=False,
                random_state=seed,
            ),
        )


def _fit_model(model, x, y):
    """Fit an estimator and count MLP convergence warnings without flooding logs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    warning_count += int(getattr(model, "nonconverged_fits", 0))
    return model, warning_count


class AverageNeuralNetworkRegressor:
    """Average predictions from several PCA+MLP networks with distinct seeds."""

    def __init__(self, params: dict, seed: int, n_networks: int = 3):
        self.params = params
        self.seed = seed
        self.n_networks = n_networks
        self.models = []
        self.nonconverged_fits = 0

    def fit(self, x, y):
        self.models = []
        self.nonconverged_fits = 0
        for index in range(self.n_networks):
            model, warning_count = _fit_model(
                _make_pca_ann(self.params, self.seed + 101 * index), x, y
            )
            self.models.append(model)
            self.nonconverged_fits += warning_count
        return self

    def predict(self, x):
        return np.mean([model.predict(x) for model in self.models], axis=0)


def _make_models(params: dict, seed: int):
    """Return the shared PCA-ANN, SVR, and RF ensemble components."""
    return (
        _make_pca_ann(params, seed),
        make_pipeline(
            StandardScaler(),
            SVR(C=params["svr_c"], gamma=params["svr_gamma"], epsilon=0.1),
        ),
        RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=seed,
            n_jobs=-1,
        ),
    )


def _metrics(y_true, y_pred, mase_scale: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    nonzero = y_true != 0
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "sMAPE_pct": (
            np.mean(2 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100
            if valid.any() else np.nan
        ),
        "MAPE_pct_sem_zeros": (
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
            if nonzero.any() else np.nan
        ),
        # Positive bias means systematic overprediction; negative means underprediction.
        "Bias_casos": float(np.mean(y_pred - y_true)),
        "MASE": (
            mean_absolute_error(y_true, y_pred) / mase_scale
            if mase_scale is not None and mase_scale > 0 else np.nan
        ),
        # Correlation is descriptive only; it does not measure forecast error.
        "Correlacao": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0
            else np.nan
        ),
    }


def _optimize_candidate(
    algorithm: str,
    initial_population: np.ndarray,
    evaluate,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Optimize the shared validation objective with GA, PSO, or Bat search."""
    rng = np.random.default_rng(seed)
    population = np.clip(initial_population.copy(), 0.0, 1.0)
    population_size, dimensions = population.shape
    objective_requests = 0

    def score(candidate):
        nonlocal objective_requests
        objective_requests += 1
        return evaluate(candidate)

    scores = np.array([score(candidate)[0] for candidate in population])
    best_index = int(np.argmin(scores))
    best_position = population[best_index].copy()
    best_score = float(scores[best_index])
    trace = [{
        "algorithm": algorithm,
        "iteration": 0,
        "fitness": best_score,
        "cv_RMSE": score(best_position)[1],
        "objective_requests": objective_requests,
    }]

    def record_iteration(iteration: int) -> None:
        nonlocal best_position, best_score
        best_index = int(np.argmin(scores))
        if scores[best_index] < best_score:
            best_score = float(scores[best_index])
            best_position = population[best_index].copy()
        trace.append({
            "algorithm": algorithm,
            "iteration": iteration,
            "fitness": best_score,
            "cv_RMSE": score(best_position)[1],
            "objective_requests": objective_requests,
        })

    if algorithm == "GA":
        mutation_probability = 1.0 / max(dimensions, 1)
        for iteration in range(1, iterations + 1):
            ranked = np.argsort(scores)
            next_population = [population[ranked[0]].copy()]
            if population_size > 1:
                next_population.append(population[ranked[1]].copy())

            def tournament():
                contestants = rng.choice(population_size, size=min(3, population_size), replace=False)
                winner = contestants[np.argmin(scores[contestants])]
                return population[winner]

            while len(next_population) < population_size:
                parent_a, parent_b = tournament(), tournament()
                mix = rng.random(dimensions)
                child = mix * parent_a + (1.0 - mix) * parent_b
                mutate = rng.random(dimensions) < mutation_probability
                child[mutate] += rng.normal(0.0, 0.15, int(mutate.sum()))
                next_population.append(np.clip(child, 0.0, 1.0))
            population = np.asarray(next_population)
            scores = np.array([score(candidate)[0] for candidate in population])
            record_iteration(iteration)

    elif algorithm == "PSO":
        velocity = rng.normal(0.0, 0.08, size=population.shape)
        personal_best = population.copy()
        personal_scores = scores.copy()
        for iteration in range(1, iterations + 1):
            velocity = (
                0.72 * velocity
                + 1.45 * rng.random(population.shape) * (personal_best - population)
                + 1.45 * rng.random(population.shape) * (best_position - population)
            )
            population = np.clip(population + velocity, 0.0, 1.0)
            scores = np.array([score(candidate)[0] for candidate in population])
            improved = scores < personal_scores
            personal_best[improved] = population[improved]
            personal_scores[improved] = scores[improved]
            record_iteration(iteration)

    elif algorithm == "BAT":
        velocity = np.zeros_like(population)
        loudness = np.full(population_size, 0.9)
        pulse_rate = np.full(population_size, 0.25)
        frequencies = rng.uniform(0.0, 2.0, population_size)
        for iteration in range(1, iterations + 1):
            for index in range(population_size):
                frequencies[index] = rng.uniform(0.0, 2.0)
                velocity[index] += (population[index] - best_position) * frequencies[index]
                candidate = np.clip(population[index] + velocity[index], 0.0, 1.0)
                if rng.random() > pulse_rate[index]:
                    candidate = np.clip(
                        best_position + rng.normal(0.0, 0.01 + 0.1 * loudness[index], dimensions),
                        0.0, 1.0,
                    )
                candidate_score = score(candidate)[0]
                if candidate_score < scores[index] and rng.random() < loudness[index]:
                    population[index] = candidate
                    scores[index] = candidate_score
                    loudness[index] *= 0.92
                    pulse_rate[index] = min(0.95, pulse_rate[index] * 1.08)
            record_iteration(iteration)
    else:
        raise ValueError(f"Algoritmo de otimização não reconhecido: {algorithm}")

    return best_position, trace


def run_forecast_comparison(
    model_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_t+1",
    test_months: int = 12,
    cv_splits: int = 3,
    cv_test_months: int = 12,
    population_size: int = 4,
    iterations: int = 3,
    seed: int = 42,
    subset_penalty: float = 0.02,
) -> ForecastResult:
    """Evaluate generic optimizers, standalone estimators, and naive baselines.

    Hyperparameters and feature masks are selected only on development rows.
    The final ``test_months`` are forecast one month at a time, refitting each
    estimator with the observations available at that forecast origin.
    """
    started = perf_counter()
    data = model_data.dropna(subset=feature_columns + [target_column]).sort_index().copy()
    minimum_development = cv_splits * cv_test_months + 24
    if test_months < 1 or len(data) < test_months + minimum_development:
        raise ValueError(
            "Amostras insuficientes: são necessários pelo menos "
            f"{minimum_development} meses de desenvolvimento e {test_months} de teste."
        )
    if not feature_columns:
        raise ValueError("A lista de atributos está vazia.")
    development = data.iloc[:-test_months]
    test = data.iloc[-test_months:]
    x_all = development[feature_columns].to_numpy(dtype=float)
    y_all = development[target_column].to_numpy(dtype=float)
    folds = list(TimeSeriesSplit(
        n_splits=cv_splits, test_size=cv_test_months
    ).split(x_all))
    n_features = len(feature_columns)
    dimensions = n_features + 4  # feature mask + SVR C/gamma + ANN alpha/width
    rng = np.random.default_rng(seed)

    def decode(position):
        mask = position[:n_features] >= 0.5
        if not mask.any():
            mask[np.argmax(position[:n_features])] = True
        return mask, {
            "svr_c": float(10 ** (-1 + 3 * position[n_features])),
            "svr_gamma": float(10 ** (-3 + 3 * position[n_features + 1])),
            "alpha": float(10 ** (-5 + 4 * position[n_features + 2])),
            "hidden": int(np.clip(round(4 + 28 * position[n_features + 3]), 4, 32)),
        }

    def make_evaluator(cache):
        fit_counter = {"nonconverged_mlp_fits": 0}

        def evaluate(position):
            key = tuple(np.round(position, 4))
            if key in cache:
                return cache[key]
            mask, params = decode(position)
            fold_rmse = []
            for fold_id, (train_idx, valid_idx) in enumerate(folds):
                x_train = x_all[train_idx][:, mask]
                x_valid = x_all[valid_idx][:, mask]
                y_train, y_valid = y_all[train_idx], y_all[valid_idx]
                models = _make_models(params, seed + fold_id)
                branch_predictions = []
                for model in models:
                    model, warning_count = _fit_model(model, x_train, y_train)
                    fit_counter["nonconverged_mlp_fits"] += warning_count
                    branch_predictions.append(np.maximum(model.predict(x_valid), 0.0))
                prediction = np.mean(branch_predictions, axis=0)
                fold_rmse.append(mean_squared_error(y_valid, prediction) ** 0.5)
            cv_rmse = float(np.mean(fold_rmse))
            objective = cv_rmse + subset_penalty * np.mean(y_all) * mask.mean()
            cache[key] = (float(objective), cv_rmse)
            return cache[key]

        return evaluate, cache, fit_counter

    shared_initial = _halton_like_population(
        population_size, dimensions, rng
    )
    best_positions, best_trace = {}, []
    nonconverged_mlp_fits = 0
    shared_candidate_cache = {}
    for offset, algorithm in enumerate(("GA", "PSO", "BAT"), start=1):
        evaluate, cache, fit_counter = make_evaluator(shared_candidate_cache)
        position, trace = _optimize_candidate(
            algorithm, shared_initial, evaluate, iterations, seed + 20_000 + offset
        )
        best_positions[algorithm] = position
        nonconverged_mlp_fits += fit_counter["nonconverged_mlp_fits"]
        for row in trace:
            row["cumulative_unique_objective_evaluations"] = len(cache)
            row["mlp_nonconverged_fits"] = fit_counter["nonconverged_mlp_fits"]
        best_trace.extend(trace)

    def forecast_candidate(position: np.ndarray):
        nonlocal nonconverged_mlp_fits
        candidate_mask, candidate_params = decode(position)
        candidate_features = [
            name for name, keep in zip(feature_columns, candidate_mask) if keep
        ]
        branch_values = np.zeros((3, len(test)), dtype=float)
        for test_position, (origin, test_row) in enumerate(test.iterrows()):
            history = data.loc[data.index < origin]
            x_history_all = history[candidate_features].to_numpy(dtype=float)
            y_history = history[target_column].to_numpy(dtype=float)
            x_one = test_row[candidate_features].to_numpy(dtype=float).reshape(1, -1)
            models = _make_models(candidate_params, seed + test_position)
            for model_index, model in enumerate(models):
                model, warning_count = _fit_model(model, x_history_all, y_history)
                nonconverged_mlp_fits += warning_count
                branch_values[model_index, test_position] = max(
                    float(model.predict(x_one)[0]), 0.0
                )
        return candidate_params, candidate_features, branch_values

    candidate_forecasts = {
        name: forecast_candidate(position)
        for name, position in best_positions.items()
    }
    y_test = test[target_column].to_numpy(dtype=float)
    target_months = test.index + pd.DateOffset(months=1)
    prediction_columns = {"real": y_test}
    selected_features, parameter_sets = {}, {}
    for algorithm, (params, chosen, branch_predictions) in candidate_forecasts.items():
        prediction_columns[algorithm] = np.mean(branch_predictions, axis=0)
        selected_features[algorithm] = chosen
        parameter_sets[algorithm] = params

    # Standalone estimators use a prespecified reference configuration and all
    # available features; optimized ensemble settings are not reused for them.
    reference_params = {"svr_c": 10.0, "svr_gamma": 0.01, "alpha": 0.001, "hidden": 16}
    standalone_names = ("P2CA_ANN", "SVR", "RF")
    for model_name, model_index in zip(standalone_names, range(3)):
        values = np.zeros(len(test), dtype=float)
        for test_position, (origin, test_row) in enumerate(test.iterrows()):
            history = data.loc[data.index < origin]
            x_history = history[feature_columns].to_numpy(dtype=float)
            y_history = history[target_column].to_numpy(dtype=float)
            x_one = test_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)
            model = _make_models(reference_params, seed + test_position)[model_index]
            model, warning_count = _fit_model(model, x_history, y_history)
            nonconverged_mlp_fits += warning_count
            values[test_position] = max(float(model.predict(x_one)[0]), 0.0)
        prediction_columns[model_name] = values
        selected_features[model_name] = list(feature_columns)
        parameter_sets[model_name] = dict(reference_params)

    ann_values = np.zeros(len(test), dtype=float)
    for test_position, (origin, test_row) in enumerate(test.iterrows()):
        history = data.loc[data.index < origin]
        x_history = history[feature_columns].to_numpy(dtype=float)
        y_history = history[target_column].to_numpy(dtype=float)
        x_one = test_row[feature_columns].to_numpy(dtype=float).reshape(1, -1)
        model, warning_count = _fit_model(
            AverageNeuralNetworkRegressor(
                reference_params, seed + test_position + 10_000, n_networks=3
            ),
            x_history,
            y_history,
        )
        nonconverged_mlp_fits += warning_count
        ann_values[test_position] = max(float(model.predict(x_one)[0]), 0.0)
    prediction_columns["ANN_media"] = ann_values
    selected_features["ANN_media"] = list(feature_columns)
    parameter_sets["ANN_media"] = dict(reference_params)

    prediction_columns.update({
        "naive_ultimo_mes": test["casos_mes_atual"].to_numpy(dtype=float),
        "media_movel_3m": test["media_3m"].to_numpy(dtype=float),
        "naive_sazonal_12m": test["casos_mes_ano_anterior"].to_numpy(dtype=float),
    })
    predictions = pd.DataFrame(prediction_columns, index=target_months)
    development_targets = development[target_column].to_numpy(dtype=float)
    seasonal_errors = np.abs(development_targets[12:] - development_targets[:-12])
    if not len(seasonal_errors):
        seasonal_errors = np.abs(np.diff(development_targets))
    mase_scale = float(np.mean(seasonal_errors)) if len(seasonal_errors) else np.nan
    metrics = pd.DataFrame([
        {"modelo": name, **_metrics(y_test, predictions[name], mase_scale)}
        for name in predictions.columns if name != "real"
    ]).sort_values("RMSE")
    result = ForecastResult(
        metrics=metrics,
        predictions=predictions,
        selected_features=selected_features,
        convergence=pd.DataFrame(best_trace),
        parameters=parameter_sets,
        elapsed_seconds=perf_counter() - started,
        nonconverged_mlp_fits=nonconverged_mlp_fits,
    )
    return result


def _block_bootstrap_mean_interval(
    values: np.ndarray,
    block_length: int = 12,
    repetitions: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Circular moving-block bootstrap interval for an average paired loss."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan
    rng = np.random.default_rng(seed)
    block_length = min(max(1, block_length), len(values))
    block_count = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        starts = rng.integers(0, len(values), size=block_count)
        sample_idx = np.concatenate([(start + offsets) % len(values) for start in starts])
        draws[index] = np.mean(values[sample_idx[:len(values)]])
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def run_rolling_origin_experiment(
    model_data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_t+1",
    initial_train_months: int = 60,
    outer_horizon: int = 12,
    cv_splits: int = 3,
    cv_test_months: int = 12,
    population_size: int = 4,
    iterations: int = 3,
    repetitions: int = 3,
    seed: int = 42,
) -> ForecastEvaluation:
    """Run nested rolling-origin evaluation with expanding training windows.

    Each outer fold starts after ``initial_train_months`` development rows.
    Search is repeated independently for every outer fold and random seed. Within
    the outer block, each forecast is one month ahead and the model is refitted
    when the next observed month becomes available.
    """
    data = model_data.dropna(subset=feature_columns + [target_column]).sort_index().copy()
    if initial_train_months < cv_splits * cv_test_months + 24:
        raise ValueError(
            "O treinamento inicial deve conter ao menos 24 meses antes das dobras internas."
        )
    if outer_horizon < 1 or repetitions < 1:
        raise ValueError("Horizonte e número de repetições devem ser positivos.")
    if len(data) <= initial_train_months:
        raise ValueError(
            f"Há {len(data)} meses utilizáveis; são necessários mais de "
            f"{initial_train_months} para iniciar a avaliação externa."
        )

    run_metric_rows, prediction_frames, selection_rows, convergence_frames = [], [], [], []
    fold_number = 0
    nonconverged_mlp_fits = 0
    start = initial_train_months
    while start < len(data):
        fold_number += 1
        end = min(start + outer_horizon, len(data))
        fold_data = data.iloc[:end]
        fold_horizon = end - start
        for repetition in range(repetitions):
            run_seed = seed + (fold_number - 1) * 10_000 + repetition * 1_000
            result = run_forecast_comparison(
                model_data=fold_data,
                feature_columns=feature_columns,
                target_column=target_column,
                test_months=fold_horizon,
                cv_splits=cv_splits,
                cv_test_months=cv_test_months,
                population_size=population_size,
                iterations=iterations,
                seed=run_seed,
            )
            nonconverged_mlp_fits += result.nonconverged_mlp_fits
            for _, row in result.metrics.iterrows():
                run_metric_rows.append({
                    "fold": fold_number,
                    "repetition": repetition + 1,
                    "seed": run_seed,
                    "meses_teste": len(result.predictions),
                    "elapsed_seconds": result.elapsed_seconds,
                    **row.to_dict(),
                })
            for model_name, features in result.selected_features.items():
                selection_rows.append({
                    "fold": fold_number,
                    "repetition": repetition + 1,
                    "seed": run_seed,
                    "modelo": model_name,
                    "numero_atributos": len(features),
                    "atributos": json.dumps(features, ensure_ascii=False),
                    "parametros": json.dumps(
                        result.parameters.get(model_name, {}), ensure_ascii=False,
                        sort_keys=True,
                    ),
                })
            frame = result.predictions.copy()
            frame.index.name = "mes"
            frame = frame.reset_index()
            frame["fold"] = fold_number
            frame["repetition"] = repetition + 1
            frame["seed"] = run_seed
            prediction_frames.append(frame)
            trace = result.convergence.copy()
            if not trace.empty:
                trace["fold"] = fold_number
                trace["repetition"] = repetition + 1
                trace["seed"] = run_seed
                convergence_frames.append(trace)
        start = end

    run_metrics = pd.DataFrame(run_metric_rows)
    selections = pd.DataFrame(selection_rows)
    run_predictions = pd.concat(prediction_frames, ignore_index=True)
    convergence = (
        pd.concat(convergence_frames, ignore_index=True)
        if convergence_frames else pd.DataFrame()
    )
    metric_columns = [
        "MAE", "RMSE", "sMAPE_pct", "MAPE_pct_sem_zeros",
        "Bias_casos", "MASE", "Correlacao",
    ]
    summary_rows = []
    for model_name, group in run_metrics.groupby("modelo", sort=False):
        row = {"modelo": model_name, "runs": len(group)}
        weights = group["meses_teste"].to_numpy(dtype=float)
        for column in metric_columns:
            valid = group[column].notna().to_numpy()
            if valid.any():
                values = group.loc[valid, column].to_numpy(dtype=float)
                valid_weights = weights[valid]
                row[column] = (
                    float(np.sqrt(np.average(values ** 2, weights=valid_weights)))
                    if column == "RMSE"
                    else float(np.average(values, weights=valid_weights))
                )
            else:
                row[column] = np.nan
            row[f"{column}_dp"] = group[column].std(ddof=1) if len(group) > 1 else np.nan
        summary_rows.append(row)
    metrics = pd.DataFrame(summary_rows).sort_values("RMSE").reset_index(drop=True)

    prediction_columns = [
        column for column in run_predictions.columns
        if column not in {"mes", "fold", "repetition", "seed"}
    ]
    predictions = run_predictions.groupby("mes", as_index=True)[prediction_columns].mean()
    predictions.index = pd.DatetimeIndex(predictions.index, name="mes")
    paired_rows = []
    reference = "naive_ultimo_mes"
    if reference in run_predictions:
        for model_name in [column for column in predictions if column not in {"real", reference}]:
            paired_by_month = run_predictions.assign(
                paired_difference=(
                    np.abs(run_predictions[model_name] - run_predictions["real"])
                    - np.abs(run_predictions[reference] - run_predictions["real"])
                )
            ).groupby("mes")["paired_difference"].mean()
            paired_loss = paired_by_month.to_numpy(dtype=float)
            mean, low, high = _block_bootstrap_mean_interval(
                paired_loss, block_length=12, seed=seed + len(paired_rows)
            )
            paired_rows.append({
                "modelo": model_name,
                "referencia": reference,
                "diferenca_MAE_casos": mean,
                "IC95_inf": low,
                "IC95_sup": high,
                "meses": int(np.isfinite(paired_loss).sum()),
            })
    return ForecastEvaluation(
        metrics=metrics,
        predictions=predictions,
        run_metrics=run_metrics,
        run_predictions=run_predictions,
        paired_comparisons=pd.DataFrame(paired_rows),
        selections=selections,
        convergence=convergence,
        outer_folds=fold_number,
        repetitions=repetitions,
        nonconverged_mlp_fits=nonconverged_mlp_fits,
    )


def _find_csv(explicit_path: str | None = None) -> Path:
    filename = "6- Autóctones Mun Fx Etária e Sexo_Dados completos_data.csv"
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV não encontrado: {path}")
        return path
    project_dir = Path(__file__).resolve().parent
    candidates = [
        project_dir / filename,
        project_dir.parent / filename,
        Path.home() / "Downloads" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "CSV não encontrado. Informe o caminho com --data ou coloque o arquivo em "
        f"{project_dir}. Nome esperado: {filename}"
    )


def _load_breves_records(csv_path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Read just the required CSV columns and retain Breves-PA rows."""
    required = {
        "BR Mun de infecção": "municipio_infeccao",
        "BR UF Infecção": "uf_infeccao",
        "BR Data de Notificação": "data_notificacao",
        "Número de registros": "numero_registros",
        "BR Espécie Parasitária": "especie",
        "BR Sexo": "sexo",
        "BR Faixa etária": "faixa_etaria",
    }
    header = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig", nrows=0)
    normalized = {
        str(column).replace("\ufeff", "").strip(): column
        for column in header.columns
    }
    missing = sorted(set(required) - set(normalized))
    if missing:
        raise ValueError(f"Colunas esperadas não encontradas no CSV: {missing}")
    source_columns = [normalized[name] for name in required]
    rename = {normalized[name]: target for name, target in required.items()}
    selected_chunks = []
    rows_read = 0
    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        encoding="utf-8-sig",
        usecols=source_columns,
        dtype={column: "string" for column in source_columns},
        chunksize=chunksize,
    ):
        rows_read += len(chunk)
        chunk = chunk.rename(columns=rename)
        municipality = chunk["municipio_infeccao"].fillna("").str.strip().str.lower()
        state = chunk["uf_infeccao"].fillna("").str.strip().str.upper()
        selected = chunk.loc[municipality.eq("breves") & state.eq("PA")].copy()
        if not selected.empty:
            selected_chunks.append(selected)
    if not selected_chunks:
        raise ValueError("Nenhuma linha foi encontrada para Breves-PA.")
    records = pd.concat(selected_chunks, ignore_index=True)
    records["peso"] = pd.to_numeric(records["numero_registros"], errors="coerce").fillna(0)
    records["data_notificacao"] = pd.to_datetime(
        records["data_notificacao"], format="%d/%m/%Y", errors="coerce"
    )
    records = records.dropna(subset=["data_notificacao"]).copy()
    records["mes"] = records["data_notificacao"].dt.to_period("M").dt.to_timestamp()
    print(f"CSV: {csv_path}")
    print(f"Linhas lidas: {rows_read:,}; registros Breves-PA válidos: {len(records):,}")
    return records


def _build_monthly_model_data(records: pd.DataFrame):
    """Build the same t -> t+1 temporal predictors used by the notebook."""
    monthly_cases = records.groupby("mes")["peso"].sum().sort_index()
    latest_record = records["data_notificacao"].max()
    latest_month_start = latest_record.to_period("M").to_timestamp()
    latest_month_end = (latest_record.to_period("M").end_time).normalize()
    if latest_record.normalize() < latest_month_end:
        # A partial reporting month is not a valid observed target for evaluation.
        monthly_cases = monthly_cases.loc[monthly_cases.index < latest_month_start]
    month_index = pd.date_range(
        monthly_cases.index.min(), monthly_cases.index.max(), freq="MS", name="mes_origem"
    )
    monthly_cases = monthly_cases.reindex(month_index, fill_value=0).astype(float)
    features = pd.DataFrame(index=monthly_cases.index)
    features["casos_mes_atual"] = monthly_cases
    # Same calendar month in the prior year, observable at forecast origin.
    features["casos_mes_ano_anterior"] = monthly_cases.shift(11)
    for lag in (1, 2, 3, 6, 12):
        features[f"lag_{lag}"] = monthly_cases.shift(lag)
    for window in (3, 6, 12):
        features[f"media_{window}m"] = monthly_cases.rolling(
            window=window, min_periods=window
        ).mean()
    for window in (3, 6):
        features[f"desvio_{window}m"] = monthly_cases.rolling(
            window=window, min_periods=window
        ).std()
    features["tendencia_3m"] = monthly_cases - monthly_cases.shift(3)
    target_month = monthly_cases.index + pd.DateOffset(months=1)
    features["mes"] = target_month.month
    features["ano"] = target_month.year
    features["trimestre"] = target_month.quarter
    features["sin_mes"] = np.sin(2 * np.pi * features["mes"] / 12)
    features["cos_mes"] = np.cos(2 * np.pi * features["mes"] / 12)
    features["indice_temporal"] = np.arange(len(features))
    features = pd.get_dummies(
        features.assign(
            estacao_periodo=[f"T{quarter}" for quarter in target_month.quarter]
        ),
        columns=["estacao_periodo"], prefix="periodo", dtype=float,
    )
    targets = pd.DataFrame(index=monthly_cases.index)
    targets["target_t+1"] = monthly_cases.shift(-1)
    enriched = add_demographic_features(features, records)
    feature_columns = list(enriched.columns)
    model_data = enriched.join(targets).dropna(
        subset=feature_columns + ["target_t+1"]
    ).copy()
    return model_data, feature_columns


def main():
    parser = argparse.ArgumentParser(
        description="Comparação temporal de métodos genéricos de previsão para Breves-PA."
    )
    parser.add_argument("--data", help="Caminho do CSV de casos.")
    parser.add_argument(
        "--output", help="PNG do painel (padrão: avaliacao_diagnostica.png)."
    )
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument(
        "--end-year", type=int, default=2026,
        help="Último ano incluído na análise (padrão: 2026).",
    )
    parser.add_argument("--initial-train-months", type=int, default=60)
    parser.add_argument("--outer-horizon", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-test-months", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year deve ser menor ou igual a --end-year")

    csv_path = _find_csv(args.data)
    records = _load_breves_records(csv_path)
    cutoff = pd.Timestamp(year=args.end_year, month=12, day=31)
    records = records.loc[records["data_notificacao"] <= cutoff].copy()
    if records.empty:
        raise ValueError(f"Não há registros de Breves-PA até {args.end_year}.")
    observed_records = records
    latest_record = records["data_notificacao"].max()
    latest_month_start = latest_record.to_period("M").to_timestamp()
    latest_month_end = latest_record.to_period("M").end_time.normalize()
    if latest_record.normalize() < latest_month_end:
        # Keep the historical backdrop on the same complete-month basis as scoring.
        observed_records = records.loc[records["mes"] < latest_month_start]
    observed_series = observed_records.groupby("mes")["peso"].sum().sort_index()
    observed_index = pd.date_range(
        observed_series.index.min(), observed_series.index.max(), freq="MS",
        name="mes_origem",
    )
    observed_series = observed_series.reindex(observed_index, fill_value=0.0)
    model_data, feature_columns = _build_monthly_model_data(records)
    first_target_month = pd.Timestamp(year=args.start_year, month=1, day=1)
    first_origin = first_target_month - pd.DateOffset(months=1)
    model_data = model_data.loc[model_data.index >= first_origin].copy()
    output_path = Path(args.output).expanduser().resolve() if args.output else (
        Path(__file__).resolve().parent / "avaliacao_diagnostica.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_rolling_origin_experiment(
        model_data=model_data,
        feature_columns=feature_columns,
        initial_train_months=args.initial_train_months,
        outer_horizon=args.outer_horizon,
        cv_splits=args.cv_splits,
        cv_test_months=args.cv_test_months,
        population_size=args.population_size,
        iterations=args.iterations,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    plot_forecast_diagnostics(result, save_path=str(output_path), show=False,
                              observed_series=observed_series)
    print(f"Painel salvo em: {output_path}")
    comparison_prefix = output_path.with_name("comparacao_metodos")
    comparison_paths = plot_forecast_comparison_panels(
        result.predictions, observed_series, output_prefix=comparison_prefix,
    )
    for comparison_path in comparison_paths:
        print(f"Gráfico comparativo salvo em: {comparison_path}")
    optimizer_comparison_path = output_path.with_name("comparacao_otimizadores.png")
    plot_optimizer_comparison(
        result.metrics, save_path=str(optimizer_comparison_path), show=False,
    )
    print(f"Comparação dos otimizadores salva em: {optimizer_comparison_path}")
    optimizer_rows = result.metrics.loc[result.metrics["modelo"].isin(["GA", "PSO", "BAT"])]
    best_optimizer = optimizer_rows.loc[optimizer_rows["RMSE"].idxmin()]
    print(
        f"Menor RMSE médio pontual entre os otimizadores: {best_optimizer['modelo']} "
        f"({best_optimizer['RMSE']:.1f} casos). Consulte as comparações pareadas; "
        "esse ranking, isoladamente, não prova superioridade."
    )
    table = readable_results_table(result.metrics)
    table_path = output_path.with_name("resultados_modelos.csv")
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"Tabela salva em: {table_path}")
    print(table.to_string(index=False))
    raw_metrics_path = output_path.with_name("metricas_agregadas.csv")
    result.metrics.to_csv(raw_metrics_path, index=False, encoding="utf-8-sig")
    run_metrics_path = output_path.with_name("metricas_por_janela_semente.csv")
    result.run_metrics.to_csv(run_metrics_path, index=False, encoding="utf-8-sig")
    predictions_path = output_path.with_name("previsoes_mensais.csv")
    result.predictions.to_csv(predictions_path, index_label="mes", encoding="utf-8-sig")
    run_predictions_path = output_path.with_name("previsoes_por_janela_semente.csv")
    result.run_predictions.to_csv(run_predictions_path, index=False, encoding="utf-8-sig")
    paired_path = output_path.with_name("comparacoes_pareadas.csv")
    result.paired_comparisons.to_csv(paired_path, index=False, encoding="utf-8-sig")
    convergence_path = output_path.with_name("convergencia_otimizadores.csv")
    result.convergence.to_csv(convergence_path, index=False, encoding="utf-8-sig")
    selections_path = output_path.with_name("selecao_atributos_parametros.csv")
    result.selections.to_csv(selections_path, index=False, encoding="utf-8-sig")
    print(
        f"Resultados gravados. Avaliação: {result.outer_folds} blocos externos × "
        f"{result.repetitions} repetições; previsões de "
        f"{result.predictions.index.min():%m/%Y} a {result.predictions.index.max():%m/%Y}; "
        f"ajustes MLP sem convergência: {result.nonconverged_mlp_fits}."
    )


if __name__ == "__main__":
    main()
