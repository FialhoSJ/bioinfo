"""Chronological train/test evaluation for monthly one-step forecasts."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from .data import MonthlyData, make_supervised
from .models import candidate_configurations, forecast_one_step, tune_model


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_configurations: pd.DataFrame
    inner_validation: pd.DataFrame
    paired_comparisons: pd.DataFrame
    metadata: dict[str, Any]


def _score(y_true: np.ndarray, y_pred: np.ndarray, mase_scale: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    smape = (
        float(np.mean(2 * np.abs(y_true[valid] - y_pred[valid]) / denom[valid]) * 100)
        if valid.any() else np.nan
    )
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE_pct": smape,
        "MASE": float(mean_absolute_error(y_true, y_pred) / mase_scale)
        if mase_scale > 0 else np.nan,
        "Bias_casos": float(np.mean(y_pred - y_true)),
        "Poisson_deviance": float(mean_poisson_deviance(y_true, np.maximum(y_pred, 1e-9))),
    }


def _paired_block_bootstrap(
    predictions: pd.DataFrame,
    reference: str = "naive_ultimo_mes",
    block_length: int = 12,
    repetitions: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    if reference not in predictions:
        return pd.DataFrame()
    actual = predictions["real"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for method in [column for column in predictions if column not in {"real", reference}]:
        model_prediction = predictions[method].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(model_prediction)
        if not valid.all():
            continue
        reference_prediction = predictions[reference].to_numpy(dtype=float)
        paired = np.abs(model_prediction - actual) - np.abs(reference_prediction - actual)
        if not len(paired):
            continue
        mean = float(np.mean(paired))
        length = min(max(1, block_length), len(paired))
        block_count = int(np.ceil(len(paired) / length))
        offsets = np.arange(length)
        draws = np.empty(repetitions, dtype=float)
        for draw in range(repetitions):
            starts = rng.integers(0, len(paired), size=block_count)
            positions = np.concatenate([(start + offsets) % len(paired) for start in starts])
            draws[draw] = np.mean(paired[positions[:len(paired)]])
        lower, upper = np.quantile(draws, [0.025, 0.975])
        rows.append({
            "modelo": method,
            "referencia": reference,
            "diferenca_MAE_casos": mean,
            "IC95_inf": float(lower),
            "IC95_sup": float(upper),
            "meses": len(paired),
        })
    return pd.DataFrame(rows)


def _plot_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
):
    ranked = metrics.dropna(subset=["RMSE"]).sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(ranked))))
    ax.barh(ranked["modelo"], ranked["RMSE"], color="#287c74")
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (casos; menor é melhor)")
    ax.set_title("Comparação de métodos — teste temporal")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_por_metodo.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    valid_methods = [name for name in ranked["modelo"].head(5) if name in predictions]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(predictions.index, predictions["real"], color="#202124", linewidth=2.2,
            label="Real observado", zorder=5)
    palette = ["#0077b6", "#e76f00", "#2a9d8f", "#7251b5", "#d1495b"]
    for name, color in zip(valid_methods, palette):
        ax.plot(predictions.index, predictions[name], linewidth=1.35,
                alpha=0.9, color=color, label=name)
    ax.axvline(test_start, color="#555555", linestyle="--", linewidth=1)
    ax.set_title(
        f"Previsões sequenciais de um mês — treino até {train_end:%m/%Y}; "
        f"teste desde {test_start:%m/%Y}"
    )
    ax.set_ylabel("Notificações de malária")
    ax.set_xlabel("Mês previsto")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "previsoes_teste.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    monthly_data: MonthlyData,
    output_dir: str | Path,
    train_fraction: float = 0.5,
    cv_splits: int = 3,
    cv_horizon: int = 12,
    seed: int = 42,
) -> ExperimentResult:
    """Tune inside the chronological first half, then score the second half.

    The second half is evaluated as a sequence of one-month-ahead forecasts.
    After each target month is observed, it may be used to fit the next origin's
    model. Hyperparameter/configuration selection remains locked to the first
    half, so no future target is used to select a method's configuration.
    """
    if not 0.25 <= train_fraction <= 0.8:
        raise ValueError("--train-fraction precisa ficar entre 0.25 e 0.8.")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    counts = monthly_data.counts.sort_index()
    supervised = make_supervised(counts)
    split = int(np.floor(len(supervised) * train_fraction))
    if split < cv_splits * cv_horizon + 24 or split >= len(supervised):
        raise ValueError(
            f"Divisão inválida: {split} alvos de treino para validação interna e "
            f"{len(supervised) - split} alvos de teste. Ajuste a fração ou CV."
        )
    development = supervised.iloc[:split].copy()
    test = supervised.iloc[split:].copy()
    train_end = development.index[-1]
    test_start = test.index[0]
    config_grid = candidate_configurations()
    metrics_rows: list[dict[str, Any]] = []
    prediction_columns: dict[str, np.ndarray] = {"real": test["target"].to_numpy(dtype=float)}
    selected_rows: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    all_methods = list(config_grid)

    for method in all_methods:
        print(f"Ajustando configuração: {method}")
        best_config, cv_rmse, cv_trace = tune_model(
            method=method,
            configurations=config_grid[method],
            development=development,
            observed_counts=counts,
            all_supervised=supervised,
            cv_splits=cv_splits,
            cv_horizon=cv_horizon,
            seed=seed,
        )
        cv_trace["mes_alvo_teste_interno"] = cv_trace["fold"].astype(str)
        validation_frames.append(cv_trace)
        selected_rows.append({
            "modelo": method,
            "configuracao_escolhida": repr(best_config),
            "RMSE_validacao_interna": cv_rmse,
            "configuracoes_testadas": len(config_grid[method]),
        })

        predictions: list[float] = []
        for step, target_month in enumerate(test.index):
            try:
                prediction = forecast_one_step(
                    method=method,
                    config=best_config,
                    observed_counts=counts,
                    supervised=supervised,
                    target_month=target_month,
                    seed=seed + step,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Falha ao prever {target_month:%Y-%m} com {method} "
                    f"({best_config}): {type(exc).__name__}: {exc}"
                ) from exc
            predictions.append(max(0.0, float(prediction)))
        prediction_columns[method] = np.asarray(predictions, dtype=float)

    forecast_frame = pd.DataFrame(prediction_columns, index=test.index)
    forecast_frame.index.name = "mes_alvo"
    mase_errors = np.abs(
        development["target"].to_numpy(dtype=float)[12:]
        - development["target"].to_numpy(dtype=float)[:-12]
    )
    mase_scale = float(np.mean(mase_errors)) if len(mase_errors) else np.nan
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        mase_scale = float(np.mean(np.abs(np.diff(development["target"].to_numpy(dtype=float)))))

    for method in all_methods:
        metrics_rows.append({
            "modelo": method,
            "alvos_teste": len(test),
            **_score(
                forecast_frame["real"].to_numpy(dtype=float),
                forecast_frame[method].to_numpy(dtype=float),
                mase_scale,
            ),
        })
    metrics = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
    paired = _paired_block_bootstrap(forecast_frame, seed=seed)
    validation = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    metadata = {
        "fonte_csv": str(monthly_data.source_path),
        "municipio": "Breves",
        "uf": "PA",
        "linhas_validas_municipio_uf": monthly_data.source_rows,
        "alvo": "soma mensal de Número de registros por Data de Notificação",
        "primeiro_mes_da_serie": counts.index.min().strftime("%Y-%m"),
        "ultimo_mes_completo": counts.index.max().strftime("%Y-%m"),
        "ultima_data_de_notificacao_no_csv": monthly_data.latest_notification.strftime("%Y-%m-%d"),
        "mes_parcial_excluido": (
            monthly_data.excluded_partial_month.strftime("%Y-%m")
            if monthly_data.excluded_partial_month is not None else None
        ),
        "meses_com_contagem_zero_inseridos": [month.strftime("%Y-%m") for month in monthly_data.inserted_zero_months],
        "meses_da_serie": len(counts),
        "soma_total_mensal": float(counts.sum()),
        "media_mensal": float(counts.mean()),
        "variancia_mensal": float(counts.var(ddof=1)),
        "razao_variancia_media_bruta": float(counts.var(ddof=1) / counts.mean()),
        "meses_com_zero": int((counts == 0).sum()),
        "alvos_supervisionados": len(supervised),
        "fracao_treinamento": train_fraction,
        "alvos_treinamento": len(development),
        "alvos_teste": len(test),
        "treino_termina_em": train_end.strftime("%Y-%m"),
        "teste_comeca_em": test_start.strftime("%Y-%m"),
        "teste_termina_em": test.index[-1].strftime("%Y-%m"),
        "politica_teste": "previsao sequencial de um mes; cada real anterior passa a estar disponivel no proximo origem",
        "semente": seed,
        "validacao_interna": f"TimeSeriesSplit: {cv_splits} dobras, {cv_horizon} meses por dobra, somente primeira metade",
        "metodos": all_methods,
        "ajuste_climatico": "nao usado: a fonte atual nao contem covariaveis climaticas",
    }

    metrics.to_csv(output_path / "metricas_por_metodo.csv", index=False, encoding="utf-8-sig")
    forecast_frame.to_csv(output_path / "previsoes_teste.csv", encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(
        output_path / "configuracoes_selecionadas.csv", index=False, encoding="utf-8-sig"
    )
    validation.to_csv(output_path / "validacao_interna.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output_path / "diferencas_pareadas.csv", index=False, encoding="utf-8-sig")
    (output_path / "resumo_execucao.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_outputs(forecast_frame, metrics, output_path, train_end, test_start)
    return ExperimentResult(
        metrics=metrics,
        predictions=forecast_frame,
        selected_configurations=pd.DataFrame(selected_rows),
        inner_validation=validation,
        paired_comparisons=paired,
        metadata=metadata,
    )
