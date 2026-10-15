"""Refresh only the forecast line charts in the saved Teste* result folders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malaria_forecasting.evaluation import _plot_forecast_line  # noqa: E402


def main() -> None:
    results_root = ROOT / "results"
    updated = []
    for output_dir in sorted(results_root.glob("Teste*")):
        predictions_path = output_dir / "previsoes_teste.csv"
        metrics_path = output_dir / "metricas_por_metodo.csv"
        metadata_path = output_dir / "resumo_execucao.json"
        if not all(path.exists() for path in (predictions_path, metrics_path, metadata_path)):
            continue

        predictions = pd.read_csv(predictions_path, parse_dates=["mes_alvo"])
        predictions = predictions.set_index("mes_alvo")
        metrics = pd.read_csv(metrics_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_end = pd.Period(metadata["treino_termina_em"], freq="M").to_timestamp()
        test_start = pd.Period(metadata["teste_comeca_em"], freq="M").to_timestamp()

        _plot_forecast_line(predictions, metrics, output_dir, train_end, test_start)
        updated.append(output_dir.name)

    print("Gráficos de previsões atualizados: " + ", ".join(updated))


if __name__ == "__main__":
    main()
