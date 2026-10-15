"""Command-line entry point for reproducible model comparisons."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .data import CSV_FILENAME, load_monthly_counts
from .evaluation import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara previsores mensais de contagens de malária em Breves-PA."
    )
    parser.add_argument("experiment", nargs="?", default="Teste1",
                        help="Nome da pasta de saída, por exemplo Teste1 ou Teste2.")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "data" / "raw" / CSV_FILENAME,
                        help="Caminho do CSV de notificações.")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results",
                        help="Pasta que receberá uma subpasta por experimento.")
    parser.add_argument("--train-fraction", type=float, default=0.5,
                        help="Fração cronológica inicial para treino/desenvolvimento (padrão: 0.5).")
    parser.add_argument("--cv-splits", type=int, default=3,
                        help="Número de dobras temporais dentro da metade de treino.")
    parser.add_argument("--cv-horizon", type=int, default=12,
                        help="Meses de validação em cada dobra interna.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Reduz a validação interna para 2 dobras de 6 meses; use só para exploração.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment):
        parser.error("O nome do experimento pode conter apenas letras, números, _ e -.")
    if args.cv_splits < 2 or args.cv_horizon < 1:
        parser.error("--cv-splits deve ser >= 2 e --cv-horizon deve ser positivo.")
    if args.quick:
        args.cv_splits = 2
        args.cv_horizon = 6
        print("Execução exploratória rápida: validação interna de 2 × 6 meses.")

    results_root = args.results_root.expanduser().resolve()
    output_dir = results_root / args.experiment
    if output_dir.exists():
        existing = [path for path in output_dir.iterdir() if path.name not in {"README.md", ".gitkeep"}]
        if existing:
            parser.error(
                f"A pasta {output_dir} já contém resultados. Escolha outro nome de experimento "
                "para preservar cada execução."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly_counts(args.data)
    print(
        f"Série completa: {monthly.counts.index.min():%Y-%m} a "
        f"{monthly.counts.index.max():%Y-%m} ({len(monthly.counts)} meses)."
    )
    if monthly.excluded_partial_month is not None:
        print(f"Mês final parcial excluído: {monthly.excluded_partial_month:%Y-%m}.")
    if monthly.inserted_zero_months:
        print(
            f"Meses sem linhas preenchidos com zero: {len(monthly.inserted_zero_months)}; "
            "confirme se isso representa ausência real de casos."
        )

    result = run_experiment(
        monthly_data=monthly,
        output_dir=output_dir,
        train_fraction=args.train_fraction,
        cv_splits=args.cv_splits,
        cv_horizon=args.cv_horizon,
        seed=args.seed,
    )
    print(
        f"Treino: {result.metadata['alvos_treinamento']} meses até "
        f"{result.metadata['treino_termina_em']}; teste: "
        f"{result.metadata['alvos_teste']} previsões de "
        f"{result.metadata['teste_comeca_em']} a {result.metadata['teste_termina_em']}."
    )
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Resultados salvos em: {output_dir}")


if __name__ == "__main__":
    main()
