"""Loading and preparing monthly notification counts from the source CSV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CSV_FILENAME = "6- Autóctones Mun Fx Etária e Sexo_Dados completos_data.csv"
SOURCE_COLUMNS = {
    "BR Mun de infecção": "municipio",
    "BR UF Infecção": "uf",
    "BR Data de Notificação": "data_notificacao",
    "Número de registros": "casos",
}


@dataclass(frozen=True)
class MonthlyData:
    counts: pd.Series
    source_path: Path
    latest_notification: pd.Timestamp
    excluded_partial_month: pd.Timestamp | None
    inserted_zero_months: tuple[pd.Timestamp, ...]
    source_rows: int


def load_monthly_counts(
    csv_path: str | Path,
    municipality: str = "Breves",
    state: str = "PA",
    chunksize: int = 250_000,
) -> MonthlyData:
    """Aggregate Breves-PA notifications by month and exclude a partial final month.

    The CSV header has contained trailing whitespace in some versions, so column
    names are normalized before selection. Months with no source rows are filled
    with zero and reported in ``inserted_zero_months`` for review.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    normalized_to_original = {
        str(column).replace("\ufeff", "").strip(): column
        for column in pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=0).columns
    }
    missing = sorted(set(SOURCE_COLUMNS) - set(normalized_to_original))
    if missing:
        raise ValueError(f"Colunas esperadas não encontradas no CSV: {missing}")

    original_columns = [normalized_to_original[name] for name in SOURCE_COLUMNS]
    monthly_parts: list[pd.Series] = []
    latest_notification: pd.Timestamp | None = None
    source_rows = 0

    for chunk in pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        usecols=original_columns,
        dtype={column: "string" for column in original_columns},
        chunksize=chunksize,
    ):
        chunk = chunk.rename(
            columns={
                original: SOURCE_COLUMNS[normalized]
                for normalized, original in normalized_to_original.items()
                if original in chunk.columns
            }
        )
        municipality_values = chunk["municipio"].fillna("").str.strip().str.casefold()
        state_values = chunk["uf"].fillna("").str.strip().str.upper()
        selected = chunk.loc[
            municipality_values.eq(municipality.strip().casefold())
            & state_values.eq(state.strip().upper())
        ].copy()
        if selected.empty:
            continue

        selected["data_notificacao"] = pd.to_datetime(
            selected["data_notificacao"], format="%d/%m/%Y", errors="coerce"
        )
        selected["casos"] = pd.to_numeric(selected["casos"], errors="coerce")
        selected = selected.dropna(subset=["data_notificacao", "casos"])
        if selected.empty:
            continue
        if (selected["casos"] < 0).any():
            raise ValueError("O CSV contém contagens negativas.")
        source_rows += len(selected)
        chunk_latest = selected["data_notificacao"].max()
        latest_notification = (
            chunk_latest if latest_notification is None else max(latest_notification, chunk_latest)
        )
        selected["mes"] = selected["data_notificacao"].dt.to_period("M").dt.to_timestamp()
        monthly_parts.append(selected.groupby("mes")["casos"].sum())

    if not monthly_parts or latest_notification is None:
        raise ValueError(f"Nenhuma notificação válida foi encontrada para {municipality}-{state}.")

    observed_monthly = pd.concat(monthly_parts).groupby(level=0).sum().sort_index()
    latest_month = latest_notification.to_period("M").to_timestamp()
    month_end = latest_notification.to_period("M").end_time.normalize()
    excluded_partial_month = None
    if latest_notification.normalize() < month_end:
        observed_monthly = observed_monthly.loc[observed_monthly.index < latest_month]
        excluded_partial_month = latest_month
    if observed_monthly.empty:
        raise ValueError("Não há meses completos para avaliar após excluir o mês parcial.")

    full_index = pd.date_range(
        observed_monthly.index.min(), observed_monthly.index.max(), freq="MS", name="mes"
    )
    inserted = tuple(full_index.difference(observed_monthly.index).to_pydatetime())
    counts = observed_monthly.reindex(full_index, fill_value=0).astype(float)
    counts.name = "casos"
    if not np.isfinite(counts.to_numpy()).all():
        raise ValueError("A série mensal contém valores ausentes ou não finitos.")
    return MonthlyData(
        counts=counts,
        source_path=path,
        latest_notification=latest_notification,
        excluded_partial_month=excluded_partial_month,
        inserted_zero_months=inserted,
        source_rows=source_rows,
    )


def make_supervised(counts: pd.Series, minimum_lag: int = 12) -> pd.DataFrame:
    """Create one-step-ahead rows using only counts known before the target month."""
    if minimum_lag < 12:
        raise ValueError("minimum_lag deve ser pelo menos 12 para representar a sazonalidade anual.")
    counts = counts.astype(float).sort_index()
    rows: list[dict[str, float]] = []
    target_months: list[pd.Timestamp] = []
    for target_position in range(minimum_lag, len(counts)):
        origin_position = target_position - 1
        history = counts.iloc[: origin_position + 1]
        target_month = counts.index[target_position]
        row: dict[str, float] = {}
        for lag in (1, 2, 3, 4, 7, 12):
            row[f"lag_{lag}"] = float(history.iloc[-lag])
        for window in (3, 6, 12):
            row[f"media_{window}"] = float(history.iloc[-window:].mean())
        row["tendencia_3m"] = float(history.iloc[-1] - history.iloc[-4])
        month = target_month.month
        row["mes_sin"] = float(np.sin(2 * np.pi * month / 12))
        row["mes_cos"] = float(np.cos(2 * np.pi * month / 12))
        row["ano_indice"] = float((target_month.year - counts.index[0].year) / 10.0)
        row["target"] = float(counts.iloc[target_position])
        rows.append(row)
        target_months.append(target_month)
    frame = pd.DataFrame(rows, index=pd.DatetimeIndex(target_months, name="mes_alvo"))
    if frame.empty:
        raise ValueError("Há poucos meses para formar atributos com 12 meses de histórico.")
    return frame


def make_feature_row(counts_before_target: pd.Series, target_month: pd.Timestamp) -> pd.DataFrame:
    """Build the predictors for one target month from its preceding observations."""
    history = counts_before_target.astype(float).sort_index()
    if len(history) < 12:
        raise ValueError("A previsão requer pelo menos 12 meses anteriores.")
    row: dict[str, float] = {}
    for lag in (1, 2, 3, 4, 7, 12):
        row[f"lag_{lag}"] = float(history.iloc[-lag])
    for window in (3, 6, 12):
        row[f"media_{window}"] = float(history.iloc[-window:].mean())
    row["tendencia_3m"] = float(history.iloc[-1] - history.iloc[-4])
    row["mes_sin"] = float(np.sin(2 * np.pi * target_month.month / 12))
    row["mes_cos"] = float(np.cos(2 * np.pi * target_month.month / 12))
    row["ano_indice"] = float((target_month.year - history.index[0].year) / 10.0)
    return pd.DataFrame([row], index=pd.DatetimeIndex([target_month], name="mes_alvo"))
