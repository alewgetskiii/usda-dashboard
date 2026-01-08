from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, DB_PATH
from .db import get_connection

# Ensure the BR package is importable when running from src/.
BR_PACKAGE_ROOT = PROJECT_ROOT
if str(BR_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BR_PACKAGE_ROOT))

from BR.scripts.build_all_datasets import main as build_br_datasets  # type: ignore

BR_FINAL_CSV = (
    PROJECT_ROOT / "BR" / "data_intermediate" / "mt_costs_with_sima_prices.csv"
)

_BACKUP_DONE = False


def backup_database_once() -> Path | None:
    """Crée un backup horodaté du fichier SQLite avant modification (une seule fois)."""
    global _BACKUP_DONE
    if _BACKUP_DONE or not DB_PATH.exists():
        _BACKUP_DONE = True
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup_name = f"{DB_PATH.stem}_backup_{timestamp}{DB_PATH.suffix}"
    backup_path = DB_PATH.parent / backup_name
    shutil.copy2(DB_PATH, backup_path)
    _BACKUP_DONE = True
    print(f"Backup SQLite créé avant chargement BR: {backup_path}")
    return backup_path


def upsert_dim_date(conn, date_series: pd.Series) -> dict:
    cur = conn.cursor()
    ids = {}
    for dt in pd.to_datetime(date_series.dropna().unique()):
        date_iso = dt.date().isoformat()
        cur.execute("SELECT date_id FROM dim_date WHERE date_iso = ?", (date_iso,))
        row = cur.fetchone()
        if row:
            ids[dt] = row[0]
            continue

        year = dt.year
        month = dt.month
        day = dt.day
        week_of_year = int(dt.strftime("%W")) + 1
        quarter = (month - 1) // 3 + 1
        is_week_end = 0

        cur.execute(
            """INSERT INTO dim_date
            (date_iso, year, month, day, week_of_year, quarter, is_week_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (date_iso, year, month, day, week_of_year, quarter, is_week_end),
        )
        ids[dt] = cur.lastrowid
    return ids


def load_br_prices(refresh_data: bool = True, csv_path: Path | None = None) -> None:
    """
    Orchestration pour charger les prix BR (MT + PR) dans fact_br_prices.
    - Optionnellement relance la génération des CSV via build_all_datasets.
    - Crée un backup SQLite avant toute modification.
    """
    if refresh_data:
        print("Construction des datasets BR (IMEA + SIMA)...")
        build_br_datasets()

    csv_path = csv_path or BR_FINAL_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Impossible de trouver le fichier fusionné: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        print("Aucune donnée BR à charger.")
        return

    df["price_value"] = pd.to_numeric(df["price_value"], errors="coerce")
    df = df.dropna(subset=["date", "price_value"])

    backup_database_once()

    with get_connection() as conn:
        date_map = upsert_dim_date(conn, df["date"])
        cur = conn.cursor()
        load_time = datetime.utcnow().isoformat()

        def safe_str(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, float) and pd.isna(value):
                return ""
            return str(value).strip()

        inserted = 0
        for row in df.itertuples(index=False):
            dt = pd.to_datetime(row.date)
            date_id = date_map.get(dt)
            if not date_id:
                continue

            state = safe_str(row.state)
            macroregion = safe_str(row.macroregion)
            city = safe_str(row.city)
            destination_city = safe_str(row.destination_city)
            commodity = safe_str(row.commodity)
            metric = safe_str(row.metric) or "Price"
            unit = safe_str(row.unit)
            product_item = safe_str(row.product_item)
            price_value = float(row.price_value)
            source = safe_str(row.source_combined)

            cur.execute(
                """INSERT INTO fact_br_prices (
                        date_id, state, macroregion, city, destination_city,
                        commodity, metric, unit, product_item, price_value,
                        source, load_time
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date_id, state, macroregion, city, destination_city,
                                commodity, metric, unit, product_item)
                    DO UPDATE SET
                        price_value=excluded.price_value,
                        source=excluded.source,
                        load_time=excluded.load_time
                """,
                (
                    date_id,
                    state,
                    macroregion or None,
                    city or None,
                    destination_city or None,
                    commodity,
                    metric,
                    unit or None,
                    product_item or None,
                    price_value,
                    source or None,
                    load_time,
                ),
            )
            inserted += 1

    print(f"Chargement fact_br_prices terminé : {inserted} lignes traitées.")
