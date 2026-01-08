from datetime import datetime

import pandas as pd
from sodapy import Socrata

from .config import DATA_DIR, SOCRATA_APP_TOKEN, SOCRATA_DOMAIN
from .db import get_connection


DATASET_ID = "3az4-jkr6"

RAW_CSV_PATH = DATA_DIR / "rail_rates_raw.csv"
PREPARED_CSV_PATH = DATA_DIR / "rail_rates_prepared.csv"

NUMERIC_COLUMNS = [
    "tariff",
    "item",
    "route_mileage",
    "min_number_of_cars",
    "max_number_of_cars",
    "tariff_per_car",
    "tariff_per_bushel",
    "tariff_per_metric_ton",
    "tariff_per_short_ton",
    "tariff_per_ton_mile",
    "fsc_per_mile",
    "fsc_per_car",
    "tariff_fsc_per_car",
    "tariff_fsc_per_bushel",
    "tariff_fsc_per_metric_ton",
    "tariff_fsc_per_short_ton",
    "tariff_fsc_per_ton_mile",
]

BOOL_COLUMNS = [
    "rule_11_rate",
    "gtr_table",
]

TEXT_COLUMNS = [
    "railroad",
    "commodity",
    "primary_class",
    "origin_city",
    "origin_state",
    "origin",
    "destination_city",
    "destination_state",
    "destination",
    "train_type",
    "car_ownership",
    "maximum_load_pounds",
    "car_volume_cubic_feet",
]


def fetch_rail_rates(limit: int = 50000) -> pd.DataFrame:
    client = Socrata(SOCRATA_DOMAIN, SOCRATA_APP_TOKEN)
    results = client.get(DATASET_ID, limit=limit)
    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0})
            )

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", "")

    return df


def prepare_rail_rates(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "date",
        "tariff",
        "item",
        "commodity",
        "primary_class",
        "railroad",
        "origin",
        "origin_city",
        "origin_state",
        "destination",
        "destination_city",
        "destination_state",
        "route_mileage",
        "train_type",
        "min_number_of_cars",
        "max_number_of_cars",
        "car_ownership",
        "rule_11_rate",
        "gtr_table",
        "maximum_load_pounds",
        "car_volume_cubic_feet",
        "tariff_fsc_per_bushel",
        "tariff_fsc_per_metric_ton",
        "tariff_fsc_per_short_ton",
        "tariff_fsc_per_ton_mile",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    return out


def save_rail_rates_csvs():
    df = fetch_rail_rates()
    if df.empty:
        print("Aucune donnée rail récupérée.")
        return

    df.to_csv(RAW_CSV_PATH, index=False)
    prepared = prepare_rail_rates(df)
    prepared.to_csv(PREPARED_CSV_PATH, index=False)

    print(f"✅ Raw saved: {RAW_CSV_PATH}")
    print(f"✅ Prepared saved: {PREPARED_CSV_PATH}")


def upsert_dim_date(conn, date_series: pd.Series):
    cur = conn.cursor()
    ids = {}
    for dt in date_series.dropna().unique():
        date_iso = dt.date().isoformat()
        cur.execute(
            "SELECT date_id FROM dim_date WHERE date_iso = ?",
            (date_iso,),
        )
        row = cur.fetchone()
        if row:
            ids[dt] = row[0]
            continue

        year = dt.year
        month = dt.month
        day = dt.day
        week_of_year = int(dt.strftime("%W")) + 1
        quarter = (month - 1) // 3 + 1
        is_week_end = 1

        cur.execute(
            """INSERT INTO dim_date
            (date_iso, year, month, day, week_of_year, quarter, is_week_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (date_iso, year, month, day, week_of_year, quarter, is_week_end),
        )
        ids[dt] = cur.lastrowid
    return ids


def upsert_dim_transport_location(conn, loc_series: pd.Series):
    cur = conn.cursor()
    ids = {}
    for loc in loc_series.dropna().unique():
        cur.execute(
            "SELECT transport_location_id FROM dim_transport_location WHERE location_name = ?",
            (loc,),
        )
        row = cur.fetchone()
        if row:
            ids[loc] = row[0]
            continue

        cur.execute(
            "INSERT INTO dim_transport_location (location_name) VALUES (?)",
            (loc,),
        )
        ids[loc] = cur.lastrowid
    return ids


def upsert_dim_rail_route(conn, routes: pd.DataFrame):
    cur = conn.cursor()
    ids = {}
    route_rows = routes.dropna(subset=["origin_loc_id", "destination_loc_id"]).drop_duplicates(
        ["origin_loc_id", "destination_loc_id", "origin", "destination", "route_mileage"]
    )
    for _, row in route_rows.iterrows():
        key = (
            int(row["origin_loc_id"]),
            int(row["destination_loc_id"]),
            row.get("origin"),
            row.get("destination"),
            row.get("route_mileage"),
        )
        cur.execute(
            """SELECT rail_route_id FROM dim_rail_route
            WHERE origin_location_id = ?
              AND destination_location_id = ?
              AND IFNULL(origin_desc, '') = IFNULL(?, '')
              AND IFNULL(destination_desc, '') = IFNULL(?, '')
              AND IFNULL(route_mileage, -1) = IFNULL(?, -1)
            """,
            key,
        )
        row_db = cur.fetchone()
        if row_db:
            ids[key] = row_db[0]
            continue

        cur.execute(
            """INSERT INTO dim_rail_route
            (origin_location_id, destination_location_id, origin_desc, destination_desc, route_mileage)
            VALUES (?, ?, ?, ?, ?)
            """,
            key,
        )
        ids[key] = cur.lastrowid
    return ids


def build_location_name(city: str | None, state: str | None, gtr_name: str | None) -> str | None:
    if gtr_name:
        return gtr_name
    city = (city or "").strip()
    state = (state or "").strip()
    if city or state:
        return f"{city}, {state}".strip().strip(",")
    return None


def load_rail_rates():
    df = fetch_rail_rates()
    if df.empty:
        print("Aucune donnée rail récupérée.")
        return

    df["origin_loc_name"] = df.apply(
        lambda r: build_location_name(r.get("origin_city"), r.get("origin_state"), r.get("origin")),
        axis=1,
    )
    df["destination_loc_name"] = df.apply(
        lambda r: build_location_name(
            r.get("destination_city"), r.get("destination_state"), r.get("destination")
        ),
        axis=1,
    )

    with get_connection() as conn:
        date_map = upsert_dim_date(conn, df["date"])
        loc_map = upsert_dim_transport_location(
            conn,
            pd.concat([df["origin_loc_name"], df["destination_loc_name"]], ignore_index=True),
        )

        df["origin_loc_id"] = df["origin_loc_name"].map(loc_map)
        df["destination_loc_id"] = df["destination_loc_name"].map(loc_map)

        route_map = upsert_dim_rail_route(conn, df)

        cur = conn.cursor()
        load_time = datetime.utcnow().isoformat()
        inserted = 0

        for _, row in df.iterrows():
            dt = row["date"]
            date_id = date_map.get(dt)
            if date_id is None:
                continue

            route_key = (
                int(row["origin_loc_id"]) if not pd.isna(row["origin_loc_id"]) else None,
                int(row["destination_loc_id"]) if not pd.isna(row["destination_loc_id"]) else None,
                row.get("origin"),
                row.get("destination"),
                row.get("route_mileage"),
            )
            rail_route_id = route_map.get(route_key)
            if rail_route_id is None:
                continue

            cur.execute(
                """INSERT OR REPLACE INTO fact_rail_rate
                (rail_rate_id, date_id, rail_route_id, railroad, tariff, item,
                 commodity, primary_class, train_type, min_number_of_cars,
                 max_number_of_cars, car_ownership, rule_11_rate, gtr_table,
                 maximum_load_pounds, car_volume_cubic_feet,
                 tariff_fsc_per_bushel, tariff_fsc_per_metric_ton,
                 tariff_fsc_per_short_ton, tariff_fsc_per_ton_mile,
                 source_system, load_time)
                VALUES (
                    COALESCE(
                        (SELECT rail_rate_id FROM fact_rail_rate
                         WHERE date_id = ? AND rail_route_id = ?
                           AND IFNULL(railroad, '') = IFNULL(?, '')
                           AND IFNULL(commodity, '') = IFNULL(?, '')
                           AND IFNULL(train_type, '') = IFNULL(?, '')
                           AND IFNULL(car_ownership, '') = IFNULL(?, '')
                           AND IFNULL(tariff, -1) = IFNULL(?, -1)
                           AND IFNULL(item, -1) = IFNULL(?, -1)),
                        NULL
                    ),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'USDA_TRANSPORT_3AZ4_JKR6', ?
                )
                """,
                (
                    date_id,
                    rail_route_id,
                    row.get("railroad"),
                    row.get("commodity"),
                    row.get("train_type"),
                    row.get("car_ownership"),
                    row.get("tariff"),
                    row.get("item"),
                    date_id,
                    rail_route_id,
                    row.get("railroad"),
                    row.get("tariff"),
                    row.get("item"),
                    row.get("commodity"),
                    row.get("primary_class"),
                    row.get("train_type"),
                    row.get("min_number_of_cars"),
                    row.get("max_number_of_cars"),
                    row.get("car_ownership"),
                    row.get("rule_11_rate"),
                    row.get("gtr_table"),
                    row.get("maximum_load_pounds"),
                    row.get("car_volume_cubic_feet"),
                    row.get("tariff_fsc_per_bushel"),
                    row.get("tariff_fsc_per_metric_ton"),
                    row.get("tariff_fsc_per_short_ton"),
                    row.get("tariff_fsc_per_ton_mile"),
                    load_time,
                ),
            )
            inserted += 1

        print(f"Rail rates chargées: {inserted} lignes.")


if __name__ == "__main__":
    save_rail_rates_csvs()
    load_rail_rates()
