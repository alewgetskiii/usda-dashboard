import pandas as pd
from sodapy import Socrata
from datetime import datetime
from .config import SOCRATA_DOMAIN, SOCRATA_APP_TOKEN
from .db import get_connection

DATASET_ID = "deqi-uken"


def fetch_barge_rates(limit: int = 50000) -> pd.DataFrame:
    client = Socrata(SOCRATA_DOMAIN, SOCRATA_APP_TOKEN)
    results = client.get(DATASET_ID, limit=limit)
    df = pd.DataFrame(results)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    for col in ["week", "month", "year", "rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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


def load_barge_rates():
    df = fetch_barge_rates()
    if df.empty:
        print("Aucune donnée barge récupérée.")
        return

    with get_connection() as conn:
        date_map = upsert_dim_date(conn, df["date"])
        loc_map = upsert_dim_transport_location(conn, df["location"])

        cur = conn.cursor()
        load_time = datetime.utcnow().isoformat()
        inserted = 0

        for _, row in df.iterrows():
            dt = row["date"]
            date_id = date_map.get(dt)
            loc_id = loc_map.get(row["location"])

            year = int(row["year"])
            month = int(row["month"])
            week_usda = int(row["week"]) if not pd.isna(row["week"]) else None
            rate = row.get("rate")

            if date_id is None or loc_id is None:
                continue

            cur.execute(
                """INSERT OR REPLACE INTO fact_barge_rate
                (id, date_id, year, month, week_usda,
                 transport_location_id, rate_percent_of_tariff,
                 source_system, load_time)
                VALUES (
                    COALESCE(
                        (SELECT id FROM fact_barge_rate
                         WHERE date_id = ? AND transport_location_id = ?),
                        NULL
                    ),
                    ?, ?, ?, ?, ?, ?, 'USDA_TRANSPORT_DEQI_UKEN', ?
                )
                """,
                (
                    date_id,
                    loc_id,
                    date_id,
                    year,
                    month,
                    week_usda,
                    loc_id,
                    rate,
                    load_time,
                ),
            )
            inserted += 1

        print(f"Barge rates chargées: {inserted} lignes.")
