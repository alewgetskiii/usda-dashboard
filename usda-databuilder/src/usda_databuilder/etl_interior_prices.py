import pandas as pd
from sodapy import Socrata
from datetime import datetime
from .config import SOCRATA_DOMAIN, SOCRATA_APP_TOKEN
from .db import get_connection

DATASET_ID = "ecws-wvgk"


def fetch_interior_prices(limit: int = 50000) -> pd.DataFrame:
    client = Socrata(SOCRATA_DOMAIN, SOCRATA_APP_TOKEN)
    results = client.get(DATASET_ID, limit=limit)
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # conversions de base
    df["date"] = pd.to_datetime(df["date"])
    for col in ["origin_price", "destination_price", "price_spread"]:
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
        is_week_end = 1  # par convention pour ces datasets

        cur.execute(
            """INSERT INTO dim_date
            (date_iso, year, month, day, week_of_year, quarter, is_week_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (date_iso, year, month, day, week_of_year, quarter, is_week_end),
        )
        ids[dt] = cur.lastrowid
    return ids


def upsert_dim_transport_route(conn, commodityod_series: pd.Series):
    cur = conn.cursor()
    ids = {}
    for cod in commodityod_series.dropna().unique():
        cur.execute(
            "SELECT route_id FROM dim_transport_route WHERE commodityod = ?",
            (cod,),
        )
        row = cur.fetchone()
        if row:
            ids[cod] = row[0]
            continue

        cur.execute(
            "INSERT INTO dim_transport_route (commodityod) VALUES (?)",
            (cod,),
        )
        ids[cod] = cur.lastrowid
    return ids


def load_interior_prices():
    df = fetch_interior_prices()
    if df.empty:
        print("Aucune donnée interior prices récupérée.")
        return

    with get_connection() as conn:
        date_map = upsert_dim_date(conn, df["date"])
        route_map = upsert_dim_transport_route(conn, df["commodityod"])

        cur = conn.cursor()
        load_time = datetime.utcnow().isoformat()

        inserted = 0
        for _, row in df.iterrows():
            dt = row["date"]
            date_id = date_map.get(dt)
            route_id = route_map.get(row["commodityod"])
            year = int(row["year"])
            month = int(row["month"])

            cur.execute(
                """INSERT OR REPLACE INTO fact_interior_price_spread
                (id, date_id, route_id, year, month,
                 origin_price, destination_price, price_spread,
                 source_system, load_time)
                VALUES (
                    COALESCE(
                        (SELECT id FROM fact_interior_price_spread
                         WHERE date_id = ? AND route_id = ?),
                        NULL
                    ),
                    ?, ?, ?, ?, ?, ?, ?, 'USDA_TRANSPORT_ECWS_WVGK', ?
                )
                """,
                (
                    date_id,
                    route_id,
                    date_id,
                    route_id,
                    year,
                    month,
                    row.get("origin_price"),
                    row.get("destination_price"),
                    row.get("price_spread"),
                    load_time,
                ),
            )
            inserted += 1

        print(f"Interior prices chargées: {inserted} lignes.")
