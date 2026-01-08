import re
import requests
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .config import USDA_API_KEY, USDA_BASE_URL
from .db import get_connection


# ---------------------------
# 1. FETCH QUICKSTATS
# ---------------------------

def fetch_usda_data(
    year_from: int,
    year_to: int,
    commodity_desc: str,
    statisticcat_desc: str,
    agg_level_desc: str,
    source_desc: str = "SURVEY",
    domain_desc: Optional[str] = None,
    state_alpha: Optional[Iterable[str]] = None,
    page_size: int = 500000,
) -> pd.DataFrame:
    q: Dict[str, Any] = {
        "key": USDA_API_KEY,
        "page_size": page_size,
        "source_desc": source_desc,
        "commodity_desc": commodity_desc,
        "statisticcat_desc": statisticcat_desc,
        "agg_level_desc": agg_level_desc,
        "year__GE": int(year_from),
        "year__LE": int(year_to),
    }

    if agg_level_desc == "STATE" and state_alpha:
        q["state_alpha"] = list(state_alpha)

    out: List[Dict[str, Any]] = []
    page = 1

    domain_str = f" | Domain: {domain_desc}" if domain_desc else ""
    print(f"-> Fetch: {commodity_desc} | {statisticcat_desc} | {agg_level_desc}{domain_str}...")

    while True:
        try:
            params = {**q, "page": page}
            if domain_desc:
                params["domain_desc"] = domain_desc

            r = requests.get(USDA_BASE_URL, params=params, timeout=60)
            if r.status_code != 200:
                print(f"  ERREUR HTTP {r.status_code}. URL: {r.url}")
                break

            data = r.json().get("data", [])
            if not data:
                break

            out.extend(data)

            if len(data) < q["page_size"]:
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"   Connection Failed: {e}")
            break

    print(f"   Terminé. {len(out)} enregistrements récupérés.")

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)

    # Nettoyage de la colonne Value -> valeur numérique
    if "Value" in df.columns:
        df["Value"] = (
            df["Value"]
            .astype(str)
            .str.replace(r"[,\s]", "", regex=True)
            .str.replace("-", "", regex=False)
        )
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    return df


# ---------------------------
# 2. HELPERS DIMENSIONS
# ---------------------------

def parse_reference_period(desc: Optional[str]):
    """
    Transforme 'YEAR', 'YEAR - AUG FORECAST', 'WEEK #25', 'WEEK 25' en:
    (period_category, week_number, forecast_month, is_forecast)
    """
    if not desc:
        return "UNKNOWN", None, None, 0

    d = desc.upper()

    # Forecast annuelle type "YEAR - AUG FORECAST"
    if "FORECAST" in d and "YEAR" in d:
        period_category = "FORECAST"
        is_forecast = 1
        forecast_month = None

        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        for m_str, m_num in months.items():
            if m_str in d:
                forecast_month = m_num
                break

        return period_category, None, forecast_month, is_forecast

    # Période annuelle classique
    if d.startswith("YEAR"):
        return "YEAR", None, None, 0

    # Période hebdo "WEEK #25", "WEEK 25", etc.
    if "WEEK" in d:
        period_category = "WEEK"
        m = re.search(r"(\d+)", d)
        week_number = int(m.group(1)) if m else None
        return period_category, week_number, None, 0

    # Autre (mois, période exotique)
    return "OTHER", None, None, 0


def upsert_dim_reference_period(conn, ref_desc: Optional[str]) -> int:
    cur = conn.cursor()
    if not ref_desc:
        ref_desc = "UNKNOWN"

    cur.execute(
        "SELECT reference_period_id FROM dim_reference_period WHERE reference_period_desc = ?",
        (ref_desc,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    period_category, week_number, forecast_month, is_forecast = parse_reference_period(ref_desc)
    cur.execute(
        """
        INSERT INTO dim_reference_period
            (reference_period_desc, period_category, week_number, forecast_month, is_forecast)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ref_desc, period_category, week_number, forecast_month, is_forecast),
    )
    return cur.lastrowid


def upsert_dim_commodity(conn, commodity_name: Optional[str],
                         group_desc: Optional[str], sector_desc: Optional[str]) -> Optional[int]:
    if not commodity_name:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT commodity_id FROM dim_commodity WHERE commodity_name = ?",
        (commodity_name,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO dim_commodity (commodity_name, group_desc, sector_desc)
        VALUES (?, ?, ?)
        """,
        (commodity_name, group_desc, sector_desc),
    )
    return cur.lastrowid


def upsert_dim_statistic(conn, statistic_name: Optional[str],
                         unit_desc: Optional[str],
                         freq_desc: Optional[str],
                         reference_type: Optional[str],
                         short_desc: Optional[str],
                         class_desc: Optional[str],
                         util_practice_desc: Optional[str],
                         prodn_practice_desc: Optional[str]) -> Optional[int]:
    if not statistic_name or not unit_desc:
        return None
    cur = conn.cursor()
    # On identifie la statistique par (statistic_name, unit_desc, short_desc)
    cur.execute(
        """
        SELECT statistic_id FROM dim_statistic
        WHERE statistic_name = ? AND unit_desc = ? AND COALESCE(short_desc,'') = COALESCE(?, '')
        """,
        (statistic_name, unit_desc, short_desc),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO dim_statistic
            (statistic_name, unit_desc, freq_desc, reference_type,
             short_desc, class_desc, util_practice_desc, prodn_practice_desc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            statistic_name,
            unit_desc or "",
            freq_desc,
            reference_type,
            short_desc,
            class_desc,
            util_practice_desc,
            prodn_practice_desc,
        ),
    )
    return cur.lastrowid


def upsert_dim_agg_level(conn, agg_level_desc: Optional[str]) -> Optional[int]:
    if not agg_level_desc:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT agg_level_id FROM dim_agg_level WHERE agg_level_desc = ?",
        (agg_level_desc,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        "INSERT INTO dim_agg_level (agg_level_desc) VALUES (?)",
        (agg_level_desc,),
    )
    return cur.lastrowid


def upsert_dim_state(conn,
                     state_alpha: Optional[str],
                     state_name: Optional[str],
                     state_fips_code: Optional[str]) -> Optional[int]:
    if not state_alpha:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT state_id FROM dim_state WHERE state_alpha = ?",
        (state_alpha,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO dim_state (state_alpha, state_name, state_fips_code)
        VALUES (?, ?, ?)
        """,
        (state_alpha, state_name, state_fips_code),
    )
    return cur.lastrowid


def upsert_dim_country(conn,
                       country_name: Optional[str],
                       country_code: Optional[Any]) -> Optional[int]:
    if not country_name or country_code is None:
        return None
    try:
        country_code_int = int(country_code)
    except Exception:
        return None

    cur = conn.cursor()
    cur.execute(
        """
        SELECT country_id FROM dim_country
        WHERE country_code = ? AND country_name = ?
        """,
        (country_code_int, country_name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO dim_country (country_code, country_name)
        VALUES (?, ?)
        """,
        (country_code_int, country_name),
    )
    return cur.lastrowid


def upsert_dim_date_from_week_ending(conn, week_ending_series: pd.Series):
    """
    Pour les séries hebdo (condition etc.) avec une colonne week_ending.
    Retourne un dict {week_ending_datetime -> date_id}
    """
    cur = conn.cursor()
    ids: Dict[pd.Timestamp, int] = {}

    series = pd.to_datetime(week_ending_series.dropna(), errors="coerce")
    for dt in series.dropna().unique():
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
            """
            INSERT INTO dim_date
                (date_iso, year, month, day, week_of_year, quarter, is_week_end)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (date_iso, year, month, day, week_of_year, quarter, is_week_end),
        )
        ids[dt] = cur.lastrowid

    return ids


# ---------------------------
# 3. CHARGEMENT DANS fact_ag_stats
# ---------------------------

def load_quickstats(
    year_from: int,
    year_to: int,
    commodity_desc: str,
    statisticcat_desc: str,
    agg_level_desc: str,
    source_desc: str = "SURVEY",
    domain_desc: Optional[str] = None,
    state_alpha: Optional[Iterable[str]] = None,
):
    """
    Récupère les données Quickstats pour (commodity, statistic, agg_level)
    et les insère dans la table fact_ag_stats + dimensions associées.
    """
    df = fetch_usda_data(
        year_from=year_from,
        year_to=year_to,
        commodity_desc=commodity_desc,
        statisticcat_desc=statisticcat_desc,
        agg_level_desc=agg_level_desc,
        source_desc=source_desc,
        domain_desc=domain_desc,
        state_alpha=state_alpha,
    )

    if df.empty:
        print("Aucune donnée Quickstats récupérée.")
        return

    # Normalisation de quelques colonnes clés (pas grave si manquantes)
    for col in ["year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    with get_connection() as conn:
        cur = conn.cursor()
        load_time = datetime.utcnow().isoformat()

        # Préparer un mapping week_ending -> date_id si la colonne existe
        date_map = {}
        if "week_ending" in df.columns and df["week_ending"].notna().any():
            date_map = upsert_dim_date_from_week_ending(conn, df["week_ending"])

        inserted = 0

        for _, row in df.iterrows():
            year = int(row["year"]) if not pd.isna(row["year"]) else None
            value = row.get("Value")

            # Champs utilisés pour les dimensions
            commodity_name = row.get("commodity_desc")
            group_desc = row.get("group_desc")
            sector_desc = row.get("sector_desc")

            statistic_name = row.get("statisticcat_desc")
            unit_desc = row.get("unit_desc")
            freq_desc = row.get("freq_desc")
            short_desc = row.get("short_desc")
            class_desc = row.get("class_desc")
            util_practice_desc = row.get("util_practice_desc")
            prodn_practice_desc = row.get("prodn_practice_desc")

            agg_level_val = row.get("agg_level_desc")

            state_alpha_val = row.get("state_alpha")
            state_name_val = row.get("state_name")
            state_fips_code_val = str(row.get("state_fips_code")) if row.get("state_fips_code") is not None else None

            reference_period_desc = row.get("reference_period_desc")

            country_name = row.get("country_name")
            country_code = row.get("country_code")

            domain_desc_val = row.get("domain_desc")
            domaincat_desc_val = row.get("domaincat_desc")
            location_desc_val = row.get("location_desc")
            source_desc_val = row.get("source_desc")

            # 1) Dimensions
            commodity_id = upsert_dim_commodity(conn, commodity_name, group_desc, sector_desc)
            statistic_id = upsert_dim_statistic(
                conn,
                statistic_name,
                unit_desc,
                freq_desc,
                reference_type=None,  # tu peux déduire ça plus tard si besoin
                short_desc=short_desc,
                class_desc=class_desc,
                util_practice_desc=util_practice_desc,
                prodn_practice_desc=prodn_practice_desc,
            )
            agg_level_id = upsert_dim_agg_level(conn, agg_level_val)
            state_id = upsert_dim_state(conn, state_alpha_val, state_name_val, state_fips_code_val)
            country_id = upsert_dim_country(conn, country_name, country_code)
            reference_period_id = upsert_dim_reference_period(conn, reference_period_desc)

            # 2) date_id (si week_ending)
            date_id = None
            if "week_ending" in row and pd.notna(row["week_ending"]):
                dt = pd.to_datetime(row["week_ending"], errors="coerce")
                if pd.notna(dt):
                    date_id = date_map.get(dt)

            # Si aucun ID de dimension critique, on skippe
            if commodity_id is None or statistic_id is None or agg_level_id is None or year is None:
                continue

            # 3) Insertion dans la table de faits
            cur.execute(
                """
                INSERT OR REPLACE INTO fact_ag_stats (
                    year,
                    date_id,
                    commodity_id,
                    statistic_id,
                    agg_level_id,
                    state_id,
                    country_id,
                    reference_period_id,
                    value,
                    domain_desc,
                    domaincat_desc,
                    location_desc,
                    source_desc,
                    load_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    year,
                    date_id,
                    commodity_id,
                    statistic_id,
                    agg_level_id,
                    state_id,
                    country_id,
                    reference_period_id,
                    value,
                    domain_desc_val,
                    domaincat_desc_val,
                    location_desc_val,
                    source_desc_val,
                    load_time,
                ),
            )
            inserted += 1

        print(f"Quickstats chargées dans fact_ag_stats: {inserted} lignes.")


# ---------------------------
# 4. BOUCLE SUR STATISTICS & AGG_LEVELS
# ---------------------------

STATISTICS = ["AREA PLANTED", "AREA HARVESTED", "CONDITION", "YIELD", "PRODUCTION"]
AGG_LEVELS = ["NATIONAL", "STATE"]


def run_quickstats_bulk(
    year_from: int,
    year_to: int,
    commodities: Iterable[str],
    statistics: Iterable[str] = STATISTICS,
    agg_levels: Iterable[str] = AGG_LEVELS,
):
    """
    Lance load_quickstats pour toutes les combinaisons :
    commodities × statistics × agg_levels.
    """
    stats_unique = list(dict.fromkeys(statistics))  # enlève le doublon 'YIELD'

    for commodity in commodities:
        for stat in stats_unique:
            for agg in agg_levels:
                print(f"=== Quickstats: {commodity} | {stat} | {agg} ===")
                load_quickstats(
                    year_from=year_from,
                    year_to=year_to,
                    commodity_desc=commodity,
                    statisticcat_desc=stat,
                    agg_level_desc=agg,
                )
