from .schema import init_db
from .db import get_connection
from .etl_interior_prices import load_interior_prices
from .etl_barges import load_barge_rates
from .etl_quickstats import run_quickstats_bulk
from .config import DEFAULT_YEAR_FROM, DEFAULT_YEAR_TO, DEFAULT_COMMODITIES
from .etl_market import load_fastmarkets, load_futures
from .etl_br_prices import load_br_prices

TRACKED_TABLES = [
    "fact_interior_price_spread",
    "fact_barge_rate",
    "fact_ag_stats",
    "fact_br_prices",
]


def get_table_counts(tables=None):
    """Retourne le nombre de lignes par table pour détecter les nouveaux enregistrements."""
    tables = tables or TRACKED_TABLES
    counts = {}
    with get_connection() as conn:
        cur = conn.cursor()
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
    return counts


def log_changes(before_counts, after_counts):
    print("\nRésumé des changements (nouvelles lignes éventuelles) :")
    for table in TRACKED_TABLES:
        before = before_counts.get(table, 0)
        after = after_counts.get(table, 0)
        delta = after - before
        status = (
            f"+{delta} nouvelles lignes" if delta > 0 else "aucune nouvelle ligne"
        )
        print(f"- {table}: {status} (avant={before}, après={after})")


def run_full_pipeline():
    print("Initialisation du schéma SQLite...")
    init_db()

    before_counts = get_table_counts()

    print("Chargement des données de fast market...")
    load_fastmarkets("data/fastmarket.csv")

    print("Chargement des données de Cc1...")
    load_futures("data/Cc1.csv")

    print("Chargement des données de transport (Interior Prices)...")
    load_interior_prices()

    print("Chargement des données de transport (Barge Rates)...")
    load_barge_rates()

    print("Chargement des données Quickstats (AREA PLANTED / HARVESTED / CONDITION / YIELD / PRODUCTION)...")
    run_quickstats_bulk(
        year_from=DEFAULT_YEAR_FROM,
        year_to=DEFAULT_YEAR_TO,
        commodities=DEFAULT_COMMODITIES,
    )

    print("Chargement des prix BR (IMEA + SIMA)...")
    load_br_prices()

    after_counts = get_table_counts()
    log_changes(before_counts, after_counts)


if __name__ == "__main__":
    run_full_pipeline()
