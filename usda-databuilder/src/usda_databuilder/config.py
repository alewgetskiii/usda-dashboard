import os
from pathlib import Path

# Emplacement de la base SQLite
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "usda.db"

# Config APIs
USDA_API_KEY = os.getenv("USDA_API_KEY", "D1ABF2AD-362D-346E-A641-93A2FA6ED6D8")
USDA_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET/"

SOCRATA_DOMAIN = "agtransport.usda.gov"
SOCRATA_APP_TOKEN = "GQzRJWK559Du5SqStCOavIUsw"  # déjà dans ton exemple

# Paramètres par défaut pour Quickstats
DEFAULT_YEAR_FROM = 1980
DEFAULT_YEAR_TO = 2025

DEFAULT_COMMODITIES = ["CORN", "SOYBEANS", "WHEAT", "COTTON"]
DEFAULT_STATISTICS = ["AREA PLANTED", "AREA HARVESTED", "CONDITION", "YIELD", "PRODUCTION"]
DEFAULT_AGG_LEVELS = ["NATIONAL", "STATE"]


def get_usda_api_key() -> str:
    """
    Resolve the USDA API key.

    Prioritizes the environment variable USDA_API_KEY to avoid hardcoding secrets.
    Raises a clear error if no key is provided.
    """
    key = os.getenv("USDA_API_KEY") or USDA_API_KEY
    if not key or key == "A_REMPLACER":
        raise ValueError(
            "Clé USDA absente. Exporte USDA_API_KEY ou mets-la dans src/usda_databuilder/config.py."
        )
    return key
