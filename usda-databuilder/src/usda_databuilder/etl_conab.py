import sqlite3
from pathlib import Path
from backup_db import backup

import pandas as pd


# ---- chemins à adapter ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # si tu le mets dans src/usda_databuilder/
DB_PATH       = PROJECT_ROOT / "data" / "usda.db"

CSV_RECENT = PROJECT_ROOT / "BR" / "data_raw" / "CONAB" /"production.csv"
CSV_HIST   = PROJECT_ROOT / "BR" / "data_raw" / "CONAB" /"histo_data.csv"


def parse_crop_year(y):
    if pd.isna(y):
        return None, None
    s = str(y).strip()
    if "/" not in s:
        # ex: '2023' ou vide
        try:
            hy = int(s)
        except ValueError:
            hy = None
        return s, hy

    left, right = s.split("/", 1)
    right = right.strip()
    try:
        if len(right) == 2:
            r = int(right)
            if r < 40:
                hy = 2000 + r
            else:
                hy = 1900 + r
        else:
            hy = int(right)
    except ValueError:
        hy = None
    return s, hy


def load_recent(csv_path: Path, default_crop_year=None):
    df = pd.read_csv(csv_path)

    # standardise les noms au cas où
    df.columns = [c.strip() for c in df.columns]

    # on suppose qu'il a déjà: safra, state, Commodity, month, area_planted, production, yield
    df["source"] = csv_path.name

    if default_crop_year is not None:
        cy, hy = parse_crop_year(default_crop_year)
        df["crop_year"] = cy
        df["harvest_year"] = hy
    else:
        df["crop_year"] = None
        df["harvest_year"] = None

    # assure que 'month' est bien numérique (ou NaN)
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
    else:
        df["month"] = None

    df = df.rename(
        columns={
            "safra": "safra",
            "state": "state",
            "Commodity": "commodity",
            "area_planted": "area_planted",
            "production": "production",
            "yield": "yield",
        }
    )

    keep_cols = [
        "crop_year",
        "harvest_year",
        "safra",
        "state",
        "commodity",
        "month",
        "area_planted",
        "production",
        "yield",
        "source",
    ]
    return df[keep_cols]


def load_hist(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    df["source"] = csv_path.name

    # parse 'year' -> crop_year + harvest_year
    crop_years = []
    harvest_years = []
    for val in df["year"]:
        cy, hy = parse_crop_year(val)
        crop_years.append(cy)
        harvest_years.append(hy)
    df["crop_year"] = crop_years
    df["harvest_year"] = harvest_years

    df["month"] = None  # données annuelles

    df = df.rename(
        columns={
            "safra": "safra",
            "state": "state",
            "Commodity": "commodity",
            "area_planted": "area_planted",
            "production": "production",
            "yield": "yield",
        }
    )

    keep_cols = [
        "crop_year",
        "harvest_year",
        "safra",
        "state",
        "commodity",
        "month",
        "area_planted",
        "production",
        "yield",
        "source",
    ]
    return df[keep_cols]


def create_table_if_needed(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_conab_cotton (
            id            INTEGER PRIMARY KEY,
            crop_year     TEXT,
            harvest_year  INTEGER,
            safra         TEXT,
            state         TEXT,
            commodity     TEXT,
            month         INTEGER,
            area_planted  REAL,
            production    REAL,
            yield         REAL,
            source        TEXT,
            UNIQUE(crop_year, state, commodity, month)
        );
        """
    )
    conn.commit()



def load_into_db(df: pd.DataFrame, conn: sqlite3.Connection):
    """
    Insère les lignes dans fact_conab_cotton en respectant la contrainte UNIQUE
    (crop_year, state, commodity, month) grâce à INSERT OR IGNORE.
    """
    # On convertit le DataFrame en liste de dicts
    records = df.to_dict(orient="records")

    sql = """
        INSERT OR IGNORE INTO fact_conab_cotton (
            crop_year,
            harvest_year,
            safra,
            state,
            commodity,
            month,
            area_planted,
            production,
            yield,
            source
        )
        VALUES (
            :crop_year,
            :harvest_year,
            :safra,
            :state,
            :commodity,
            :month,
            :area_planted,
            :production,
            :yield,
            :source
        );
    """

    cur = conn.cursor()
    cur.executemany(sql, records)
    conn.commit()


def main():
    df_recent = load_recent(CSV_RECENT, default_crop_year="2023/24")  # ajuste si besoin
    df_hist   = load_hist(CSV_HIST)

    df_all = pd.concat([df_recent, df_hist], ignore_index=True)
    backup()

    conn = sqlite3.connect(DB_PATH)
    create_table_if_needed(conn)

    load_into_db(df_all, conn)

    conn.close()
    print(f"Inséré {len(df_all)} lignes dans fact_conab_cotton.")


if __name__ == "__main__":
    main()
