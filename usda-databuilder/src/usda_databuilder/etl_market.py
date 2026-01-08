import re
import pandas as pd
from datetime import datetime
from .db import get_connection

def parse_fractional_price(s: str) -> float:
    """
    Convertit une chaîne du type '4 11/32' ou '4  5/16' en float.
    """
    if not isinstance(s, str):
        return None

    s = s.strip()
    
    # Cas simple : juste un entier
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    
    # Format '4 11/32'
    m = re.match(r"(\d+)\s+(\d+)/(\d+)", s)
    if m:
        base = float(m.group(1))
        num = float(m.group(2))
        den = float(m.group(3))
        return base + num / den
    
    # Format '11/32'
    m = re.match(r"(\d+)/(\d+)", s)
    if m:
        num = float(m.group(1))
        den = float(m.group(2))
        return num / den
    
    return None



FASTMARKET_CODES = {
    "AG-CRN-0089": ("FOB_DEC", "Cash FOB Decatur"),
    "AG-CRN-0090": ("BASIS_DEC", "Basis Decatur"),
    "AG-CRN-0074": ("BARGE_GULF", "Barge Premium Gulf"),
    "AG-CRN-0075": ("FOB_GULF", "FOB Gulf"),
    "AG-CRN-0073": ("CIF_GULF", "CIF Gulf"),
}


def load_fastmarkets(csv_path: str):
    df = pd.read_csv(csv_path)
    # Certaines lignes (ex: avis) contiennent du texte plutôt qu'une date -> on les ignore
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[df["Date"].notna()].copy()

    inserted = 0

    with get_connection() as conn:
        cur = conn.cursor()

        for _, row in df.iterrows():
            date_iso = row["Date"].date().isoformat()

            # Trouver/Créer date_id
            cur.execute(
                "SELECT date_id FROM dim_date WHERE date_iso = ?",
                (date_iso,)
            )
            r = cur.fetchone()
            if r:
                date_id = r[0]
            else:
                # on crée une date
                dt = row["Date"]
                cur.execute("""
                    INSERT INTO dim_date (date_iso, year, month, day, week_of_year, quarter, is_week_end)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (
                    date_iso, dt.year, dt.month, dt.day, int(dt.strftime("%W"))+1, (dt.month-1)//3+1
                ))
                date_id = cur.lastrowid

            # Insérer chaque instrument
            for code, (short, desc) in FASTMARKET_CODES.items():
                price = pd.to_numeric(row[code], errors="coerce")
                if pd.isna(price):
                    continue

                # Insérer instrument si pas existant
                cur.execute(
                    "SELECT instrument_id FROM dim_market_instrument WHERE code = ?",
                    (code,)
                )
                r = cur.fetchone()
                if r:
                    inst_id = r[0]
                else:
                    cur.execute("""
                        INSERT INTO dim_market_instrument (code, description, instrument_type)
                        VALUES (?, ?, 'CASH')
                    """, (code, desc))
                    inst_id = cur.lastrowid

                # Insérer prix
                cur.execute("""
                    INSERT OR REPLACE INTO fact_market_price
                        (date_id, instrument_id, price, currency, unit)
                    VALUES (?, ?, ?, 'USD', NULL)
                """, (date_id, inst_id, float(price)))

                inserted += 1

    print(f"[FASTMARKETS] {inserted} lignes chargées.")

def load_futures(csv_path: str):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    inserted = 0
    with get_connection() as conn:
        cur = conn.cursor()

        # Instrument unique futures
        code = "ZC_CONT"
        cur.execute(
            "SELECT instrument_id FROM dim_market_instrument WHERE code = ?",
            (code,)
        )
        r = cur.fetchone()
        if r:
            inst_id = r[0]
        else:
            cur.execute("""
                INSERT INTO dim_market_instrument (code, description, instrument_type)
                VALUES (?, ?, 'FUTURES')
            """, (code, "Corn Futures Continuous"))
            inst_id = cur.lastrowid

        for _, row in df.iterrows():
            date_iso = row["Date"].date().isoformat()
            close_raw = row["Close"]
            close_dec = parse_fractional_price(close_raw)

            if close_dec is None:
                continue

            # date_id
            cur.execute(
                "SELECT date_id FROM dim_date WHERE date_iso = ?",
                (date_iso,)
            )
            r = cur.fetchone()
            if r:
                date_id = r[0]
            else:
                dt = row["Date"]
                cur.execute("""
                    INSERT INTO dim_date (date_iso, year, month, day, week_of_year, quarter, is_week_end)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (
                    date_iso, dt.year, dt.month, dt.day, int(dt.strftime("%W"))+1, (dt.month-1)//3+1
                ))
                date_id = cur.lastrowid

            cur.execute("""
                INSERT OR REPLACE INTO fact_futures_price
                    (date_id, instrument_id, close_price, raw_close)
                VALUES (?, ?, ?, ?)
            """, (date_id, inst_id, close_dec, close_raw))

            inserted += 1

    print(f"[FUTURES] {inserted} lignes chargées.")
