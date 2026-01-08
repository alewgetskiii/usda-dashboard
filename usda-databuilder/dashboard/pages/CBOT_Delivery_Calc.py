from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "usda.db"
RATES_PATH = PROJECT_ROOT / "data" / "rates.csv"


def get_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def get_latest_effr() -> tuple[str | None, float | None]:
    if not RATES_PATH.exists():
        return None, None
    df = pd.read_csv(RATES_PATH)
    df = df[df["Rate Type"] == "EFFR"].copy()
    if df.empty:
        return None, None
    df["Effective Date"] = pd.to_datetime(df["Effective Date"], errors="coerce")
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")
    df = df.dropna(subset=["Effective Date", "Rate"])
    if df.empty:
        return None, None
    latest = df.sort_values("Effective Date", ascending=False).iloc[0]
    return latest["Effective Date"].date().isoformat(), float(latest["Rate"])


@st.cache_data(show_spinner=False)
def get_locations() -> list[str]:
    query = """
        SELECT DISTINCT l.location_name
        FROM fact_barge_rate f
        JOIN dim_transport_location l ON f.transport_location_id = l.transport_location_id
        ORDER BY l.location_name
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return df["location_name"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_dates_for_location(location: str) -> list[str]:
    query = """
        SELECT DISTINCT d.date_iso
        FROM fact_barge_rate f
        JOIN dim_transport_location l ON f.transport_location_id = l.transport_location_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE l.location_name = ?
        ORDER BY d.date_iso DESC
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(location,))
    return df["date_iso"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_barge_rate(location: str, date_iso: str) -> float | None:
    query = """
        SELECT f.rate_percent_of_tariff
        FROM fact_barge_rate f
        JOIN dim_transport_location l ON f.transport_location_id = l.transport_location_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE l.location_name = ? AND d.date_iso = ?
        LIMIT 1
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(location, date_iso))
    if df.empty:
        return None
    return df.loc[0, "rate_percent_of_tariff"]


def build_table(
    zones: list[str],
    delivery_diff: list[float],
    loadout_charge: float,
    misc_cost: float,
    barge_rate_pct: float,
    tariff: list[float],
    cash_bid: list[float],
    storage_days: float,
    storage_rate: float,
    futures_price: float,
    interest_rate: float,
) -> pd.DataFrame:
    delivery_value = [d + loadout_charge + misc_cost for d in delivery_diff]
    barge_freight = [t * barge_rate_pct / 100 for t in tariff]
    delivered_value = [dv + bf for dv, bf in zip(delivery_value, barge_freight)]
    cash_vs_delivery = [c - dv for c, dv in zip(cash_bid, delivered_value)]
    storage_cost = storage_days * storage_rate
    interest_cost = futures_price * interest_rate * storage_days / 365
    ttl_carry = storage_cost + interest_cost
    cash_vs_delivery_carry = [c - (dv + ttl_carry) for c, dv in zip(cash_bid, delivered_value)]

    rows = {
        "Delivery Differential": delivery_diff,
        "Del. Loadout Charge": [loadout_charge] * len(zones),
        "Misc. Costs": [misc_cost] * len(zones),
        "Delivery Value": delivery_value,
        "Barge Rate (%)": [barge_rate_pct] * len(zones),
        "Tariff (ct/bu)": tariff,
        "Barge Freight (ct/bu)": barge_freight,
        "Delivered Value": delivered_value,
        "Cash Bid": cash_bid,
        "Cash vs. Delivery": cash_vs_delivery,
        "# of Days in Storage": [storage_days] * len(zones),
        "Storage Cost": [storage_cost] * len(zones),
        "Futures Price": [futures_price] * len(zones),
        "Interest Cost": [interest_cost] * len(zones),
        "TTL Est. Carry Cost": [ttl_carry] * len(zones),
        "Cash vs. Delivery (incl carry)": cash_vs_delivery_carry,
    }

    df = pd.DataFrame(rows, index=zones).T
    return df


st.set_page_config(page_title="Barge Delivery Calculator", layout="wide")
st.title("CBOT Delivery Calculations")

locations = get_locations()
if not locations:
    st.warning("Aucune donnée barge disponible dans la base.")
    st.stop()

with st.expander("Parameters", expanded=False):
    location = st.selectbox("Barge location (from DB)", locations, index=0)
    dates = get_dates_for_location(location)
    date_iso = st.selectbox("Date", dates, index=0) if dates else None

    rate_from_db = get_barge_rate(location, date_iso) if date_iso else None
    default_rate = float(rate_from_db) if rate_from_db is not None else 0.0

    st.caption(f"DB barge rate: {default_rate:.2f}% (location={location}, date={date_iso})")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        barge_rate_pct = st.number_input("Barge Rate (%)", value=default_rate, format="%.3f")
        loadout_charge = st.number_input("Del. Loadout Charge (ct/bu)", value=6.0, format="%.2f")
        misc_cost = st.number_input("Misc. Costs (ct/bu)", value=0.0, format="%.2f")
        storage_days = st.number_input("# of Days in Storage", value=31.0, format="%.1f")
    with col_right:
        storage_rate = st.number_input("Storage Rate (ct/bu/day)", value=0.265, format="%.3f")
        effr_date, effr_rate = get_latest_effr()
        if effr_rate is None:
            default_interest = 6.0
            st.caption("EFFR introuvable dans data/rates.csv, défaut = 6.00%")
        else:
            default_interest = effr_rate + 2.0
            st.caption(f"EFFR {effr_rate:.2f}% ({effr_date}) + 2.00% = {default_interest:.2f}%")
        interest_rate = st.number_input("Interest Rate (annual %)", value=default_interest, format="%.2f") / 100.0
        futures_price = st.number_input("Futures Price (ct/bu)", value=445.0, format="%.2f")

zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
tariff = [16.18, 14.67, 14.20, 13.47]
delivery_diff = [0.0, 2.0, 2.5, 3.0]

cash_bid_value = st.number_input("Cash Bid (ct/bu)", value=78.0, format="%.2f")
cash_bid = [cash_bid_value] * len(zones)

table = build_table(
    zones=zones,
    delivery_diff=delivery_diff,
    loadout_charge=loadout_charge,
    misc_cost=misc_cost,
    barge_rate_pct=barge_rate_pct,
    tariff=tariff,
    cash_bid=cash_bid,
    storage_days=storage_days,
    storage_rate=storage_rate,
    futures_price=futures_price,
    interest_rate=interest_rate,
)

table_display = table.copy().round(2)

bold_rows = {
    "Barge Rate (%)",
    "Barge Freight (ct/bu)",
    "Cash Bid",
    "# of Days in Storage",
    "Futures Price",
}
highlight_rows = {"Cash vs. Delivery"}
thick_top = {"Barge Rate (%)", "Futures Price"}
thick_bottom = {"Barge Rate (%)", "Futures Price", "Barge Freight (ct/bu)"}

def style_row(row: pd.Series):
    styles = []
    for _ in row:
        s = ""
        if row.name in bold_rows:
            s += "font-weight: 700;"
        if row.name in highlight_rows:
            s += "background-color: #ffe59a;"
        if row.name in thick_top:
            s += "border-top: 2px solid #222;"
        if row.name in thick_bottom:
            s += "border-bottom: 2px solid #222;"
        styles.append(s)
    return styles

styled = (
    table_display.style
    .apply(style_row, axis=1)
    .format("{:.2f}")
    .set_table_attributes('style="border-collapse:collapse;"')
)

st.markdown(styled.to_html(), unsafe_allow_html=True)
