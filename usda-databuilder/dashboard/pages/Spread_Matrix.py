from __future__ import annotations

from datetime import date
from calendar import monthrange

import pandas as pd
import streamlit as st


MONTH_CODES = {
    "F": 1,
    "H": 3,
    "K": 5,
    "N": 7,
    "U": 9,
    "X": 11,
    "Z": 12,
}

DELIVERY_WINDOWS = {
    "H": (3, 4),   # Mar -> Apr
    "K": (5, 6),   # May -> Jun
    "N": (7, 8),   # Jul -> Aug
    "U": (9, 11),  # Sep -> Nov
    "Z": (12, 2),  # Dec -> Feb
}


def delivery_window(month_code: str, year: int) -> tuple[date, date, int]:
    start_month, end_month = DELIVERY_WINDOWS[month_code]
    start_year = year
    end_year = year + 1 if end_month < start_month else year
    start_dt = date(start_year, start_month, 1)
    end_day = monthrange(end_year, end_month)[1]
    end_dt = date(end_year, end_month, end_day)
    total_days = (end_dt - start_dt).days + 1
    return start_dt, end_dt, total_days


def build_contracts(root: str, base_year: int) -> list[dict]:
    month_codes = ["H", "K", "N", "U", "Z", "H"]
    years = [base_year, base_year, base_year, base_year, base_year, base_year + 1]
    contracts = []
    for code, yr in zip(month_codes, years):
        yy = str(yr)[-2:]
        ticker = f"{root}{code}{yy}.CBT"
        contracts.append({"code": code, "year": yr, "ticker": ticker})
    return contracts


@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list[str]) -> dict[str, float | None]:
    try:
        import yfinance as yf
    except ImportError:
        st.warning("yfinance n'est pas installé. Installe-le: pip install yfinance")
        return {t: None for t in tickers}

    data = yf.download(tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=False)
    prices = {}
    for t in tickers:
        if t in data.columns.get_level_values(0):
            close = data[t]["Close"].dropna()
            prices[t] = float(close.iloc[-1]) if not close.empty else None
        elif "Close" in data:
            close = data["Close"].dropna()
            prices[t] = float(close.iloc[-1]) if not close.empty else None
        else:
            prices[t] = None
    return prices


@st.cache_data(show_spinner=False)
def fetch_history(tickers: list[str]) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        st.warning("yfinance n'est pas installé. Installe-le: pip install yfinance")
        return pd.DataFrame()
    return yf.download(tickers, period="max", interval="1d", group_by="ticker", auto_adjust=False)


def extract_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.get_level_values(0):
        return data[ticker]["Close"]
    if "Close" in data.columns:
        return data["Close"]
    return pd.Series(dtype=float)


st.set_page_config(page_title="Spread Matrix", layout="wide")
st.title("Spread Matrix (Full Carry)")

product = st.selectbox(
    "Produit",
    ["Corn (CBOT)", "Soybeans (CBOT)", "Wheat (CBOT)", "KC HRW (KCBT)"],
    index=0,
)

root_map = {
    "Corn (CBOT)": "ZC",
    "Soybeans (CBOT)": "ZS",
    "Wheat (CBOT)": "ZW",
    "KC HRW (KCBT)": "KE",
}
root = root_map[product]

current_year = date.today().year
base_year = st.number_input("Base year (H contract)", min_value=2000, max_value=2100, value=current_year)

storage_rate = st.number_input("Storage rate (cents/bu/day)", value=0.265, format="%.3f")
interest_rate = st.number_input("Interest rate (annual %)", value=6.0, format="%.2f") / 100.0

contracts = build_contracts(root, int(base_year))
tickers = [c["ticker"] for c in contracts]

prices = fetch_prices(tickers)
price_map = {t: prices.get(t) for t in tickers}

columns = []
for c in contracts:
    start_dt, end_dt, total_days = delivery_window(c["code"], c["year"])
    columns.append(
        {
            "label": f"{c['code']}{str(c['year'])[-2:]}",
            "start": start_dt,
            "end": end_dt,
            "days": total_days,
            "price": price_map.get(c["ticker"]),
        }
    )

table = pd.DataFrame(columns=[c["label"] for c in columns])

table.loc["Starts"] = [c["start"].strftime("%d-%b") for c in columns]
table.loc["Ends"] = [c["end"].strftime("%d-%b") for c in columns]
table.loc["Total Days"] = [c["days"] for c in columns]
closing_label = f"{root} Closing Price"
table.loc[closing_label] = [c["price"] for c in columns]
table.loc["Storage rate"] = [storage_rate for _ in columns]
table.loc["Total Storage"] = [round(c["days"] * storage_rate, 2) for c in columns]
table.loc["Total Interest"] = [
    round((c["price"] or 0) * interest_rate * c["days"] / 365, 2) for c in columns
]
table.loc["Full Carry"] = [
    round(table.loc["Total Storage", c["label"]] + table.loc["Total Interest", c["label"]], 2)
    for c in columns
]

spread_close = []
full_carry = []
for idx, c in enumerate(columns):
    if idx >= len(columns) - 1:
        spread_close.append(None)
        full_carry.append(None)
        continue
    p0 = c["price"]
    p1 = columns[idx + 1]["price"]
    spread_close.append(round((p0 or 0) - (p1 or 0), 2))
    full_carry.append(table.loc["Full Carry", c["label"]])

table.loc["Spread Close"] = spread_close
table.loc["% Full Carry"] = [
    round((-(sc) / fc) * 100, 0) if sc is not None and fc not in (None, 0) else None
    for sc, fc in zip(spread_close, full_carry)
]

table_display = table.copy().astype(str)
highlight_rows = {closing_label, "% Full Carry"}

def highlight_row(row: pd.Series):
    if row.name in highlight_rows:
        return ["background-color: #ffe59a; font-weight: 600"] * len(row)
    return [""] * len(row)

spread_cols = [c["label"] for c in columns[:-1]]
default_spread = spread_cols[0] if spread_cols else None

styled_table = table_display.style.apply(highlight_row, axis=1)

col_table, col_chart = st.columns([2.4, 1.6])
with col_table:
    st.markdown(styled_table.to_html(), unsafe_allow_html=True)
    selected_spreads = st.multiselect(
        "Spreads",
        spread_cols,
        default=[default_spread] if default_spread else [],
        label_visibility="collapsed",
    )

if not selected_spreads:
    st.stop()

history = fetch_history(tickers)
series_map = {}
for spread in selected_spreads:
    selected_idx = spread_cols.index(spread)
    near_ticker = contracts[selected_idx]["ticker"]
    far_ticker = contracts[selected_idx + 1]["ticker"]
    days = columns[selected_idx]["days"]

    near_series = extract_close_series(history, near_ticker)
    far_series = extract_close_series(history, far_ticker)
    hist = pd.DataFrame({"near": near_series, "far": far_series}).dropna()
    if hist.empty:
        continue
    hist["spread"] = hist["near"] - hist["far"]
    hist["full_carry"] = (storage_rate * days) + (hist["near"] * interest_rate * days / 365)
    series_map[spread] = -(hist["spread"] / hist["full_carry"]) * 100
plot_df = pd.DataFrame(series_map).dropna(how="all")

with col_chart:
    st.subheader("% Full Carry – Historique")
    if plot_df.empty:
        st.info("Pas de données historiques disponibles pour ces spreads.")
    else:
        st.line_chart(plot_df, height=380)
