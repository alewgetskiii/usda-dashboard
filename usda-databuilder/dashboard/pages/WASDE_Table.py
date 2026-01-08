from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WASDE_extracted_US_corn_EST_wide.csv"


@st.cache_data(show_spinner=False)
def load_wasde_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ReleaseDateTime"] = pd.to_datetime(df["ReleaseDateTime"], errors="coerce")
    df = df.dropna(subset=["ReleaseDateTime"])
    return df


def parse_column_meta(columns: list[str]):
    meta = []
    for col in columns:
        if col in {"ForecastYear", "ForecastMonth", "ReleaseDateTime", "MarketYear"}:
            continue
        parts = col.rsplit("_", 2)
        if len(parts) != 3:
            continue
        metric, commodity, region = parts
        meta.append({"col": col, "metric": metric, "commodity": commodity, "region": region})
    return meta


def market_year_key(value: str):
    if isinstance(value, str) and "/" in value:
        head = value.split("/")[0]
        try:
            return int(head)
        except ValueError:
            return value
    try:
        return int(value)
    except Exception:
        return value


def format_value(value: float | int | None, is_percent: bool) -> str:
    if value is None or pd.isna(value):
        return ""
    if is_percent:
        return f"{value:.1f}%"
    if abs(value - round(value)) < 0.005:
        return f"{int(round(value))}"
    return f"{value:.2f}"


st.set_page_config(page_title="WASDE S&D Table", layout="wide")
st.title("WASDE S&D Table")

if not DATA_PATH.exists():
    st.error(f"Missing data file: {DATA_PATH}")
    st.stop()

df = load_wasde_data(DATA_PATH)
meta = parse_column_meta(df.columns.tolist())

commodities = sorted({m["commodity"] for m in meta})
regions = sorted({m["region"] for m in meta})

if not commodities:
    st.warning("No commodity columns detected in the dataset.")
    st.stop()

selected_commodity = st.selectbox("Commodity", commodities, index=0)
selected_region = st.selectbox("Region", regions, index=0)

df_filtered = df[
    df["ReleaseDateTime"].notna()
].copy()

market_years_all = sorted(df_filtered["MarketYear"].dropna().unique(), key=market_year_key)
start_year = st.selectbox("Start Market Year", market_years_all, index=0)
market_years = [y for y in market_years_all if market_year_key(y) >= market_year_key(start_year)]

latest_releases = (
    df_filtered["ReleaseDateTime"]
    .dropna()
    .sort_values(ascending=False)
    .unique()
    .tolist()
)
latest_releases = latest_releases[:3]

def get_metric_labels(commodity: str) -> list[tuple[str, str]]:
    if commodity == "Corn":
        return [
            ("Planted", "Planted"),
            ("Harvested", "Harvested"),
            ("Yield", "Yield"),
            ("CarryIn", "Carry In"),
            ("Production", "Production"),
            ("Imports", "Imports"),
            ("TotalSupply", "Total Supply"),
            ("FSI", "FSI"),
            ("FSI_Ethanol", "Ethanol"),
            ("FSI_Food", "Food"),
            ("FeedResidual", "Feed & Residual"),
            ("Exports", "Exports"),
            ("TotalUse", "Total Use"),
            ("CarryOut", "Carry Out"),
            ("StocksToUse", "STU"),
        ]
    if commodity == "Soybean":
        return [
            ("Planted", "Planted"),
            ("Harvested", "Harvested"),
            ("Yield", "Yield"),
            ("CarryIn", "Carry In"),
            ("Production", "Production"),
            ("Imports", "Imports"),
            ("TotalSupply", "Total Supply"),
            ("Crushings", "Crushings"),
            ("Exports", "Exports"),
            ("Seed", "Seed"),
            ("Residual", "Residual"),
            ("TotalUse", "Total Use"),
            ("CarryOut", "Carry Out"),
            ("StocksToUse", "STU"),
        ]
    if commodity == "Wheat":
        return [
            ("Planted", "Planted"),
            ("Harvested", "Harvested"),
            ("Yield", "Yield"),
            ("CarryIn", "Carry In"),
            ("Production", "Production"),
            ("Imports", "Imports"),
            ("TotalSupply", "Total Supply"),
            ("Food", "Food"),
            ("Seed", "Seed"),
            ("FeedResidual", "Feed & Residual"),
            ("DomesticTotal", "Domestic, Total"),
            ("Exports", "Exports"),
            ("TotalUse", "Total Use"),
            ("CarryOut", "Carry Out"),
            ("StocksToUse", "STU"),
        ]
    return []

metric_to_label = get_metric_labels(selected_commodity)

col_map = {
    m["metric"]: m["col"]
    for m in meta
    if m["commodity"] == selected_commodity and m["region"] == selected_region
}

final_df = (
    df_filtered.sort_values("ReleaseDateTime")
    .groupby("MarketYear", as_index=False)
    .tail(1)
)

latest_market_year = market_years[-1] if market_years else None
columns = [str(my) for my in market_years]
for rel in latest_releases:
    label = f"Est {pd.to_datetime(rel).strftime('%Y-%m-%d')}"
    columns.append(label)

table = pd.DataFrame(
    index=[label for _, label in metric_to_label],
    columns=columns,
    dtype=object,
)

for metric, label in metric_to_label:
    col = col_map.get(metric)
    if not col:
        continue
    is_percent = label == "STU"

    if col in final_df.columns:
        final_series = final_df.set_index("MarketYear")[col]
        for year in market_years:
            table.loc[label, str(year)] = format_value(final_series.get(year), is_percent)

    if latest_market_year is not None:
        for rel in latest_releases:
            rel_label = f"Est {pd.to_datetime(rel).strftime('%Y-%m-%d')}"
            rel_df = df_filtered[df_filtered["ReleaseDateTime"] == rel]
            if col not in rel_df.columns:
                continue
            rel_series = rel_df.set_index("MarketYear")[col]
            table.loc[label, rel_label] = format_value(
                rel_series.get(latest_market_year),
                is_percent,
            )

highlight_rows = {"Total Supply", "Total Use", "Carry Out"}

def style_rows(row: pd.Series):
    if row.name in highlight_rows:
        return ["background-color: #f3c7c7; font-weight: 600"] * len(row)
    return [""] * len(row)

styled_table = table.style.apply(style_rows, axis=1)

st.caption(f"{selected_commodity} – {selected_region.replace('UnitedStates', 'United States')}")
st.dataframe(styled_table, use_container_width=True)
