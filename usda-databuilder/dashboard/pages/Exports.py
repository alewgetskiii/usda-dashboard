import duckdb
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HS2_KEY_LABELS = {
    "02": "Meat",
    "09": "Coffee/Tea/Spices",
    "10": "Cereals",
    "12": "Oil seeds",
    "17": "Sugars",
    "18": "Cocoa",
    "26": "Ores",
    "27": "Mineral fuels",
}


# ---------- CONFIG DB ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMEX_DB_PATH = PROJECT_ROOT / "data" / "comex.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(COMEX_DB_PATH))


# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def get_available_years() -> list[int]:
    query = "SELECT DISTINCT year FROM fact_exports ORDER BY year DESC"
    with get_connection() as conn:
        df = conn.execute(query).df()
    return df["year"].astype(int).tolist()


@st.cache_data(show_spinner=False)
def get_available_commodities() -> list[str]:
    query = """
        SELECT DISTINCT p.NO_SH4_ING AS commodity
        FROM fact_exports f
        JOIN dim_product p ON substr(f.co_ncm, 1, 6) = p.CO_SH6
        WHERE p.NO_SH4_ING IS NOT NULL
        ORDER BY commodity
    """
    with get_connection() as conn:
        df = conn.execute(query).df()
    return df["commodity"].dropna().tolist()


@st.cache_data(show_spinner=False)
def load_exports(year: int, commodity: str, value_field: str) -> pd.DataFrame:
    """
    Monthly exports for a given year and SH4 commodity.
    """
    query = """
        SELECT
            f.year,
            f.month,
            f.date,
            p.NO_SH4_ING AS commodity,
            c.NO_PAIS_ING AS destination,
            SUM(f.{value_field}) AS total_value
        FROM fact_exports f
        JOIN dim_product p ON substr(f.co_ncm, 1, 6) = p.CO_SH6
        JOIN dim_country c ON f.co_pais = c.CO_PAIS
        WHERE f.year = ? AND p.NO_SH4_ING = ?
        GROUP BY 1,2,3,4,5
        ORDER BY f.month, destination
    """.format(
        value_field=value_field
    )
    with get_connection() as conn:
        df = conn.execute(query, [year, commodity]).df()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    df = df.dropna(subset=["total_value"])
    df["month_start"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def plot_exports(df: pd.DataFrame, year: int, commodity: str, top_n: int, exclude_china: bool, value_label: str):
    if df.empty:
        st.info("No data for this year / commodity.")
        return

    if exclude_china:
        df = df[df["destination"] != "China"]

    top_dest = (
        df.groupby("destination")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    df = df[df["destination"].isin(top_dest)]

    if df.empty:
        st.info("No destination left after filtering (Top N / exclusions).")
        return

    pivot = (
        df.groupby(["month_start", "destination"], as_index=False)["total_value"]
        .sum()
        .pivot(index="month_start", columns="destination", values="total_value")
        .fillna(0)
        .sort_index()
    )

    fig = go.Figure()
    for dest in pivot.columns:
        fig.add_bar(
            x=pivot.index,
            y=pivot[dest],
            name=dest,
            marker_line_width=0.4,
            marker_line_color="black",
        )

    totals = pivot.sum(axis=1)
    fig.add_scatter(
        x=pivot.index,
        y=totals,
        mode="text",
        text=[f"{val/1e6:.1f}M" for val in totals],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        textfont=dict(size=10),
    )

    fig.update_layout(
        barmode="stack",
        title=f"Monthly {commodity} exports – {value_label} by destination ({year})",
        xaxis_title="Month",
        yaxis_title=value_label,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title=None),
        margin=dict(l=20, r=20, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_hs2_exports(year: int, value_field: str, codes: tuple[str, ...] | None = None) -> pd.DataFrame:
    """
    Monthly exports aggregated by HS2 family (NO_SH2_ING).
    """
    query = """
        SELECT
            f.year,
            f.month,
            p.CO_SH2 AS hs2_code,
            p.NO_SH2_ING AS hs2_name,
            SUM(f.{value_field}) AS total_value
        FROM fact_exports f
        JOIN dim_product p ON substr(f.co_ncm, 1, 6) = p.CO_SH6
        WHERE f.year = ?
        GROUP BY 1,2,3,4
        ORDER BY f.month, hs2_code
    """.format(
        value_field=value_field
    )
    with get_connection() as conn:
        df = conn.execute(query, [year]).df()

    if df.empty:
        return df

    if codes:
        df = df[df["hs2_code"].isin(set(codes))].copy()

    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")
    df = df.dropna(subset=["total_value"])
    df["month_start"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def plot_hs2(df: pd.DataFrame, year: int, top_n: int, use_key_labels: bool, value_label: str):
    if df.empty:
        st.info("No data available to plot HS2 families.")
        return

    top_codes = (
        df.groupby("hs2_code")["total_value"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    df = df[df["hs2_code"].isin(top_codes)]

    if df.empty:
        st.info("No HS2 family left after filtering (Top N).")
        return

    def _make_label(row: pd.Series) -> str:
        base_name = HS2_KEY_LABELS.get(row["hs2_code"]) if use_key_labels else row["hs2_name"]
        if base_name is None or pd.isna(base_name):
            base_name = row["hs2_code"]
        return f"{row['hs2_code']} – {base_name}"

    df["label"] = df.apply(_make_label, axis=1)

    pivot = (
        df.groupby(["month_start", "label"], as_index=False)["total_value"]
        .sum()
        .pivot(index="month_start", columns="label", values="total_value")
        .fillna(0)
        .sort_index()
    )

    fig = go.Figure()
    for label in pivot.columns:
        fig.add_bar(
            x=pivot.index,
            y=pivot[label],
            name=label,
            marker_line_width=0.4,
            marker_line_color="black",
        )

    fig.update_layout(
        barmode="stack",
        title=f"Monthly exports – HS2 families ({year})",
        xaxis_title="Month",
        yaxis_title=value_label,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title=None),
        margin=dict(l=20, r=20, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------- UI ----------
def main():
    st.title("Exports by destination (Comex)")
    st.subheader("Main Export Destinations")

    years = get_available_years()
    commodities = get_available_commodities()

    if not years or not commodities:
        st.warning("No export data available in comex.duckdb.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.selectbox("Year", years, index=0)
    with col2:
        commodity = st.selectbox(
            "Commodity (SH4)", commodities, index=commodities.index("Maize (corn)") if "Maize (corn)" in commodities else 0
        )
    with col3:
        top_n = st.slider("Destinations (Top N)", min_value=3, max_value=15, value=8, step=1)

    measure = st.radio("Measure", ["FOB value (USD)", "Quantity"], horizontal=True)
    value_field = "vl_fob" if "FOB" in measure else "qt_estat"
    value_label = "FOB value (USD)" if "FOB" in measure else "Quantity"

    exclude_china = st.checkbox("Exclude China", value=True)

    df = load_exports(year, commodity, value_field)
    plot_exports(df, year, commodity, top_n, exclude_china, value_label)

    st.markdown("---")
    st.subheader("Main Exported Product Families")

    hs2_scope = st.radio(
        "HS2 selection",
        ["All HS2 codes", "Key HS2 only"],
        horizontal=True,
    )
    hs2_top = st.slider("HS2 families to show (Top N)", min_value=3, max_value=15, value=8, step=1)

    key_codes = tuple(HS2_KEY_LABELS.keys()) if hs2_scope == "Key HS2 only" else None
    hs2_df = load_hs2_exports(year, value_field, key_codes)
    plot_hs2(hs2_df, year, hs2_top, use_key_labels=hs2_scope == "Key HS2 only", value_label=value_label)


if __name__ == "__main__":
    main()
