import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "usda.db"


METRIC_CATEGORY_DEFS = [
    ("Market & hedging", ["Price", "Futures"]),
    ("Logistics & distribution", ["Transport cost", "Post-harvest operations", "Storage"]),
    ("Operating totals", ["Operating cost", "Total operating cost", "Effective operating cost", "Total cost"]),
    (
        "Inputs & crop care",
        [
            "Seed",
            "Fertilizers & soil amendments",
            "Macronutrients",
            "Micronutrients",
            "Pesticides",
            "Herbicide",
            "Insecticide",
            "Fungicide",
        ],
    ),
    (
        "Field operations & labor",
        [
            "Applications by airplane",
            "Applications with machines",
            "Mechanized operations",
            "Labor",
            "Technical assistance",
        ],
    ),
    ("Land & finance", ["Land leasing", "Land opportunity cost", "Financing", "Opportunity cost"]),
    (
        "Machinery & depreciation",
        [
            "Machines & equipment",
            "Maintenance",
            "Maintenance - machinery & equipment",
            "Depreciation - equipment",
            "Depreciation - implements",
            "Depreciation - machines",
            "Depreciation - utilities",
        ],
    ),
    (
        "Insurance & taxes",
        ["Production insurance", "Insurance - machinery & equipment", "Funrural tax"],
    ),
]

CONAB_METRICS = OrderedDict(
    [
        ("Yield", {"column": "yield", "agg": "mean"}),
        ("Production", {"column": "production", "agg": "sum"}),
        ("Planted area", {"column": "area_planted", "agg": "sum"}),
    ]
)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def get_date_range() -> Tuple[pd.Timestamp, pd.Timestamp]:
    query = """
        SELECT MIN(d.date_iso) AS min_date, MAX(d.date_iso) AS max_date
        FROM fact_br_prices f
        JOIN dim_date d ON f.date_id = d.date_id
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    min_date = pd.to_datetime(df.loc[0, "min_date"]).date()
    max_date = pd.to_datetime(df.loc[0, "max_date"]).date()
    return min_date, max_date


@st.cache_data(show_spinner=False)
def get_all_commodities() -> List[str]:
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT commodity FROM fact_br_prices WHERE commodity IS NOT NULL ORDER BY commodity",
            conn,
        )
    return df.iloc[:, 0].tolist()


@st.cache_data(show_spinner=False)
def get_unit_list() -> List[str]:
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT DISTINCT unit FROM fact_br_prices WHERE unit IS NOT NULL ORDER BY unit",
            conn,
        )
    return df.iloc[:, 0].tolist()


@st.cache_data(show_spinner=False)
def get_metrics_for_commodities(commodities: Sequence[str]) -> List[str]:
    if not commodities:
        return []
    query = """
        SELECT DISTINCT metric
        FROM fact_br_prices
        WHERE metric IS NOT NULL
    """
    params: List[str] = []
    if commodities:
        query += _build_in_clause("commodity", commodities, params)
    query += " ORDER BY metric"

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)
    return df.iloc[:, 0].tolist()


@st.cache_data(show_spinner=False)
def get_states_for_filters(commodities: Sequence[str], metrics: Sequence[str]) -> List[str]:
    if not commodities or not metrics:
        return []
    query = """
        SELECT DISTINCT state
        FROM fact_br_prices
        WHERE state IS NOT NULL
    """
    params: List[str] = []
    if commodities:
        query += _build_in_clause("commodity", commodities, params)
    if metrics:
        query += _build_in_clause("metric", metrics, params)
    query += " ORDER BY state"

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)
    return df.iloc[:, 0].tolist()


@st.cache_data(show_spinner=False)
def get_conab_filters() -> dict:
    queries = {
        "states": "SELECT DISTINCT state FROM fact_conab_cotton WHERE state IS NOT NULL ORDER BY state",
        "commodities": "SELECT DISTINCT commodity FROM fact_conab_cotton WHERE commodity IS NOT NULL ORDER BY commodity",
    }

    results = {}
    with get_connection() as conn:
        for key, query in queries.items():
            df = pd.read_sql(query, conn)
            results[key] = df.iloc[:, 0].dropna().tolist()
    return results


def _build_in_clause(column: str, values: Sequence[str], params: List[str]) -> str:
    placeholders = ",".join("?" for _ in values)
    params.extend(values)
    return f" AND {column} IN ({placeholders})"


def build_metric_categories(all_metrics: Sequence[str]) -> "OrderedDict[str, List[str]]":
    categories: "OrderedDict[str, List[str]]" = OrderedDict()
    used = set()

    for label, metric_list in METRIC_CATEGORY_DEFS:
        available = [m for m in metric_list if m in all_metrics]
        if available:
            categories[label] = available
            used.update(available)

    remaining = sorted(set(all_metrics) - used)
    if remaining:
        categories["Other"] = remaining

    ordered = OrderedDict()
    ordered["All metrics"] = sorted(all_metrics)
    for label, values in categories.items():
        ordered[label] = values
    return ordered


def ensure_selection(key: str, available: Sequence[str]) -> List[str]:
    available_list = list(available)
    current = st.session_state.get(key, [])
    if current:
        current = [item for item in current if item in available_list]
    st.session_state[key] = current
    return current


@st.cache_data(show_spinner=False)
def load_conab_records(states: Sequence[str], commodities: Sequence[str]) -> pd.DataFrame:
    base_query = """
        SELECT
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
        FROM fact_conab_cotton
        WHERE 1=1
    """
    params: List[str] = []
    if states:
        base_query += _build_in_clause("state", states, params)
    if commodities:
        base_query += _build_in_clause("commodity", commodities, params)
    base_query += " ORDER BY harvest_year, state, commodity, month"

    with get_connection() as conn:
        df = pd.read_sql(base_query, conn, params=params or None)
    return df


def build_conab_period(df: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(df["harvest_year"], errors="coerce")
    months = pd.to_numeric(df["month"], errors="coerce")
    months = months.fillna(1).clip(1, 12)

    period = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    valid = years.notna()
    if not valid.any():
        return period

    period_strings = (
        years.loc[valid].astype(int).astype(str)
        + "-"
        + months.loc[valid].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    period.loc[valid] = pd.to_datetime(period_strings, errors="coerce")
    return period


@st.cache_data(show_spinner=False)
def get_macroregions_for_filters(
    states: Sequence[str], commodities: Sequence[str], metrics: Sequence[str]
) -> List[str]:
    if not states:
        return []
    query = """
        SELECT DISTINCT macroregion
        FROM fact_br_prices
        WHERE macroregion IS NOT NULL
    """
    params: List[str] = []
    if states:
        query += _build_in_clause("state", states, params)
    if commodities:
        query += _build_in_clause("commodity", commodities, params)
    if metrics:
        query += _build_in_clause("metric", metrics, params)
    query += " ORDER BY macroregion"

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)
    return df["macroregion"].dropna().tolist()


@st.cache_data(show_spinner=False)
def get_cities_for_filters(
    states: Sequence[str],
    macroregions: Sequence[str],
    commodities: Sequence[str],
    metrics: Sequence[str],
) -> List[str]:
    if not states:
        return []
    query = """
        SELECT DISTINCT city
        FROM fact_br_prices
        WHERE city IS NOT NULL
    """
    params: List[str] = []
    if states:
        query += _build_in_clause("state", states, params)
    if macroregions:
        query += _build_in_clause("macroregion", macroregions, params)
    if commodities:
        query += _build_in_clause("commodity", commodities, params)
    if metrics:
        query += _build_in_clause("metric", metrics, params)
    query += " ORDER BY city"

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=params or None)
    return df["city"].dropna().tolist()


@st.cache_data(show_spinner=False)
def load_br_prices(
    start_date: str,
    end_date: str,
    states: Sequence[str],
    macroregions: Sequence[str],
    commodities: Sequence[str],
    cities: Sequence[str],
    units: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    base_query = """
        SELECT
            d.date_iso AS date,
            f.state,
            f.macroregion,
            f.city,
            f.destination_city,
            f.commodity,
            f.metric,
            f.unit,
            f.product_item,
            f.price_value,
            f.source
        FROM fact_br_prices f
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE d.date_iso BETWEEN ? AND ?
    """
    params: List[str] = [start_date, end_date]
    if states:
        base_query += _build_in_clause("f.state", states, params)
    if macroregions:
        base_query += _build_in_clause("f.macroregion", macroregions, params)
    if commodities:
        base_query += _build_in_clause("f.commodity", commodities, params)
    if cities:
        base_query += _build_in_clause("f.city", cities, params)
    if units:
        base_query += _build_in_clause("f.unit", units, params)
    if metrics:
        base_query += _build_in_clause("f.metric", metrics, params)

    base_query += " ORDER BY d.date_iso, f.state, f.city, f.commodity"

    with get_connection() as conn:
        df = pd.read_sql(base_query, conn, params=params)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["price_value"] = pd.to_numeric(df["price_value"], errors="coerce")
    df = df.dropna(subset=["price_value"])
    return df


def main():
    st.set_page_config(page_title="BR Prices Explorer", layout="wide")
    st.title("Brazil price explorer")


    st.sidebar.markdown(
        """
        <style>
            section[data-testid="stSidebar"] > div:first-child,
            div[data-testid="stSidebar"] > div:nth-child(2) {
                padding-top: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    min_date, max_date = get_date_range()
    all_commodities = get_all_commodities()
    unit_selection = get_unit_list()

    metric_selection: List[str] = []
    metric_options: List[str] = []
    with st.sidebar:
        st.header("Filters")
        date_range = st.date_input(
            "Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        commodity_key = "br_commodity_filter"
        ensure_selection(commodity_key, all_commodities)
        commodity_selection = st.multiselect(
            "Commodities",
            options=all_commodities,
            default=st.session_state[commodity_key],
            key=commodity_key,
        )
        metric_options = get_metrics_for_commodities(tuple(commodity_selection))
        st.caption(f"Units locked: {len(unit_selection)} distinct values present.")
        if metric_options:
            metric_categories = build_metric_categories(metric_options)
            metric_category = st.selectbox("Metric category", list(metric_categories.keys()))
            metric_values = metric_categories.get(metric_category, [])
            metric_key = "metric_multiselect"
            ensure_selection(metric_key, metric_values)
            metric_selection = st.multiselect(
                "Metrics",
                options=metric_values,
                default=st.session_state[metric_key],
                key=metric_key,
            )
        else:
            st.session_state["metric_multiselect"] = []
            st.info("Select at least one commodity to see the available metrics.")
            metric_selection = []

        available_states = get_states_for_filters(tuple(commodity_selection), tuple(metric_selection))
        state_key = "br_state_filter"
        ensure_selection(state_key, available_states)
        state_selection = st.multiselect(
            "States",
            options=available_states,
            default=st.session_state[state_key],
            key=state_key,
        )

        available_macro = get_macroregions_for_filters(
            tuple(state_selection), tuple(commodity_selection), tuple(metric_selection)
        )
        macro_key = "br_macro_filter"
        ensure_selection(macro_key, available_macro)
        macro_selection = st.multiselect(
            "Macro-regions",
            options=available_macro,
            default=st.session_state[macro_key],
            key=macro_key,
        )

        available_cities = get_cities_for_filters(
            tuple(state_selection),
            tuple(macro_selection),
            tuple(commodity_selection),
            tuple(metric_selection),
        )
        city_key = "br_city_filter"
        ensure_selection(city_key, available_cities)
        city_selection = st.multiselect(
            "Cities",
            options=available_cities,
            default=st.session_state[city_key],
            key=city_key,
        )

    start_date = None
    end_date = None
    date_valid = True
    if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
        st.error("Please select a valid date range.")
        date_valid = False
    else:
        start_date, end_date = date_range
        if start_date > end_date:
            st.error("Start date must be before end date.")
            date_valid = False

    if commodity_selection and metric_selection and not available_states:
        st.warning("No states available for this commodity/metric combination.")

    price_filters_ready = bool(commodity_selection and metric_selection and state_selection)
    if not commodity_selection:
        st.info("Select at least one commodity to load the BR price data.")
    elif not metric_selection:
        st.info("Select at least one metric for the selected commodities.")
    elif not state_selection:
        st.info("Select at least one state to build the chart.")

    price_df: Optional[pd.DataFrame] = None
    if date_valid and price_filters_ready and start_date and end_date:
        df = load_br_prices(
            start_date.isoformat(),
            end_date.isoformat(),
            state_selection,
            macro_selection,
            commodity_selection,
            city_selection,
            unit_selection,
            metric_selection,
        )
        if df.empty:
            st.warning("No records found for the selected filters.")
        else:
            price_df = df

    if price_df is not None:
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Observations", f"{len(price_df):,}".replace(",", " "))
        col2.metric("Average price (BRL)", f"{price_df['price_value'].mean():.2f}")
        col3.metric("Latest date", price_df["date"].max().date().isoformat())

        grouping_options = {
            "All data (average)": None,
            "By state": "state",
            "By city": "city",
            "By commodity": "commodity",
            "By unit": "unit",
        }
        group_key = st.selectbox("Line coloring", list(grouping_options.keys()))
        color_col = grouping_options[group_key]

        plot_df = price_df.copy()
        if color_col:
            plot_df = (
                plot_df.groupby(["date", color_col], as_index=False)["price_value"]
                .mean()
            )
        else:
            plot_df = (
                plot_df.groupby("date", as_index=False)["price_value"]
                .mean()
            )

        hover_fields = [color_col] if color_col else None
        fig = px.line(
            plot_df.sort_values("date"),
            x="date",
            y="price_value",
            color=color_col,
            hover_data=hover_fields,
        )
        fig.update_layout(
            height=500,
            yaxis_title="Price",
            xaxis_title="Date",
            legend_title=group_key if color_col else "",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detailed records")
        st.dataframe(
            price_df.sort_values(["date", "state", "city", "commodity"]),
            use_container_width=True,
            height=500,
        )
        csv_bytes = price_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv_bytes, file_name="br_prices_filtered.csv")

    st.divider()
    st.header("CONAB production explorer")
    conab_filters = get_conab_filters()
    conab_states = conab_filters.get("states", [])
    conab_commodities = conab_filters.get("commodities", [])
    if not conab_states or not conab_commodities:
        st.info("CONAB data is not available in fact_conab_cotton.")
        return

    conab_state_key = "conab_state_filter"
    ensure_selection(conab_state_key, conab_states)
    conab_state_selection = st.multiselect(
        "States (CONAB)",
        options=conab_states,
        default=st.session_state[conab_state_key],
        key=conab_state_key,
    )

    conab_commodity_key = "conab_commodity_filter"
    ensure_selection(conab_commodity_key, conab_commodities)
    conab_commodity_selection = st.multiselect(
        "Commodities (CONAB)",
        options=conab_commodities,
        default=st.session_state[conab_commodity_key],
        key=conab_commodity_key,
    )

    metric_label = st.selectbox("Metric", list(CONAB_METRICS.keys()), key="conab_metric_choice")
    metric_config = CONAB_METRICS[metric_label]
    metric_column = metric_config["column"]
    agg_func = metric_config["agg"]

    if not conab_state_selection or not conab_commodity_selection:
        st.info("Select at least one CONAB state and commodity to visualize this data.")
        return

    conab_df = load_conab_records(tuple(conab_state_selection), tuple(conab_commodity_selection))
    if conab_df.empty:
        st.warning("No CONAB records found for the selected state/commodity filters.")
        return

    conab_df[metric_column] = pd.to_numeric(conab_df[metric_column], errors="coerce")
    metric_df = conab_df.dropna(subset=[metric_column]).copy()
    if metric_df.empty:
        st.warning(f"No {metric_label.lower()} values available for this selection.")
        return

    metric_df["period_date"] = build_conab_period(metric_df)
    chart_df = metric_df.dropna(subset=["period_date"])
    if chart_df.empty:
        st.warning("Unable to build a time series because no valid harvest years were found.")
    else:
        grouping_options = {
            "All data (aggregate)": None,
            "By state": "state",
            "By commodity": "commodity",
            "By safra": "safra",
        }
        grouping_label = st.selectbox("Series grouping (CONAB)", list(grouping_options.keys()))
        color_col = grouping_options[grouping_label]

        group_cols = ["period_date"]
        if color_col:
            group_cols.append(color_col)
        plot_df = (
            chart_df.groupby(group_cols, as_index=False)[metric_column]
            .agg(agg_func)
            .rename(columns={metric_column: "value"})
        )
        fig = px.line(
            plot_df.sort_values("period_date"),
            x="period_date",
            y="value",
            color=color_col,
            labels={"period_date": "Harvest period", "value": metric_label},
        )
        fig.update_layout(height=450, yaxis_title=metric_label, legend_title=grouping_label if color_col else "")
        st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "crop_year",
        "harvest_year",
        "month",
        "safra",
        "state",
        "commodity",
        metric_column,
        "source",
    ]
    available_cols = [col for col in display_cols if col in conab_df.columns]
    st.dataframe(
        conab_df[available_cols].sort_values(["harvest_year", "month", "state", "commodity"]),
        use_container_width=True,
        height=400,
    )


if __name__ == "__main__":
    main()
