import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events


# ---------- CONFIG DB ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "usda.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------- HELPERS CACHÉS ----------

@st.cache_data(show_spinner=False)
def get_year_bounds():
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM fact_ag_stats",
            conn,
        )
    return int(df["min_year"][0]), int(df["max_year"][0])


@st.cache_data(show_spinner=False)
def get_commodities():
    query = """
        SELECT DISTINCT commodity_name
        FROM dim_commodity
        ORDER BY commodity_name
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return df["commodity_name"].tolist()


@st.cache_data(show_spinner=False)
def get_states_for_commodity(commodity_name: str):
    query = """
        SELECT DISTINCT st.state_alpha, st.state_name
        FROM fact_ag_stats f
        JOIN dim_commodity c ON f.commodity_id = c.commodity_id
        JOIN dim_state st ON f.state_id = st.state_id
        WHERE c.commodity_name = ?
        ORDER BY st.state_alpha
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(commodity_name,))
    return df


@st.cache_data(show_spinner=False)
def get_by_year(state_alpha: str, metric: str, year_from: int, year_to: int, commodity: str):
    """
    Série temporelle par année pour un état & une métrique.
    On se limite aux enregistrements annuels (period_category = 'YEAR').
    """
    query = """
        SELECT
            f.year,
            f.value AS value
        FROM fact_ag_stats f
        JOIN dim_commodity c ON f.commodity_id = c.commodity_id
        JOIN dim_statistic s ON f.statistic_id = s.statistic_id
        JOIN dim_state st     ON f.state_id = st.state_id
        JOIN dim_reference_period r ON f.reference_period_id = r.reference_period_id
        WHERE c.commodity_name = ?
          AND s.statistic_name = ?
          AND st.state_alpha = ?
          AND f.year BETWEEN ? AND ?
          AND r.period_category = 'YEAR'
        GROUP BY f.year
        ORDER BY f.year
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(commodity, metric, state_alpha, year_from, year_to))
    return df


@st.cache_data(show_spinner=False)
def get_forecast_by_year(state_alpha: str, metric: str, year_from: int, year_to: int, commodity: str):
    """
    Dernier forecast annuel disponible par année (on prend le forecast le plus tardif, via forecast_month).
    """
    query = """
        WITH ranked AS (
            SELECT
                f.year,
                f.value AS value,
                r.forecast_month,
                ROW_NUMBER() OVER (
                    PARTITION BY f.year
                    ORDER BY r.forecast_month DESC
                ) AS rn
            FROM fact_ag_stats f
            JOIN dim_commodity c ON f.commodity_id = c.commodity_id
            JOIN dim_statistic s ON f.statistic_id = s.statistic_id
            JOIN dim_state st     ON f.state_id = st.state_id
            JOIN dim_reference_period r ON f.reference_period_id = r.reference_period_id
            WHERE c.commodity_name = ?
              AND s.statistic_name = ?
              AND st.state_alpha = ?
              AND f.year BETWEEN ? AND ?
              AND r.period_category = 'FORECAST'
        )
        SELECT year, value
        FROM ranked
        WHERE rn = 1
        ORDER BY year
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(commodity, metric, state_alpha, year_from, year_to))
    return df


@st.cache_data(show_spinner=False)
def get_by_state(metric: str, year: int, commodity: str):
    query = """
        SELECT
            st.state_alpha,
            st.state_name,
            MAX(f.value) AS value
        FROM fact_ag_stats f
        JOIN dim_commodity c      ON f.commodity_id = c.commodity_id
        JOIN dim_statistic s      ON f.statistic_id = s.statistic_id
        JOIN dim_state st         ON f.state_id = st.state_id
        JOIN dim_reference_period r ON f.reference_period_id = r.reference_period_id
        JOIN dim_agg_level a      ON f.agg_level_id = a.agg_level_id
        WHERE c.commodity_name = ?
          AND s.statistic_name = ?
          AND f.year = ?
          AND r.period_category = 'YEAR'
          AND a.agg_level_desc = 'STATE'
          AND (f.domain_desc = 'TOTAL' OR f.domain_desc IS NULL)
        GROUP BY st.state_alpha, st.state_name
        ORDER BY st.state_name
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(commodity, metric, year))

    # 🔸 Assure que 'value' est bien numérique
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    return df




@st.cache_data(show_spinner=False)
def get_weekly_condition(state_alpha: str, year: int, commodity: str):
    """
    Séries hebdo de condition de culture pour un État & une année.
    On suppose :
      - statistic_name = 'CONDITION'
      - r.period_category = 'WEEK'
      - les catégories (PCT GOOD, etc.) sont dans dim_statistic.short_desc
    """
    query = """
        SELECT
            d.week_of_year AS week,
            s.short_desc   AS condition_label,
            f.value   AS value
        FROM fact_ag_stats f
        JOIN dim_commodity c      ON f.commodity_id = c.commodity_id
        JOIN dim_statistic s      ON f.statistic_id = s.statistic_id
        JOIN dim_state st         ON f.state_id = st.state_id
        JOIN dim_reference_period r ON f.reference_period_id = r.reference_period_id
        JOIN dim_date d           ON f.date_id = d.date_id
        WHERE c.commodity_name = ?
          AND s.statistic_name = 'CONDITION'
          AND st.state_alpha = ?
          AND f.year = ?
          AND r.period_category = 'WEEK'
        GROUP BY d.week_of_year, s.short_desc
        ORDER BY d.week_of_year
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=[commodity, state_alpha, str(year)])

    out_path = PROJECT_ROOT / "data" / "condition.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    if df.empty:
        return df

    # Pivot pour avoir une colonne par condition (PCT EXCELLENT, etc.)
    pivot = df.pivot_table(
        index="week",
        columns="condition_label",
        values="value",
        aggfunc="mean",
    ).sort_index()

    pivot = pivot.reset_index()
    return pivot


# ---------- UI ----------

st.set_page_config(page_title="USDA Dashboard", layout="wide")
st.title("Quickstat Data Visulizer")

if not DB_PATH.exists():
    st.error(f"Base SQLite introuvable: {DB_PATH}. Lance d'abord la pipeline ETL.")
    st.stop()



min_year, max_year = get_year_bounds()
slider_max_year = max(max_year, 2025)

commodity_options = get_commodities()
default_commodity = "CORN" if "CORN" in commodity_options else commodity_options[0]
default_detail_year = 2025 if min_year <= 2025 <= slider_max_year else min(slider_max_year, max_year)

# ---- Sidebar: filtres ----
# ---- Paramètres sur la page ----
param_col1, param_col2, param_col3 = st.columns([1, 1, 1])

with param_col1:
    commodity_name = st.selectbox(
        "Commodity",
        commodity_options,
        index=commodity_options.index(default_commodity),
    )

    states_df = get_states_for_commodity(commodity_name)
    state_options = states_df["state_alpha"].tolist()
    state_default = "IA" if "IA" in state_options else (state_options[0] if state_options else "")
    st.session_state.setdefault("state_selector", state_default)
    if "pending_state" in st.session_state:
        st.session_state["state_selector"] = st.session_state.pop("pending_state")

    state_alpha = st.selectbox(
        "State",
        state_options,
        index=state_options.index(st.session_state["state_selector"])
        if st.session_state.get("state_selector") in state_options
        else 0,
        key="state_selector",
    )
    st.session_state["selected_state"] = state_alpha

with param_col2:
    metric_labels = {
        "Area Planted": "AREA PLANTED",
        "Area Harvested": "AREA HARVESTED",
        "Yield": "YIELD",
        "Production": "PRODUCTION",
    }
    metric_label = st.selectbox("Metric", list(metric_labels.keys()))
    metric = metric_labels[metric_label]

    year_from, year_to = st.slider(
        "Year range (by Year chart)",
        min_value=min_year,
        max_value=slider_max_year,
        value=(max(min_year, slider_max_year - 30), min(slider_max_year, max_year)),
    )

with param_col3:
    year_for_detail = st.slider(
        "Year (Map & Weekly Condition)",
        min_value=min_year,
        max_value=slider_max_year,
        value=default_detail_year,
    )

# ---------- Layout principal ----------

# ---- Ligne 1 : By Year + Map ----
col1, col2 = st.columns([2, 2])

# -- Graph by Year --
by_year_df = get_by_year(state_alpha, metric, year_from, year_to, commodity_name)
forecast_df = get_forecast_by_year(state_alpha, metric, year_from, year_to, commodity_name)

with col1:
    st.subheader(f"{metric_label} – {state_alpha} (by Year)")
    show_forecast = st.checkbox(
        "Afficher les forecasts USDA (dernière publi par année)",
        value=False,
        key="show_forecast",
    )
    unit_labels = {
        "AREA PLANTED": "Acres",
        "AREA HARVESTED": "Acres",
        "YIELD": "Bushels per Acre",
        "PRODUCTION": "Bushels",
    }
    y_unit = unit_labels.get(metric, "")
    if by_year_df.empty:
        st.info("Aucune donnée disponible pour cette combinaison.")
    else:
        fig_year = go.Figure()
        fig_year.add_trace(
            go.Scatter(
                x=by_year_df["year"],
                y=by_year_df["value"],
                mode="lines+markers",
                name="Historique",
                line=dict(color="#1f77b4"),
            )
        )
        if show_forecast and not forecast_df.empty:
            fig_year.add_trace(
                go.Scatter(
                    x=forecast_df["year"],
                    y=forecast_df["value"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#ff7f0e", dash="dash"),
                    marker=dict(symbol="circle-open"),
                )
            )

        fig_year.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            yaxis_title=f"{metric_label}" + (f" ({y_unit})" if y_unit else ""),
            xaxis_title="Year",
        )
        st.plotly_chart(fig_year, width="stretch")

# -- Map by State --
by_state_df = get_by_state(metric, year_for_detail, commodity_name)
by_state_df = by_state_df[~by_state_df["state_alpha"].isin(["AK", "HI"])]

with col2:
    st.subheader(f"{metric_label} – {year_for_detail} (by State)")
    if by_state_df.empty:
        st.info("Aucune donnée pour cette année / métrique.")
    else:
        values = pd.to_numeric(by_state_df["value"], errors="coerce")
        vmin, vmax = values.quantile([0.05, 0.95])
        if pd.isna(vmin) or pd.isna(vmax):
            vmin, vmax = values.min(), values.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
            vmin, vmax = (0, 1) if pd.isna(vmin) or pd.isna(vmax) else (vmin - 1, vmax + 1)
        values_clipped = values.clip(lower=vmin, upper=vmax)

        fig_map = go.Figure(
            go.Choropleth(
                locations=by_state_df["state_alpha"].tolist(),
                z=values_clipped.tolist(),
                locationmode="USA-states",
                colorscale="Blues",
                zmin=vmin,
                zmax=vmax,
                customdata=by_state_df["state_alpha"].tolist(),
                hovertext=by_state_df["state_name"].tolist(),
                hovertemplate="<b>%{hovertext}</b><br>Value: %{z}<extra></extra>",
                marker_line_color="white",
                marker_line_width=0.5,
                colorbar=dict(title=metric_label),
            )
        )
        fig_map.update_layout(
            geo=dict(
                scope="usa",
                projection_type="albers usa",
                showframe=False,
                showcoastlines=False,
                showcountries=False,
            ),
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            clickmode="event",  # évite le grisé des autres États après clic
            dragmode=False,
        )
        fig_map.update_geos(
            lataxis_range=[24, 50],  # coupe Hawaii/Alaska visuellement
            lonaxis_range=[-125, -66],
        )

        selected_points = plotly_events(
            fig_map,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_width="100%",
            override_height=360,
            key="map_click",
        )

        if selected_points:
            first = selected_points[0]
            clicked_state = (
                first.get("location")
                or first.get("pointData", {}).get("location")
                or first.get("customdata")
                or (
                    by_state_df.iloc[first["pointIndex"]]["state_alpha"]
                    if first.get("pointIndex") is not None and 0 <= first["pointIndex"] < len(by_state_df)
                    else None
                )
            )
            # Evite les reruns en boucle: on ne rerun que si l'événement est nouveau
            event_id = (first.get("curveNumber"), first.get("pointIndex"), clicked_state)
            if (
                clicked_state
                and clicked_state in state_options
                and event_id != st.session_state.get("last_map_event")
            ):
                st.session_state["last_map_event"] = event_id
                # on pousse la sélection dans pending_state pour la prochaine exécution
                st.session_state["pending_state"] = clicked_state
                st.rerun()


# ---- Ligne 2 : Weekly Condition ----
st.subheader(f"Weekly Condition – US – {year_for_detail}")

# Données condition uniquement dispo pour US
weekly_df = get_weekly_condition("US", year_for_detail, commodity_name)

if weekly_df.empty:
    st.info("Pas (encore) de données de condition hebdomadaire pour cette combinaison.")
else:
    # Normalise les noms de colonnes (les champs bruts sont du type
    # 'WHEAT - CONDITION, MEASURED IN PCT GOOD' -> 'PCT GOOD')
    def _simplify(col: str) -> str:
        if col == "week":
            return col
        if "PCT" in col:
            return "PCT " + col.split("PCT", 1)[1].strip().upper()
        return col

    weekly_df = weekly_df.rename(columns={c: _simplify(c) for c in weekly_df.columns})
    # Si plusieurs colonnes deviennent identiques après simplification, on les additionne
    if weekly_df.columns.duplicated().any():
        week_series = weekly_df["week"]
        cond_df = weekly_df.drop(columns=["week"])
        cond_df = cond_df.T.groupby(level=0).sum().T
        weekly_df = pd.concat([week_series, cond_df], axis=1)

    preferred_order = [
        "PCT VERY POOR",
        "PCT POOR",
        "PCT FAIR",
        "PCT GOOD",
        "PCT EXCELLENT",  # ajouté en dernier pour être au sommet de la pile
    ]
    color_map = {
        "PCT VERY POOR": "#d62728",   # rouge
        "PCT POOR": "#ff7f0e",        # orange
        "PCT FAIR": "#1f77b4",        # bleu
        "PCT GOOD": "#8bc34a",        # vert clair
        "PCT EXCELLENT": "#2e7d32",   # vert foncé
    }

    available = [c for c in preferred_order if c in weekly_df.columns]
    if not available:
        st.info("Aucune donnée de condition disponible.")
    else:
        weekly_filled = weekly_df.fillna(0)

        fig_cond = go.Figure()
        for cond in available:  # bottom -> top
            fig_cond.add_trace(
                go.Scatter(
                    x=weekly_filled["week"],
                    y=pd.to_numeric(weekly_filled[cond], errors="coerce").fillna(0),
                    mode="lines",
                    name=cond,
                    stackgroup="one",
                    line=dict(width=1, color=color_map.get(cond)),
                    fillcolor=color_map.get(cond),
                    hovertemplate="<b>%{fullData.name}</b><br>Week %{x}: %{y:.1f}%<extra></extra>",
                )
            )

        fig_cond.update_layout(
            height=350,
            legend_title_text="Condition",
            yaxis_tickformat=".0f",
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_cond, width="stretch")
