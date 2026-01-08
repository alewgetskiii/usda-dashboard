import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events


# ---------- CONFIG DB ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "usda.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ---------- HELPERS ----------

@st.cache_data(show_spinner=False)
def get_barge_last_date():
    """
    Retourne la dernière date disponible pour fact_barge_rate (date_iso) + année correspondante.
    """
    query = """
        SELECT MAX(d.date_iso) AS max_date
        FROM fact_barge_rate f
        JOIN dim_date d ON f.date_id = d.date_id
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    if df.empty or df["max_date"][0] is None:
        return None, None

    max_date = pd.to_datetime(df["max_date"][0]).date()
    return max_date, max_date.year


@st.cache_data(show_spinner=False)
def get_barge_locations():
    """
    Retourne la liste des locations disponibles, filtrée sur les 7 qu'on veut afficher.
    """
    desired = [
        "Twin Cities",
        "Lower Ohio",
        "Cincinnati",
        "Cairo-Memphis",
        "St. Louis",
        "Lower Illinois",
        "Mid-Mississippi",
    ]

    query = """
        SELECT DISTINCT location_name
        FROM dim_transport_location
        WHERE location_name IN ({})
        ORDER BY location_name
    """.format(
        ",".join("?" for _ in desired)
    )

    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=desired)

    # On garde l'ordre de desired
    locs = [loc for loc in desired if loc in df["location_name"].tolist()]
    return locs


@st.cache_data(show_spinner=False)
def get_latest_delta_by_location():
    """
    Dernier rate par location + delta vs semaine précédente (rate - prev_rate).
    """
    query = """
        WITH ordered AS (
            SELECT
                t.location_name,
                d.date_iso,
                f.rate_percent_of_tariff AS rate,
                LAG(f.rate_percent_of_tariff) OVER (
                    PARTITION BY t.location_name
                    ORDER BY d.date_iso
                ) AS prev_rate,
                ROW_NUMBER() OVER (
                    PARTITION BY t.location_name
                    ORDER BY d.date_iso DESC
                ) AS rn
            FROM fact_barge_rate f
            JOIN dim_date d ON f.date_id = d.date_id
            JOIN dim_transport_location t ON f.transport_location_id = t.transport_location_id
        )
        SELECT
            location_name,
            date_iso,
            rate,
            prev_rate,
            rate - prev_rate AS delta
        FROM ordered
        WHERE rn = 1
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)
    return df


@st.cache_data(show_spinner=False)
def get_barge_timeseries(location_name: str):
    """
    Série temporelle complète (date_iso, rate) pour une location donnée.
    On joindra sur dim_date pour avoir la vraie date.
    """
    query = """
        SELECT
            d.date_iso,
            f.year,
            f.week_usda,
            f.rate_percent_of_tariff AS rate
        FROM fact_barge_rate f
        JOIN dim_transport_location t ON f.transport_location_id = t.transport_location_id
        JOIN dim_date d ON f.date_id = d.date_id
        WHERE t.location_name = ?
        ORDER BY d.date_iso
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=(location_name,))

    if df.empty:
        return df

    df["date_iso"] = pd.to_datetime(df["date_iso"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["rate"])
    return df


def get_barge_last_year(df: pd.DataFrame, last_date: datetime.date):
    """
    Filtre les données sur la dernière année (365 jours) à partir de last_date.
    """
    if df.empty:
        return df

    cutoff = pd.Timestamp(last_date) - pd.Timedelta(days=365)
    return df[df["date_iso"] >= cutoff].copy()


def compute_barge_stats(df: pd.DataFrame):
    """
    À partir d'une série temporelle (date_iso, year, week_usda, rate),
    calcule :
      - last_rate
      - prev_rate
      - delta_week
      - ytd_avg
      - ytd_prev_avg (année précédente)
      - delta_ytd
    """
    if df.empty:
        return None

    df_sorted = df.sort_values("date_iso").reset_index(drop=True)
    last_row = df_sorted.iloc[-1]
    last_rate = float(last_row["rate"])
    last_year = int(last_row["year"])
    last_week = int(last_row["week_usda"]) if not pd.isna(last_row["week_usda"]) else None

    prev_rate = None
    delta_week = None
    if len(df_sorted) >= 2:
        prev_row = df_sorted.iloc[-2]
        prev_rate = float(prev_row["rate"])
        delta_week = last_rate - prev_rate

    # YTD pour l'année en cours (jusqu'à la semaine actuelle)
    cur_mask = df_sorted["year"] == last_year
    if last_week is not None:
        cur_mask &= df_sorted["week_usda"] <= last_week
    df_cur_ytd = df_sorted[cur_mask]

    ytd_avg = float(df_cur_ytd["rate"].mean()) if not df_cur_ytd.empty else None

    # YTD pour l'année précédente (jusqu'à la même semaine)
    prev_year = last_year - 1
    prev_mask = df_sorted["year"] == prev_year
    if last_week is not None:
        prev_mask &= df_sorted["week_usda"] <= last_week
    df_prev_ytd = df_sorted[prev_mask]

    ytd_prev_avg = float(df_prev_ytd["rate"].mean()) if not df_prev_ytd.empty else None
    delta_ytd = None
    if ytd_avg is not None and ytd_prev_avg is not None:
        delta_ytd = ytd_avg - ytd_prev_avg

    return {
        "last_rate": last_rate,
        "prev_rate": prev_rate,
        "delta_week": delta_week,
        "ytd_avg": ytd_avg,
        "ytd_prev_avg": ytd_prev_avg,
        "delta_ytd": delta_ytd,
        "last_year": last_year,
        "last_week": last_week,
        "last_date": last_row["date_iso"].date(),
    }


# coordonnées approx pour les locations
BARGE_COORDS = {
    "Twin Cities": (44.95, -93.10),
    "Lower Ohio": (37.0, -89.2),
    "Cincinnati": (39.1, -84.5),
    "Cairo-Memphis": (36.0, -90.0),
    "St. Louis": (38.63, -90.20),
    "Lower Illinois": (39.0, -90.4),
    "Mid-Mississippi": (41.0, -91.0),
}


def build_barge_map(selected_location: str, locations: list, values: dict):
    """
    Carte des barges en scatter_geo (fond simple), couleur = delta hebdo.
    """
    lats, lons, texts, zs = [], [], [], []
    for loc in locations:
        if loc not in BARGE_COORDS:
            continue
        lat, lon = BARGE_COORDS[loc]
        lats.append(lat)
        lons.append(lon)
        texts.append(loc)
        zs.append(values.get(loc))

    if lats and lons:
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        lat_pad = 1.5
        lon_pad = 2.5
        geos_range = {
            "lataxis_range": [lat_min - lat_pad, lat_max + lat_pad],
            "lonaxis_range": [lon_min - lon_pad, lon_max + lon_pad],
            "center": {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2},
        }
    else:
        geos_range = {
            "lataxis_range": [24, 50],
            "lonaxis_range": [-125, -66],
            "center": {"lat": 38, "lon": -90},
        }

    non_null = [v for v in zs if v is not None]
    if len(non_null) >= 2:
        numeric = pd.Series(non_null, dtype=float)
        vmin, vmax = numeric.min(), numeric.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    else:
        vmin, vmax = -1, 1

    fig = go.Figure()
    rivers = [
        {"lon": [-93, -90, -89, -90, -91, -90, -90, -90], "lat": [47, 45, 43, 41, 38.6, 36, 33, 30]},
        {"lon": [-84.5, -85.5, -87, -88.5, -89.1], "lat": [39.1, 38.7, 38.3, 37.8, 37]},
        {"lon": [-90.4, -90.2, -90], "lat": [40.7, 39.5, 38.6]},
    ]
    for r in rivers:
        fig.add_trace(
            go.Scattergeo(
                lon=r["lon"],
                lat=r["lat"],
                mode="lines",
                line=dict(color="#91a6ff", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            text=texts,
            customdata=texts,
            mode="markers+text",
            textposition="top center",
            textfont=dict(color="#000000", size=11),
            marker=dict(
                size=14,
                color=zs,
                colorscale=[[0, "#d32f2f"], [0.5, "#f0f0f0"], [1, "#2e7d32"]],
                cmin=vmin,
                cmax=vmax,
                showscale=False,
                line=dict(width=1, color="white"),
            ),
            hovertemplate="<b>%{text}</b><br>Δ vs prev week: %{marker.color:.1f} pts<extra></extra>",
        )
    )

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showframe=False,
            showcoastlines=False,
            showcountries=False,
            **geos_range,
        ),
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode=False,
        clickmode="event",
        showlegend=False,
    )
    return fig


# ---------- UI BARGE PAGE ----------

st.title("Barge Rates – Weekly View")

if not DB_PATH.exists():
    st.error(f"Base SQLite introuvable: {DB_PATH}. Lance d'abord la pipeline ETL.")
    st.stop()

last_date, last_year = get_barge_last_date()
if last_date is None:
    st.info("Aucune donnée de barge rate disponible.")
    st.stop()

locations = get_barge_locations()
if not locations:
    st.info("Aucun emplacement de barge trouvé dans la base.")
    st.stop()

# état de sélection de la location
default_loc = "St. Louis" if "St. Louis" in locations else locations[0]
st.session_state.setdefault("selected_barge_location", default_loc)
st.session_state.setdefault("barge_location_select", default_loc)
# Consomme la valeur cliquée avant de créer le selectbox
if "pending_barge_location" in st.session_state:
    pending = st.session_state.pop("pending_barge_location")
    if pending in locations:
        st.session_state["selected_barge_location"] = pending
        st.session_state["barge_location_select"] = pending

# sidebar : select + info
with st.sidebar:
    st.subheader("Barge Filters")
    selected_loc = st.selectbox(
        "Location",
        locations,
        index=locations.index(st.session_state["barge_location_select"])
        if st.session_state["barge_location_select"] in locations
        else 0,
        key="barge_location_select",
    )
    st.session_state["selected_barge_location"] = selected_loc

if st.button("Retour au dashboard Quickstats"):
    try:
        st.switch_page("app.py")
    except Exception:
        st.warning("Impossible de changer de page automatiquement (lance app.py depuis la vue multipage).")


# on charge la série pour la location sélectionnée
ts_full = get_barge_timeseries(st.session_state["selected_barge_location"])
ts_last_year = get_barge_last_year(ts_full, last_date)
stats = compute_barge_stats(ts_full) if not ts_full.empty else None

# ---------- Row 1 : Carte + KPIs ----------
col_map, col_kpi = st.columns([2, 1])

with col_map:
    st.subheader("Barge locations (clickable)")

    # récupérer le last_rate pour chaque location pour colorer les points
    last_values = {}
    for loc in locations:
        df_loc = get_barge_timeseries(loc)
        if df_loc.empty:
            last_values[loc] = None
        else:
            last_values[loc] = float(df_loc.sort_values("date_iso")["rate"].iloc[-1])

    latest_delta = get_latest_delta_by_location()
    delta_map = {row["location_name"]: row["delta"] for _, row in latest_delta.iterrows()}
    fig_map = build_barge_map(st.session_state["selected_barge_location"], locations, delta_map)

    selected_points = plotly_events(
        fig_map,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=360,
        override_width="100%",
        key="barge_map_click",
    )

    if selected_points:
        first = selected_points[0]
        clicked_loc = first.get("customdata") or first.get("text")
        if isinstance(clicked_loc, list):
            # plotly_events peut renvoyer customdata sous forme de liste
            clicked_loc = clicked_loc[0] if clicked_loc else None
        if not clicked_loc and first.get("pointIndex") is not None:
            idx = first.get("pointIndex")
            if 0 <= idx < len(locations):
                clicked_loc = locations[idx]
        if clicked_loc in locations and clicked_loc != st.session_state.get("selected_barge_location"):
            st.session_state["pending_barge_location"] = clicked_loc
            st.rerun()

with col_kpi:
    st.subheader("Current stats")

    if not stats:
        st.info("Pas de statistiques disponibles pour cette location.")
    else:
        last_rate = stats["last_rate"]
        delta_week = stats["delta_week"]
        ytd_avg = stats["ytd_avg"]
        delta_ytd = stats["delta_ytd"]

        st.metric(
            label=f"Last rate ({stats['last_date'].isoformat()})",
            value=f"{last_rate:.1f} %",
            delta=f"{delta_week:+.1f} pts" if delta_week is not None else "N/A",
        )

        st.metric(
            label=f"YTD {stats['last_year']}",
            value=f"{ytd_avg:.1f} %",
            delta=(
                f"{delta_ytd:+.1f} pts vs YTD {stats['last_year']-1}"
                if delta_ytd is not None
                else "N/A"
            ),
        )

# ---------- Row 2 : Timeseries last year ----------
st.subheader(f"Weekly barge rate – {st.session_state['selected_barge_location']} (last 12 months)")

if ts_last_year.empty:
    st.info("Pas de données sur la dernière année pour cette location.")
else:
    fig_ts = go.Figure(
        go.Scatter(
            x=ts_last_year["date_iso"],
            y=ts_last_year["rate"],
            mode="lines+markers",
            name="Rate",
            line=dict(color="#1f77b4"),
        )
    )
    fig_ts.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Rate (% of tariff)",
    )
    st.plotly_chart(fig_ts, width="stretch")
