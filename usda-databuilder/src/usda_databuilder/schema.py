from .db import execute_script


SCHEMA_SQL = r"""
-- === DIMENSIONS ===

CREATE TABLE IF NOT EXISTS dim_date (
    date_id        INTEGER PRIMARY KEY,
    date_iso       TEXT NOT NULL UNIQUE,
    year           INTEGER NOT NULL,
    month          INTEGER NOT NULL,
    day            INTEGER NOT NULL,
    week_of_year   INTEGER,
    quarter        INTEGER,
    is_week_end    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS dim_reference_period (
    reference_period_id   INTEGER PRIMARY KEY,
    reference_period_desc TEXT NOT NULL UNIQUE,
    period_category       TEXT NOT NULL,
    week_number           INTEGER,
    forecast_month        INTEGER,
    is_forecast           INTEGER DEFAULT 0,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS dim_state (
    state_id        INTEGER PRIMARY KEY,
    state_alpha     TEXT NOT NULL UNIQUE,
    state_name      TEXT NOT NULL,
    state_fips_code TEXT
);

CREATE TABLE IF NOT EXISTS dim_country (
    country_id    INTEGER PRIMARY KEY,
    country_code  INTEGER NOT NULL UNIQUE,
    country_name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_commodity (
    commodity_id   INTEGER PRIMARY KEY,
    commodity_name TEXT NOT NULL UNIQUE,
    group_desc     TEXT,
    sector_desc    TEXT
);

CREATE TABLE IF NOT EXISTS dim_statistic (
    statistic_id        INTEGER PRIMARY KEY,
    statistic_name      TEXT NOT NULL,
    unit_desc           TEXT NOT NULL,
    freq_desc           TEXT,
    reference_type      TEXT,
    short_desc          TEXT,
    class_desc          TEXT,
    util_practice_desc  TEXT,
    prodn_practice_desc TEXT
);

CREATE TABLE IF NOT EXISTS dim_agg_level (
    agg_level_id   INTEGER PRIMARY KEY,
    agg_level_desc TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS dim_transport_location (
    transport_location_id INTEGER PRIMARY KEY,
    location_name         TEXT NOT NULL UNIQUE,
    river_system          TEXT,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS dim_transport_route (
    route_id         INTEGER PRIMARY KEY,
    commodityod      TEXT NOT NULL UNIQUE,
    commodity_id     INTEGER,
    origin_desc      TEXT,
    destination_desc TEXT,
    FOREIGN KEY (commodity_id) REFERENCES dim_commodity (commodity_id)
);

CREATE TABLE IF NOT EXISTS dim_rail_route (
    rail_route_id          INTEGER PRIMARY KEY,
    origin_location_id     INTEGER NOT NULL,
    destination_location_id INTEGER NOT NULL,
    origin_desc            TEXT,
    destination_desc       TEXT,
    route_mileage          REAL,
    UNIQUE (
        origin_location_id, destination_location_id,
        origin_desc, destination_desc, route_mileage
    ),
    FOREIGN KEY (origin_location_id) REFERENCES dim_transport_location (transport_location_id),
    FOREIGN KEY (destination_location_id) REFERENCES dim_transport_location (transport_location_id)
);

-- === FACTS ===

CREATE TABLE IF NOT EXISTS fact_interior_price_spread (
    id                INTEGER PRIMARY KEY,
    date_id           INTEGER NOT NULL,
    route_id          INTEGER NOT NULL,
    year              INTEGER NOT NULL,
    month             INTEGER NOT NULL,
    origin_price      REAL,
    destination_price REAL,
    price_spread      REAL,
    source_system     TEXT NOT NULL DEFAULT 'USDA_TRANSPORT_ECWS_WVGK',
    load_time         TEXT,
    UNIQUE (date_id, route_id),
    FOREIGN KEY (date_id) REFERENCES dim_date (date_id),
    FOREIGN KEY (route_id) REFERENCES dim_transport_route (route_id)
);

CREATE TABLE IF NOT EXISTS fact_barge_rate (
    id                     INTEGER PRIMARY KEY,
    date_id                INTEGER NOT NULL,
    year                   INTEGER NOT NULL,
    month                  INTEGER NOT NULL,
    week_usda              INTEGER NOT NULL,
    transport_location_id  INTEGER NOT NULL,
    rate_percent_of_tariff REAL,
    source_system          TEXT NOT NULL DEFAULT 'USDA_TRANSPORT_DEQI_UKEN',
    load_time              TEXT,
    UNIQUE (date_id, transport_location_id),
    FOREIGN KEY (date_id)               REFERENCES dim_date (date_id),
    FOREIGN KEY (transport_location_id) REFERENCES dim_transport_location (transport_location_id)
);

CREATE TABLE IF NOT EXISTS fact_rail_rate (
    rail_rate_id              INTEGER PRIMARY KEY,
    date_id                   INTEGER NOT NULL,
    rail_route_id             INTEGER NOT NULL,
    railroad                  TEXT,
    tariff                    INTEGER,
    item                      INTEGER,
    commodity                 TEXT,
    primary_class             TEXT,
    train_type                TEXT,
    min_number_of_cars        INTEGER,
    max_number_of_cars        INTEGER,
    car_ownership             TEXT,
    rule_11_rate              INTEGER,
    gtr_table                 INTEGER,
    maximum_load_pounds       TEXT,
    car_volume_cubic_feet     TEXT,
    tariff_fsc_per_bushel     REAL,
    tariff_fsc_per_metric_ton REAL,
    tariff_fsc_per_short_ton  REAL,
    tariff_fsc_per_ton_mile   REAL,
    source_system             TEXT NOT NULL DEFAULT 'USDA_TRANSPORT_3AZ4_JKR6',
    load_time                 TEXT,
    UNIQUE (
        date_id, rail_route_id, railroad, commodity,
        train_type, car_ownership, tariff, item
    ),
    FOREIGN KEY (date_id)       REFERENCES dim_date (date_id),
    FOREIGN KEY (rail_route_id) REFERENCES dim_rail_route (rail_route_id)
);

CREATE TABLE IF NOT EXISTS fact_ag_stats (
    id                  INTEGER PRIMARY KEY,
    year                INTEGER NOT NULL,
    date_id             INTEGER,
    commodity_id        INTEGER NOT NULL,
    statistic_id        INTEGER NOT NULL,
    agg_level_id        INTEGER NOT NULL,
    state_id            INTEGER,
    country_id          INTEGER,
    reference_period_id INTEGER NOT NULL,
    value               REAL,
    domain_desc         TEXT,
    domaincat_desc      TEXT,
    location_desc       TEXT,
    source_desc         TEXT,
    load_time           TEXT,
    UNIQUE (
        year, commodity_id, statistic_id,
        agg_level_id, state_id, reference_period_id,
        domain_desc
    ),
    FOREIGN KEY (date_id)             REFERENCES dim_date (date_id),
    FOREIGN KEY (commodity_id)        REFERENCES dim_commodity (commodity_id),
    FOREIGN KEY (statistic_id)        REFERENCES dim_statistic (statistic_id),
    FOREIGN KEY (agg_level_id)        REFERENCES dim_agg_level (agg_level_id),
    FOREIGN KEY (state_id)            REFERENCES dim_state (state_id),
    FOREIGN KEY (country_id)          REFERENCES dim_country (country_id),
    FOREIGN KEY (reference_period_id) REFERENCES dim_reference_period (reference_period_id)
);

-- === EFast MArket & CME ===

CREATE TABLE IF NOT EXISTS dim_market_instrument (
    instrument_id INTEGER PRIMARY KEY,
    code TEXT UNIQUE,
    description TEXT,
    instrument_type TEXT
);

CREATE TABLE IF NOT EXISTS fact_market_price (
    id INTEGER PRIMARY KEY,
    date_id INTEGER,
    instrument_id INTEGER,
    price REAL,
    currency TEXT,
    unit TEXT,
    UNIQUE(date_id, instrument_id),
    FOREIGN KEY(date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY(instrument_id) REFERENCES dim_market_instrument(instrument_id)
);

CREATE TABLE IF NOT EXISTS fact_futures_price (
    id INTEGER PRIMARY KEY,
    date_id INTEGER,
    instrument_id INTEGER,
    close_price REAL,
    raw_close TEXT,
    UNIQUE(date_id, instrument_id),
    FOREIGN KEY(date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY(instrument_id) REFERENCES dim_market_instrument(instrument_id)
);

CREATE TABLE IF NOT EXISTS fact_br_prices (
    br_price_id      INTEGER PRIMARY KEY,
    date_id          INTEGER NOT NULL,
    state            TEXT NOT NULL,
    macroregion      TEXT,
    city             TEXT,
    destination_city TEXT,
    commodity        TEXT NOT NULL,
    metric           TEXT NOT NULL,
    unit             TEXT,
    product_item     TEXT,
    price_value      REAL NOT NULL,
    source           TEXT,
    load_time        TEXT,
    UNIQUE(
        date_id, state, macroregion, city, destination_city,
        commodity, metric, unit, product_item
    ),
    FOREIGN KEY(date_id) REFERENCES dim_date(date_id)
);



-- === ETL METADATA (optionnel) ===

CREATE TABLE IF NOT EXISTS etl_run (
    run_id        INTEGER PRIMARY KEY,
    source_system TEXT NOT NULL,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    status        TEXT,
    rows_inserted INTEGER,
    message       TEXT
);
""";


def init_db():
    execute_script(SCHEMA_SQL)


if __name__ == "__main__":
    init_db()
    print("Schema SQLite créé / mis à jour.")
