# usda-databuilder

Pipeline ETL pour construire une base de données SQLite à partir des APIs USDA
(Transport + Quickstats), avec intégration future dans un Docker et un dashboard.

## Structure

- `src/usda_databuilder` : code ETL (DB, schéma, extraction, pipeline).
- `dashboard/` : app pour visualiser les données (ex: Streamlit).
- `data/` : base SQLite et éventuels fichiers bruts.
