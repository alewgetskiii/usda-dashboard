from pathlib import Path
import pandas as pd


def run_merge_pipeline() -> None:
    """
    Merge MT cost data with PR SIMA price data on (date, state, macroregion,
    city, commodity). When keys do not overlap (different states), rows from
    each dataset are still preserved thanks to the outer join.
    """
    base_dir = Path(__file__).resolve().parents[1]
    cost_path = base_dir / "data_intermediate" / "mt_costs_core_plus_cop.csv"
    sima_path = base_dir / "data_intermediate" / "PR_sima_histo_prices.csv"
    output_path = base_dir / "data_intermediate" / "mt_costs_with_sima_prices.csv"

    df_costs = pd.read_csv(cost_path, parse_dates=["date"])
    df_prices = pd.read_csv(sima_path, parse_dates=["date"])

    # Rename columns to avoid collisions after merge.
    df_costs = df_costs.rename(
        columns={
            "metric_en": "cost_metric",
            "item_raw": "cost_item_raw",
            "value": "cost_value",
            "unit": "cost_unit",
            "source_file": "cost_source_file",
            "source_sheet": "cost_source_sheet",
        }
    )

    df_prices = df_prices.rename(
        columns={
            "region": "state",
            "price_mean": "sima_price",
            "unit": "sima_unit",
            "product": "sima_product",
            "source": "sima_source",
        }
    )

    merge_keys = ["date", "state", "macroregion", "city", "commodity"]

    # If any of the merge keys is missing in either DF (e.g. macroregion in MT costs
    # may be NaN), fill with empty strings to avoid NaN != NaN mismatch.
    for key in merge_keys:
        df_costs[key] = df_costs[key].fillna("")
        df_prices[key] = df_prices[key].fillna("")

    merged = pd.merge(
        df_costs,
        df_prices,
        on=merge_keys,
        how="outer",
        sort=True,
        suffixes=("_cost", "_sima"),
    )

    def combine_metric(row):
        cost_metric = str(row.get("cost_metric", "")).strip()
        sima_price = row.get("sima_price")
        if cost_metric and cost_metric.lower() != "nan":
            return cost_metric
        if pd.notna(sima_price):
            return "Price"
        return ""

    DEFAULT_UNITS = {
        "corn": "BRL per 60kg sack",
        "soybean": "BRL per 60kg sack",
        "cotton": "BRL per 15kg arroba",
        "cotton lint": "BRL per 15kg arroba",
        "wheat": "BRL per 60kg sack",
        "rice": "BRL per 60kg sack",
    }

    def combine_unit(row):
        cost_unit = str(row.get("cost_unit", "")).strip()
        sima_unit = str(row.get("sima_unit", "")).strip()
        if cost_unit and cost_unit.lower() != "nan":
            return cost_unit
        if sima_unit and sima_unit.lower() != "nan":
            return sima_unit
        commodity = str(row.get("commodity", "")).strip().lower()
        return DEFAULT_UNITS.get(commodity, "")

    def combine_product_item(row):
        cost_item = str(row.get("cost_item_raw", "")).strip()
        price_product = str(row.get("sima_product", "")).strip()
        if cost_item and cost_item != "nan":
            if price_product and price_product != "nan":
                return f"{cost_item} | {price_product}"
            return cost_item
        return price_product

    def combine_source(row):
        sources = []
        cost_file = str(row.get("cost_source_file", "")).strip()
        cost_sheet = str(row.get("cost_source_sheet", "")).strip()
        sima_source = str(row.get("sima_source", "")).strip()
        if cost_file and cost_file != "nan":
            if cost_sheet and cost_sheet != "nan":
                sources.append(f"{cost_file}::{cost_sheet}")
            else:
                sources.append(cost_file)
        if sima_source and sima_source != "nan":
            sources.append(sima_source)
        return " | ".join(sources)

    def combine_price_value(row):
        cost_val = row.get("cost_value")
        sima_price = row.get("sima_price")
        if pd.notna(cost_val):
            return cost_val
        return sima_price

    merged["metric"] = merged.apply(combine_metric, axis=1)
    merged["unit"] = merged.apply(combine_unit, axis=1)
    merged["product_item"] = merged.apply(combine_product_item, axis=1)
    merged["source_combined"] = merged.apply(combine_source, axis=1)
    merged["price_value"] = merged.apply(combine_price_value, axis=1)

    keep_columns = [
        "date",
        "state",
        "macroregion",
        "city",
        "commodity",
        "metric",
        "unit",
        "product_item",
        "price_value",
        "destination_city",
        "source_combined",
    ]

    # Some rows (from SIMA) won't have destination_city, so fill with empty string
    merged["destination_city"] = merged.get("destination_city", "").fillna("")

    merged = merged[keep_columns].sort_values(
        ["date", "state", "city", "commodity"]
    ).reset_index(drop=True)

    merged.to_csv(output_path, index=False)

    print(f"Merged dataset written to: {output_path}")
    print(f"Rows from MT costs: {len(df_costs):,}")
    print(f"Rows from SIMA: {len(df_prices):,}")
    print(f"Rows in merged output: {len(merged):,}")


if __name__ == "__main__":
    run_merge_pipeline()
