from pathlib import Path
import unicodedata
import pandas as pd
import numpy as np

# ---------- Utils ----------

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def normalize_key(s: str) -> str:
    """
    Clef normalisée:
    - sans accents
    - lower case
    - espaces réduits
    """
    s = strip_accents(str(s)).lower()
    s = " ".join(s.split())
    return s


def clean_text(value: str) -> str:
    """Remove accents and non-ASCII chars while keeping empty values blank."""
    if value is None:
        return ""
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.lower() in {"nan", "none"}:
            return ""
    else:
        if pd.isna(value):
            return ""
        candidate = str(value).strip()
    candidate = strip_accents(candidate)
    candidate = "".join(ch for ch in candidate if ord(ch) < 128)
    return candidate.strip()

def find_col(norm_cols: dict, candidates: list[str]) -> str | None:
    """
    Cherche dans norm_cols (dict {norm_name: original_name})
    la première colonne dont le nom normalisé contient un des patterns.
    """
    for pat in candidates:
        for k, orig in norm_cols.items():
            if pat in k:
                return orig
    return None

# ---------- Listes d'indicateurs à garder ----------

CORE_IMPORTANT = {
    "corn price july contract parity",
    "plume future price - july",
    "available plume price",
    "available soybean buying price",
    "soybean price march contract parity",
    "available corn buying price",
    "available grain transport price",
    "cotton lint transport price",
    "production transportation",
    "post-production",
    "post production",
    "storage",
    "armazenagem",
    "fertilizantes e corretivos",
    "fertilizantes e corretivo",
    "macronutrient",
    "macronutriente",
    "micronutrient",
    "micronutriente",
    "herbicida",
    "herbicide",
    "fungicida",
    "fungicide",
    "insecticide",
    "inseticida",
    "pesticides",
    "soybean seed",
    "corn seed",
    "corn seeds",
}

COP_ITEMS = {
    "mao de obra",
    "mao de obra familiar",
    "manutencao",
    "maquinas, implem., equip. e utilit.",
    "maquinas, implem., equip. e utilit",
    "depreciacao maquinas",
    "depreciacao equipamentos",
    "depreciacao implementos",
    "depreciacao utilitarios",
    "manutencao maq. equip. utilit.",
    "manutencao maq. equip. utilit",
    "seguro maq. equip. utilit.",
    "seguro maq. equip. utilit",
    "seguro da producao",
    "seguro da produção",
    "financiamentos",
    "funrural",
    "assistencia tecnica",
    "assistência tecnica",
    "aplicacoes com maquinas",
    "aplicacoes com maquina",
    "aplicacoes com aviao",
    "aplicacoes com avião",
    "operacoes mecanizadas",
    "operações mecanizadas",
    "arrendamento",
    "custeio",
    "custo operacional total",
    "custo total",
    "custo operacional efetivo",
    "custo de oportunidade",
    "custo de oportunidade da terra",
}

ALL_KEEP = CORE_IMPORTANT | COP_ITEMS

# ---------- Traduction EN (nom de métrique générique) ----------

METRIC_EN = {
    # core : noms courts, la commo est dans la colonne commodity
    "corn price july contract parity": "Futures parity",
    "soybean price march contract parity": "Futures parity",
    "plume future price - july": "Futures price",
    "available plume price": "Price",
    "available soybean buying price": "Buying price",
    "available corn buying price": "Buying price",
    "available grain transport price": "Transport cost",
    "cotton lint transport price": "Transport cost",
    "production transportation": "Transport cost",
    "post-production": "Post-harvest operations",
    "post production": "Post-harvest operations",
    "storage": "Storage",
    "armazenagem": "Storage",

    "fertilizantes e corretivos": "Fertilizers & soil amendments",
    "fertilizantes e corretivo": "Fertilizers & soil amendments",
    "macronutrient": "Macronutrients",
    "macronutriente": "Macronutrients",
    "micronutrient": "Micronutrients",
    "micronutriente": "Micronutrients",
    "herbicida": "Herbicide",
    "herbicide": "Herbicide",
    "fungicida": "Fungicide",
    "fungicide": "Fungicide",
    "insecticide": "Insecticide",
    "inseticida": "Insecticide",
    "pesticides": "Pesticides",
    "soybean seed": "Seed",
    "corn seed": "Seed",
    "corn seeds": "Seed",

    # COP
    "mao de obra": "Labor",
    "mao de obra familiar": "Family labor",
    "manutencao": "Maintenance",
    "maquinas, implem., equip. e utilit.": "Machines & equipment",
    "maquinas, implem., equip. e utilit": "Machines & equipment",
    "depreciacao maquinas": "Depreciation - machines",
    "depreciacao equipamentos": "Depreciation - equipment",
    "depreciacao implementos": "Depreciation - implements",
    "depreciacao utilitarios": "Depreciation - utilities",
    "manutencao maq. equip. utilit.": "Maintenance - machinery & equipment",
    "manutencao maq. equip. utilit": "Maintenance - machinery & equipment",
    "seguro maq. equip. utilit.": "Insurance - machinery & equipment",
    "seguro maq. equip. utilit": "Insurance - machinery & equipment",
    "seguro da producao": "Production insurance",
    "seguro da produção": "Production insurance",
    "financiamentos": "Financing",
    "funrural": "Funrural tax",
    "assistencia tecnica": "Technical assistance",
    "assistência tecnica": "Technical assistance",
    "aplicacoes com maquinas": "Applications with machines",
    "aplicacoes com maquina": "Applications with machines",
    "aplicacoes com aviao": "Applications by airplane",
    "aplicacoes com avião": "Applications by airplane",
    "operacoes mecanizadas": "Mechanized operations",
    "operações mecanizadas": "Mechanized operations",
    "arrendamento": "Land leasing",
    "custeio": "Operating cost",
    "custo operacional total": "Total operating cost",
    "custo total": "Total cost",
    "custo operacional efetivo": "Effective operating cost",
    "custo de oportunidade": "Opportunity cost",
    "custo de oportunidade da terra": "Land opportunity cost",
}

METRIC_EN_RENAME = {
    "Futures price": "Futures",
    "Futures parity": "Futures",
    "Buying price": "Price",
}

# ---------- Lecture & merge des 2 xls ----------

def load_cost_xls(path: Path) -> pd.DataFrame:
    """
    Lecture d'un fichier xls/xlsx :
    - item (indicator)
    - value
    - date
    - unit
    - commodity (si dispo ou inférée)
    - state / macroregion / city
    - destination city (si dispo)
    """
    xls = pd.ExcelFile(path)
    dfs = []
    for sheet in xls.sheet_names:
        df_raw = pd.read_excel(xls, sheet_name=sheet)
        if df_raw.empty:
            continue

        norm_cols = {normalize_key(col): col for col in df_raw.columns}

        # colonne item
        item_col = find_col(norm_cols, ["indicator", "item", "descricao", "description"])
        if item_col is None:
            item_col = df_raw.columns[0]

        # valeur
        value_col = find_col(norm_cols, ["value", "valor", "valores"])
        if value_col is None:
            value_col = df_raw.columns[-1]

        # commodity
        commodity_col = find_col(norm_cols, ["commodity", "cultura", "produto", "crop", "chain"])

        # date
        date_col = find_col(norm_cols, ["data", "date", "mes", "month", "periodo"])

        # unité
        unit_col = find_col(norm_cols, ["unidade", "unit", "moeda", "reais", "r$"])

        # localisation principale
        state_col = find_col(norm_cols, ["state", "estado", "uf"])
        macroregion_col = find_col(norm_cols, ["macroregion", "macro regiao", "mesorregiao"])
        city_col = find_col(norm_cols, ["city", "cidade", "municipio"])

        # destination (uniquement la ville)
        dest_city_col = find_col(
            norm_cols,
            ["destination city", "municipio destino", "cidade destino", "porto", "terminal", "destino"],
        )

        df = pd.DataFrame()
        df["item_raw"] = df_raw[item_col]

        df["value"] = pd.to_numeric(df_raw[value_col], errors="coerce")

        if commodity_col is not None:
            df["commodity_raw"] = df_raw[commodity_col]
        else:
            df["commodity_raw"] = ""

        if date_col is not None:
            df["date"] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")
        else:
            df["date"] = pd.NaT

        if unit_col is not None:
            df["unit"] = df_raw[unit_col].astype(str).str.strip()
        else:
            df["unit"] = ""

        # localisation
        df["state"] = df_raw[state_col].astype(str).str.strip() if state_col else ""
        df["macroregion"] = df_raw[macroregion_col].astype(str).str.strip() if macroregion_col else ""
        df["city"] = df_raw[city_col].astype(str).str.strip() if city_col else ""

        if dest_city_col:
            df["destination_city"] = df_raw[dest_city_col].astype(str).str.strip()
        else:
            df["destination_city"] = ""

        df["source_file"] = path.name
        df["source_sheet"] = sheet

        df = df.dropna(subset=["item_raw", "value"], how="any")
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    return out


def run_mt_imea_pipeline() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    raw_mt_dir = BASE_DIR / "data_raw" / "MT"
    output_dir = BASE_DIR / "data_intermediate"

    mt_files = sorted(
        [
            path
            for path in raw_mt_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".xls", ".xlsx"}
        ]
    )
    if not mt_files:
        raise FileNotFoundError(f"Aucun fichier .xls/.xlsx trouvé dans {raw_mt_dir}")

    dfs = []
    for mt_file in mt_files:
        try:
            dfs.append(load_cost_xls(mt_file))
        except Exception as exc:
            print(f"⚠️ Impossible de lire {mt_file}: {exc}")
    if not dfs:
        raise RuntimeError("Aucune donnée IMEA valide chargée.")

    df_all = pd.concat(dfs, ignore_index=True)

    # ---------- Normalisation noms + commodity ----------

    df_all["item_raw"] = df_all["item_raw"].astype(str).str.strip()
    df_all["item_key"] = df_all["item_raw"].apply(normalize_key)
    df_all["commodity_raw"] = df_all["commodity_raw"].astype(str).str.strip()

    def infer_commodity(row):
        c = normalize_key(row["commodity_raw"])
        if c not in {"", "nan"}:
            # nettoie un peu
            if "soy" in c or "soja" in c:
                return "Soybean"
            if "corn" in c or "milho" in c:
                return "Corn"
            if "cotton" in c or "pluma" in c:
                return "Cotton"
            return row["commodity_raw"]
        # sinon on essaie depuis l'item
        k = row["item_key"]
        if "soybean" in k or "soja" in k:
            return "Soybean"
        if "corn" in k or "milho" in k:
            return "Corn"
        if "cotton" in k or "pluma" in k:
            return "Cotton"
        return ""

    df_all["commodity"] = df_all.apply(infer_commodity, axis=1)

    # nom de métrique en anglais
    def map_metric_en(key: str, raw: str) -> str:
        return METRIC_EN.get(key, raw)

    df_all["metric_en"] = [
        map_metric_en(k, r) for k, r in zip(df_all["item_key"], df_all["item_raw"])
    ]
    df_all["metric_en"] = df_all["metric_en"].replace(METRIC_EN_RENAME)

    # ---------- Filtres pour les 2 CSV ----------

    mask_core = df_all["item_key"].isin(CORE_IMPORTANT)
    mask_all = df_all["item_key"].isin(ALL_KEEP)

    df_core = df_all[mask_core].copy()
    df_core_and_cop = df_all[mask_all].copy()

    cols_order = [
        "date",
        "state",
        "macroregion",
        "city",
        "commodity",
        "metric_en",
        "item_raw",
        "value",
        "unit",
        "destination_city",
        "source_file",
        "source_sheet",
    ]

    df_core = df_core[cols_order].sort_values(["commodity", "metric_en", "date"])
    df_core_and_cop = df_core_and_cop[cols_order].sort_values(
        ["commodity", "metric_en", "date"]
    )

    text_columns = [
        "state",
        "macroregion",
        "city",
        "commodity",
        "metric_en",
        "item_raw",
        "unit",
        "destination_city",
        "source_file",
        "source_sheet",
    ]

    for col in text_columns:
        df_core.loc[:, col] = df_core[col].apply(clean_text)
        df_core_and_cop.loc[:, col] = df_core_and_cop[col].apply(clean_text)

    # ---------- Export CSV ----------

    df_core.to_csv(output_dir / "mt_costs_core.csv", index=False)
    df_core_and_cop.to_csv(output_dir / "mt_costs_core_plus_cop.csv", index=False)

    print("Lignes core :", len(df_core))
    print("Lignes core+COP :", len(df_core_and_cop))


if __name__ == "__main__":
    run_mt_imea_pipeline()
