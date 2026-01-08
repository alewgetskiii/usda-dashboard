from pathlib import Path
import unicodedata
import pandas as pd
import numpy as np


def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


STATE_NAME = "Parana"

CITY_NAME_FIXES = {
    "c mourao": "Campo Mourao",
    "c procopio": "Cornelio Procopio",
    "f beltrao": "Francisco Beltrao",
    "laranj sul": "Laranjeiras do Sul",
    "p branco": "Pato Branco",
    "p grossa": "Ponta Grossa",
    "u vitoria": "Uniao da Vitoria",
}

CITY_MACROREGIONS = {
    "apucarana": "Norte Central Paranaense",
    "campo mourao": "Centro-Ocidental Paranaense",
    "cascavel": "Oeste Paranaense",
    "cornelio procopio": "Norte Pioneiro Paranaense",
    "curitiba": "Metropolitana de Curitiba",
    "francisco beltrao": "Sudoeste Paranaense",
    "guarapuava": "Centro-Sul Paranaense",
    "irati": "Sudeste Paranaense",
    "ivaipora": "Centro-Ocidental Paranaense",
    "jacarezinho": "Norte Pioneiro Paranaense",
    "laranjeiras do sul": "Centro-Sul Paranaense",
    "londrina": "Norte Central Paranaense",
    "maringa": "Norte Central Paranaense",
    "paranavai": "Noroeste Paranaense",
    "pato branco": "Sudoeste Paranaense",
    "pitanga": "Centro-Sul Paranaense",
    "ponta grossa": "Centro Oriental Paranaense",
    "toledo": "Oeste Paranaense",
    "umuarama": "Noroeste Paranaense",
    "uniao da vitoria": "Sudeste Paranaense",
}

INVALID_PRICE_TOKENS = {"", "nan", "aus", "sinf", "sem informacao"}


def normalize_city_label(value) -> str:
    base = strip_accents(str(value)).replace(",", " ").replace(".", " ")
    base = " ".join(base.split())
    key = base.lower()
    if key in CITY_NAME_FIXES:
        return CITY_NAME_FIXES[key]
    if not base:
        return ""
    words = []
    for token in base.lower().split():
        if token in {"da", "de", "do", "dos", "das"}:
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words)


def map_macroregion(city: str) -> str:
    if not city:
        return ""
    key = strip_accents(str(city)).lower()
    return CITY_MACROREGIONS.get(key, "")


def normalize_numeric_string(s: str) -> str:
    s = s.replace(" ", "")
    if "." in s and "," in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
        return s
    if "," in s and "." not in s:
        return s.replace(",", ".")
    if "." in s and "," not in s:
        parts = s.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1]) <= 2:
            return s
        return s.replace(".", "")
    return s


def parse_price_cell(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    s = (
        s.replace("\\", " ")
        .replace("/", " ")
        .replace("*", " ")
        .replace("R$", " ")
    )
    s = " ".join(s.split())
    key = strip_accents(s).lower()
    if key in INVALID_PRICE_TOKENS:
        return np.nan
    normalized = normalize_numeric_string(s)
    return pd.to_numeric(normalized, errors="coerce")


def extract_sheet_date(df: pd.DataFrame):
    """
    Cherche une valeur ressemblant à une date dans les 5 premières lignes.
    On teste toutes les colonnes, on renvoie la première date plausible.
    """
    for col in df.columns:
        col_series = df[col].head(10)
        for val in col_series:
            if pd.isna(val):
                continue
            try:
                dt = pd.to_datetime(str(val), dayfirst=True, errors="raise")
            except Exception:
                continue
            if pd.Timestamp("1990-01-01") <= dt <= pd.Timestamp("2100-01-01"):
                return dt.normalize()
    return None


def normalize_sima_sheet(df_raw: pd.DataFrame,
                         region: str,
                         source: str) -> pd.DataFrame:
    """
    Prend une feuille brute SIMA (comme ton screenshot), extrait :
      - date
      - produit
      - prix "M C" (mais comum) via colonne Média Diária
    Renvoie un DF long: date, region, product, price, source.
    Si la feuille ne matche pas le format attendu -> DF vide.
    """
    if df_raw.empty:
        return pd.DataFrame()

    # Copier & nettoyer les colonnes
    df = df_raw.copy()
    df.columns = [strip_accents(str(c)).strip() for c in df.columns]

    # 1) Trouver la date
    dt = extract_sheet_date(df)
    if dt is None:
        return pd.DataFrame()

    # 2) Trouver la colonne "PRODUTOS" et la ligne d'entête
    product_col = None
    header_row_idx = None
    for c in df.columns:
        nc = strip_accents(str(c)).lower()
        if "produto" in nc:
            product_col = c
            header_row_idx = 0
            break
        col_vals = df[c].head(12)
        for idx, val in col_vals.items():
            norm_v = strip_accents(str(val)).strip().lower()
            if norm_v and norm_v != "nan" and "produto" in norm_v:
                product_col = c
                header_row_idx = idx
                break
        if product_col is not None:
            break
    if product_col is None:
        return pd.DataFrame()

    # 3) Trouver la colonne "M C" / type (MIN / M C / MAX)
    # souvent la 2e colonne après produits, mais on le cherche par contenu
    type_col = None
    for c in df.columns:
        if c == product_col:
            continue
        # on regarde les 10 premières lignes pour quelque chose comme MIN, MAX, M C
        vals = df[c].astype(str).str.upper().str.replace(" ", "")
        if any(v in {"MIN", "MAX", "MC"} for v in vals.head(15)):
            type_col = c
            break
    if type_col is None:
        return pd.DataFrame()

    # 4) Trouver la colonne "Média Diária" (recherche aussi dans les premières lignes) - optionnel
    avg_col = None
    for c in df.columns:
        nc = strip_accents(str(c)).lower().replace(" ", "")
        if any(pat in nc for pat in ("mediadia", "mediadiaria", "mediadaria")) or nc == "media":
            avg_col = c
            break
        col_vals = df[c].head(6).astype(str).apply(lambda x: strip_accents(x).strip().lower())
        if any("media" in v or "mediadia" in v or "mediadiaria" in v for v in col_vals if v and v != "nan"):
            avg_col = c
            break

    city_cols = []
    city_label_map = {}
    if header_row_idx is not None and header_row_idx in df.index:
        header_row = df.loc[header_row_idx]
        for c in df.columns:
            if c in {product_col, type_col}:
                continue
            raw_label = header_row.get(c, None)
            if pd.isna(raw_label):
                continue
            norm_label = strip_accents(str(raw_label)).strip().lower()
            if not norm_label or norm_label == "nan":
                continue
            if "media" in norm_label or "var" in norm_label or "dia" in norm_label:
                continue
            clean_label = normalize_city_label(raw_label)
            if not clean_label:
                continue
            city_cols.append(c)
            city_label_map[c] = clean_label

    # 5) Garder seulement les lignes de type "M C" (prix le plus commun)
    # Normaliser la colonne type: enlever accents et caractères non alphanumériques
    df[type_col] = (
        df[type_col]
        .astype(str)
        .apply(strip_accents)
        .str.upper()
        .str.replace(r'[^A-Z0-9]', '', regex=True)
    )
    # Certains fichiers put the product name on the row above the MIN/MC/MAX rows;
    # forward-fill product names so the MC row has the product value.
    df[product_col] = df[product_col].astype(str).replace('nan', np.nan)
    df[product_col] = df[product_col].ffill()
    mask_mc = df[type_col].eq("MC")
    df_mc_rows = df.loc[mask_mc].copy()
    df_mc_rows = df_mc_rows.dropna(subset=[product_col])
    if df_mc_rows.empty:
        return pd.DataFrame()

    region_label = STATE_NAME if str(region).strip().upper() == "PR" else region

    out = pd.DataFrame()
    if city_cols:
        df_cities = df_mc_rows[[product_col] + city_cols].copy()
        df_long = df_cities.melt(
            id_vars=product_col,
            value_vars=city_cols,
            var_name="city_col",
            value_name="price_raw",
        )
        df_long["city"] = df_long["city_col"].map(city_label_map)
        df_long["price_mean"] = df_long["price_raw"].apply(parse_price_cell)
        df_long = df_long.dropna(subset=["city", "price_mean"])
        if not df_long.empty:
            out = pd.DataFrame({
                "date": dt,
                "region": region_label,
                "city": df_long["city"].values,
                "product": df_long[product_col].astype(str).str.strip().values,
                "price_mean": df_long["price_mean"].astype(float).values,
                "source": source,
            })

    if out.empty and avg_col is not None:
        df_mc = df_mc_rows[[product_col, avg_col]].copy()
        df_mc = df_mc.dropna(subset=[avg_col])
        if df_mc.empty:
            return pd.DataFrame()
        df_mc["price_mean"] = df_mc[avg_col].apply(parse_price_cell)
        df_mc = df_mc.dropna(subset=["price_mean"])
        if df_mc.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "date": dt,
            "region": region_label,
            "city": "",
            "product": df_mc[product_col].astype(str).str.strip().values,
            "price_mean": df_mc["price_mean"].astype(float).values,
            "source": source,
        })

    if out.empty:
        return pd.DataFrame()

    # Map Portuguese product names to English standard names and units
    product_mappings = [
        (["arroz"], "Rice", "BRL per 60kg sack"),
        (["cafe em coco", "café em coco"], "Coffee cherry", "BRL/kg"),
        (["cafe beneficiado", "café beneficiado"], "Coffee processed", "BRL per 60kg sack"),
        (["erva mate", "erva-mate", "erva mate folha"], "Yerba Mate", "BRL per @"),
        (["feijao carioca", "feijão carioca"], "Beans Carioca", "BRL per 60kg sack"),
        (["feijao preto", "feijão preto"], "Beans Black", "BRL per 60kg sack"),
        (["mandioca", "mandioca padrao"], "Cassava", "BRL per ton"),
        (["milho amarelo", "milho"], "Corn", "BRL per 60kg sack"),
        (["soja industrial", "soja"], "Soybean", "BRL per 60kg sack"),
        (["trigo"], "Wheat", "BRL per 60kg sack"),
        (["boi em pe", "boi em pé"], "Live cattle", "BRL per @"),
        (["novilho"], "Steer", "BRL per @"),
        (["vaca em pe", "vaca em pé"], "Cow", "BRL per @"),
        (
            [
                "suino em pe tipo carne nao integrado",
                "suino em pe tipo carne nao integrado kg",
            ],
            "Hog",
            "BRL/kg",
        ),
    ]

    def map_product(p: str):
        if p is None:
            return (None, None)
        s = " ".join(strip_accents(str(p)).lower().split())
        for keys, en, unit in product_mappings:
            for k in keys:
                norm_k = " ".join(strip_accents(str(k)).lower().split())
                if norm_k in s:
                    return (en, unit)
        return (None, None)

    en_names = []
    units = []
    for p in out["product"]:
        en, u = map_product(p)
        en_names.append(en if en is not None else p)
        units.append(u if u is not None else "")

    out["commodity"] = en_names
    out["unit"] = units
    out["macroregion"] = out["city"].apply(map_macroregion)

    desired_columns = [
        "date",
        "region",
        "macroregion",
        "city",
        "commodity",
        "price_mean",
        "unit",
        "product",
        "source",
    ]
    out = out[desired_columns]

    return out


def collect_sima_from_dir(root_dir: str | Path,
                          region: str = "PR") -> pd.DataFrame:
    """
    Parcourt récursivement BR_histo / root_dir,
    lit tous les .xls / .xlsx,
    extrait les prix "M C" (Média Diária) dans un DF unique.
    """
    root_dir = Path(root_dir)
    all_rows = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf not in {".xls", ".xlsx"}:
            continue

        try:
            # choose engine explicitly: .xls needs xlrd, .xlsx uses openpyxl
            if suf == '.xls':
                xls = pd.ExcelFile(path, engine='xlrd')
            else:
                xls = pd.ExcelFile(path)
        except ImportError as e:
            print(f"⚠️ Impossible de lire {path}: {e}")
            print("Installez 'xlrd' pour le support .xls: pip install xlrd")
            continue
        except Exception as e:
            print(f"⚠️ Impossible de lire {path}: {e}")
            continue

        for sheet_name in xls.sheet_names:
            try:
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                # parfois l'entête n'est pas en ligne 0 : on laisse normalize_sima_sheet se débrouiller
                df_norm = normalize_sima_sheet(
                    df_raw,
                    region=region,
                    source=f"{path.name}::{sheet_name}",
                )
                if not df_norm.empty:
                    all_rows.append(df_norm)
                else:
                    print(f"(vide) {path.name}::{sheet_name}")
            except Exception as e:
                print(f"⚠️ Erreur sur {path} / sheet '{sheet_name}': {e}")

    if not all_rows:
        print("Aucune donnée SIMA valide trouvée.")
        return pd.DataFrame()

    big_df = pd.concat(all_rows, ignore_index=True)

    # enlever les doublons éventuels (date + region + produit + prix)
    big_df = big_df.drop_duplicates(
        subset=["date", "region", "macroregion", "city", "product", "price_mean"],
        keep="last",
    ).sort_values(["region", "macroregion", "city", "product", "date"]).reset_index(drop=True)

    # Remplacer les valeurs extrêmes (>10000 ou == 0) par la dernière valeur valide de la même série
    extreme_mask = (big_df["price_mean"] > 10000) | (big_df["price_mean"] == 0)
    price_no_extremes = big_df["price_mean"].mask(extreme_mask)
    filled_prices = price_no_extremes.groupby(
        [big_df["region"], big_df["macroregion"], big_df["city"], big_df["product"]]
    ).ffill()
    big_df["price_mean"] = filled_prices.where(
        ~filled_prices.isna(),
        big_df["price_mean"],
    )

    return big_df

def run_pr_sima_pipeline() -> None:
    BASE_DIR = Path(__file__).resolve().parents[1]
    region_code = "PR"
    root = BASE_DIR / "data_raw" / region_code.upper()
    df_br = collect_sima_from_dir(root, region=STATE_NAME)

    df_br.head()
    df_br.tail()
    if df_br.empty:
        print("Aucune donnée SIMA valide trouvée. Aucun fichier généré.")
    else:
        output_name = (
            BASE_DIR
            / "data_intermediate"
            / f"{region_code.upper()}_sima_histo_prices.csv"
        )
        df_br.to_csv(output_name, index=False)


if __name__ == "__main__":
    run_pr_sima_pipeline()
