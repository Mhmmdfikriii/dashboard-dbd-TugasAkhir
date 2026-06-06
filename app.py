import random
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import MiniMap
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dashboard DBD Tangerang",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
DEFAULT_GEO_CANDIDATES = [
    ROOT_DIR / "Peta Kelurahan Kota Tangerang.geojson",
    ROOT_DIR / "map.geojson",
]
DEFAULT_DBD_CANDIDATES = [
    ROOT_DIR / "data.xlsx",
    DATA_DIR / "dbd_kelurahan.csv",
]
DEFAULT_HEALTH_FACILITY_CANDIDATES = [
    ROOT_DIR / "Jumlah Fasilitas Kesehatan Kota Tangerang, 2022.xlsx",
]
DEFAULT_CLIMATE_CANDIDATES = [
    DATA_DIR / "bmkg_climate_kecamatan.csv",
    ROOT_DIR / "bmkg_climate_kecamatan.csv",
]
YEAR_COLUMNS = ["2023", "2024", "2025"]
WARNING_CASE_THRESHOLD = 80
RISK_CASE_THRESHOLD = 60
CLIMATE_YEARS = [2023, 2024, 2025]
CLIMATE_MONTHS = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]
MONTH_NUMBER_TO_NAME = {idx: month for idx, month in enumerate(CLIMATE_MONTHS, start=1)}
MONTH_NAME_TO_NAME = {month.upper(): month for month in CLIMATE_MONTHS}
MONTH_RAIN_BASE = {
    "Januari": 320,
    "Februari": 290,
    "Maret": 240,
    "April": 170,
    "Mei": 110,
    "Juni": 60,
    "Juli": 40,
    "Agustus": 35,
    "September": 55,
    "Oktober": 120,
    "November": 220,
    "Desember": 300,
}
MONTH_TEMP_BASE = {
    "Januari": 27.8,
    "Februari": 28.0,
    "Maret": 28.4,
    "April": 29.1,
    "Mei": 29.8,
    "Juni": 30.2,
    "Juli": 30.6,
    "Agustus": 31.0,
    "September": 31.3,
    "Oktober": 30.5,
    "November": 29.4,
    "Desember": 28.4,
}
MONTH_HUMIDITY_BASE = {
    "Januari": 87,
    "Februari": 85,
    "Maret": 83,
    "April": 80,
    "Mei": 76,
    "Juni": 72,
    "Juli": 68,
    "Agustus": 66,
    "September": 68,
    "Oktober": 74,
    "November": 81,
    "Desember": 86,
}
KECAMATAN_OFFSETS = {
    "BATUCEPER": {"hujan": 10, "suhu": 0.1, "kelembapan": 1},
    "BENDA": {"hujan": -20, "suhu": 0.5, "kelembapan": -2},
    "CIBODAS": {"hujan": 18, "suhu": -0.1, "kelembapan": 2},
    "CIPONDOH": {"hujan": 25, "suhu": -0.2, "kelembapan": 3},
    "JATIUWUNG": {"hujan": -5, "suhu": 0.4, "kelembapan": -1},
    "KARANG TENGAH": {"hujan": 12, "suhu": -0.1, "kelembapan": 2},
    "KARAWACI": {"hujan": 5, "suhu": 0.2, "kelembapan": 1},
    "LARANGAN": {"hujan": 15, "suhu": 0.0, "kelembapan": 2},
    "NEGLASARI": {"hujan": -15, "suhu": 0.6, "kelembapan": -3},
    "PERIUK": {"hujan": 8, "suhu": 0.3, "kelembapan": 0},
    "PINANG": {"hujan": 20, "suhu": -0.2, "kelembapan": 3},
    "TANGERANG": {"hujan": 6, "suhu": 0.1, "kelembapan": 1},
    "CILEDUG": {"hujan": 14, "suhu": 0.0, "kelembapan": 2},
}
# Fallback jika file fasilitas kesehatan tidak tersedia.
PUSKESMAS_BY_KECAMATAN = {
    "TANGERANG": 3,
    "JATIUWUNG": 2,
    "BATUCEPER": 3,
    "BENDA": 2,
    "CIPONDOH": 4,
    "CILEDUG": 3,
    "KARAWACI": 4,
    "PERIUK": 4,
    "CIBODAS": 3,
    "NEGLASARI": 2,
    "PINANG": 4,
    "KARANG TENGAH": 3,
    "LARANGAN": 2,
}
OPTIONAL_DBD_FEATURE_ALIASES = {
    "jumlah_fasilitas_kesehatan": {
        "JUMLAHFASILITASKESEHATAN",
        "JUMLAHFASKES",
        "JUMLAHPUSKESMAS",
    },
}


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def normalize_column_token(value: str) -> str:
    return "".join(char for char in str(value).upper() if char.isalnum())


def find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def detect_optional_dbd_columns(columns: list[str]) -> dict[str, str]:
    detected: dict[str, str] = {}
    normalized_lookup = {normalize_column_token(col): col for col in columns}
    for canonical_name, aliases in OPTIONAL_DBD_FEATURE_ALIASES.items():
        for alias in aliases:
            if alias in normalized_lookup:
                detected[canonical_name] = normalized_lookup[alias]
                break
    return detected


def get_optional_health_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["jumlah_fasilitas_kesehatan"]
        if column in df.columns
    ]


def add_health_access_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


@st.cache_data
def load_health_facility_reference() -> pd.DataFrame:
    default_health_path = find_existing_path(DEFAULT_HEALTH_FACILITY_CANDIDATES)
    if default_health_path is not None:
        try:
            health_df = pd.read_excel(default_health_path, header=3)
            health_df = health_df.rename(
                columns={
                    health_df.columns[0]: "KECAMATAN_KEY",
                    health_df.columns[1]: "jumlah_rumah_sakit",
                    health_df.columns[2]: "jumlah_puskesmas",
                }
            )
            health_df["KECAMATAN_KEY"] = normalize_text(health_df["KECAMATAN_KEY"]).str.replace("  ", " ", regex=False)
            health_df["KECAMATAN_KEY"] = health_df["KECAMATAN_KEY"].replace({"KARANGTENGAH": "KARANG TENGAH"})
            health_df = health_df[health_df["KECAMATAN_KEY"] != "KOTA TANGERANG"].copy()
            for col in ["jumlah_rumah_sakit", "jumlah_puskesmas"]:
                health_df[col] = pd.to_numeric(health_df[col], errors="coerce")
            health_df["jumlah_fasilitas_kesehatan"] = (
                health_df["jumlah_rumah_sakit"].fillna(0) + health_df["jumlah_puskesmas"].fillna(0)
            )
            return health_df[["KECAMATAN_KEY", "jumlah_fasilitas_kesehatan"]].dropna(
                subset=["KECAMATAN_KEY"]
            ).reset_index(drop=True)
        except Exception:
            pass

    return pd.DataFrame(
        {
            "KECAMATAN_KEY": list(PUSKESMAS_BY_KECAMATAN.keys()),
            "jumlah_fasilitas_kesehatan": list(PUSKESMAS_BY_KECAMATAN.values()),
        }
    )


def generate_dummy_climate_dataset(dbd_df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(42)
    rows = []
    kecamatan_list = sorted(normalize_text(dbd_df["KECAMATAN"]).unique().tolist())

    for tahun in CLIMATE_YEARS:
        year_rain_shift = {2023: 0, 2024: -10, 2025: 12}[tahun]
        year_temp_shift = {2023: 0.0, 2024: 0.2, 2025: 0.4}[tahun]
        year_humidity_shift = {2023: 0, 2024: -1, 2025: 1}[tahun]

        for bulan in CLIMATE_MONTHS:
            for kecamatan in kecamatan_list:
                offsets = KECAMATAN_OFFSETS.get(kecamatan, {"hujan": 0, "suhu": 0.0, "kelembapan": 0})
                curah_hujan = MONTH_RAIN_BASE[bulan] + year_rain_shift + offsets["hujan"] + rng.randint(-25, 25)
                suhu = MONTH_TEMP_BASE[bulan] + year_temp_shift + offsets["suhu"] + rng.uniform(-0.6, 0.6)
                kelembapan = (
                    MONTH_HUMIDITY_BASE[bulan]
                    + year_humidity_shift
                    + offsets["kelembapan"]
                    + rng.randint(-4, 4)
                )
                rows.append(
                    {
                        "tahun": tahun,
                        "bulan": bulan,
                        "kecamatan": kecamatan,
                        "curah_hujan": max(0, min(500, round(curah_hujan, 1))),
                        "suhu": max(26.0, min(34.0, round(suhu, 1))),
                        "kelembapan": max(60, min(95, int(round(kelembapan)))),
                    }
                )

    climate_df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    climate_df.to_csv(DATA_DIR / "bmkg_climate_kecamatan.csv", index=False)
    return climate_df


@st.cache_data
def load_geo(src) -> gpd.GeoDataFrame:
    try:
        if src is not None:
            gdf = gpd.read_file(src)
        else:
            default_geo = find_existing_path(DEFAULT_GEO_CANDIDATES)
            if default_geo is None:
                st.error("File GeoJSON tidak ditemukan.")
                return gpd.GeoDataFrame()
            gdf = gpd.read_file(default_geo)

        if gdf.empty:
            return gpd.GeoDataFrame()

        if gdf.crs is not None:
            gdf = gdf.to_crs(epsg=4326)

        kec_candidates = ["NAME_3", "KECAMATAN", "kecamatan", "NAMA_KEC", "WADMKC"]
        kel_candidates = ["NAME_4", "KELURAHAN", "kelurahan", "NAMA_KEL", "WADMKD"]

        kec_col = next((col for col in kec_candidates if col in gdf.columns), None)
        kel_col = next((col for col in kel_candidates if col in gdf.columns), None)

        if kec_col is None or kel_col is None:
            st.error("Kolom kecamatan/kelurahan pada GeoJSON tidak ditemukan.")
            return gpd.GeoDataFrame()

        gdf["KECAMATAN"] = normalize_text(gdf[kec_col])
        gdf["KELURAHAN"] = normalize_text(gdf[kel_col])
        gdf["KECAMATAN_KEY"] = gdf["KECAMATAN"]
        gdf["KELURAHAN_KEY"] = gdf["KELURAHAN"]
        gdf["SPATIAL_KEY"] = gdf["KECAMATAN_KEY"] + "||" + gdf["KELURAHAN_KEY"]
        return gdf
    except Exception as exc:
        st.error(f"Error loading GeoJSON: {exc}")
        return gpd.GeoDataFrame()


def build_spatial_block_lookup(gdf: gpd.GeoDataFrame, n_blocks: int = 5) -> pd.DataFrame:
    if gdf.empty or "geometry" not in gdf.columns or "SPATIAL_KEY" not in gdf.columns:
        return pd.DataFrame()

    block_source = gdf[["SPATIAL_KEY", "KECAMATAN", "KELURAHAN", "geometry"]].drop_duplicates("SPATIAL_KEY").copy()
    block_source = block_source[block_source.geometry.notna()].copy()
    if len(block_source) < 2:
        return pd.DataFrame()

    projected = block_source.to_crs(epsg=3857) if block_source.crs is not None else block_source
    centroids = projected.geometry.centroid
    block_source["centroid_x"] = centroids.x.to_numpy()
    block_source["centroid_y"] = centroids.y.to_numpy()
    block_source = block_source.sort_values(["centroid_x", "centroid_y", "KECAMATAN", "KELURAHAN"]).reset_index(drop=True)

    n_splits = min(n_blocks, len(block_source))
    split_indices = np.array_split(np.arange(len(block_source)), n_splits)
    block_source["SPATIAL_BLOCK_ID"] = 0
    for block_id, indices in enumerate(split_indices, start=1):
        block_source.loc[indices, "SPATIAL_BLOCK_ID"] = block_id

    block_source["SPATIAL_BLOCK"] = block_source["SPATIAL_BLOCK_ID"].map(lambda value: f"Blok {int(value)}")
    return block_source[
        ["SPATIAL_KEY", "KECAMATAN", "KELURAHAN", "SPATIAL_BLOCK_ID", "SPATIAL_BLOCK"]
    ].reset_index(drop=True)


@st.cache_data
def load_dbd_data(src) -> pd.DataFrame:
    try:
        if src is not None:
            file_name = getattr(src, "name", "").lower()
            if file_name.endswith(".xlsx"):
                df = pd.read_excel(src)
            else:
                df = pd.read_csv(src)
        else:
            default_dbd = find_existing_path(DEFAULT_DBD_CANDIDATES)
            if default_dbd is None:
                st.error("File data DBD tidak ditemukan.")
                return pd.DataFrame()
            if default_dbd.suffix.lower() == ".xlsx":
                df = pd.read_excel(default_dbd)
            else:
                df = pd.read_csv(default_dbd)

        df.columns = [str(col).strip().upper() for col in df.columns]
        rename_map = {}
        if "KELURAHAN" not in df.columns:
            for candidate in ["NAMA", "KEL", "DESA"]:
                if candidate in df.columns:
                    rename_map[candidate] = "KELURAHAN"
                    break
        if "KECAMATAN" not in df.columns:
            for candidate in ["KEC", "NAMA_KEC"]:
                if candidate in df.columns:
                    rename_map[candidate] = "KECAMATAN"
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        required_columns = {"KECAMATAN", "KELURAHAN"}
        if not required_columns.issubset(df.columns):
            st.error("Data DBD harus memiliki kolom KECAMATAN dan KELURAHAN.")
            return pd.DataFrame()

        year_cols = [col for col in df.columns if str(col).isdigit()]
        if not year_cols:
            st.error("Data DBD tidak memiliki kolom tahun.")
            return pd.DataFrame()

        optional_columns = detect_optional_dbd_columns(df.columns.tolist())
        selected_cols = ["KECAMATAN", "KELURAHAN"] + sorted(year_cols) + list(optional_columns.values())
        df = df[selected_cols].copy()
        rename_optional_map = {original: canonical for canonical, original in optional_columns.items()}
        if rename_optional_map:
            df = df.rename(columns=rename_optional_map)
        df["KECAMATAN"] = normalize_text(df["KECAMATAN"])
        df["KELURAHAN"] = normalize_text(df["KELURAHAN"])
        for col in year_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        if "jumlah_fasilitas_kesehatan" in df.columns:
            df["jumlah_fasilitas_kesehatan"] = pd.to_numeric(df["jumlah_fasilitas_kesehatan"], errors="coerce")

        df["KECAMATAN_KEY"] = df["KECAMATAN"]
        df["KELURAHAN_KEY"] = df["KELURAHAN"]
        df["SPATIAL_KEY"] = df["KECAMATAN_KEY"] + "||" + df["KELURAHAN_KEY"]
        health_reference_df = load_health_facility_reference()
        if not health_reference_df.empty:
            reference_cols = [col for col in health_reference_df.columns if col != "KECAMATAN_KEY"]
            df = df.merge(health_reference_df, on="KECAMATAN_KEY", how="left", suffixes=("", "_ref"))
            for col in reference_cols:
                ref_col = f"{col}_ref"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if ref_col in df.columns:
                        df[col] = df[col].fillna(df[ref_col])
                        df = df.drop(columns=[ref_col])
                elif ref_col in df.columns:
                    df = df.rename(columns={ref_col: col})
        return df
    except Exception as exc:
        st.error(f"Error loading data DBD: {exc}")
        return pd.DataFrame()


@st.cache_data
def load_climate_data(src, dbd_df: pd.DataFrame) -> pd.DataFrame:
    try:
        if src is not None:
            file_name = getattr(src, "name", "").lower()
            if file_name.endswith(".xlsx"):
                df = pd.read_excel(src)
            else:
                df = pd.read_csv(src)
        else:
            df = None
            for candidate in DEFAULT_CLIMATE_CANDIDATES:
                if candidate.exists():
                    temp_df = pd.read_csv(candidate)
                    if not temp_df.empty:
                        df = temp_df
                        break
            if df is None:
                df = generate_dummy_climate_dataset(dbd_df)

        climate_aliases = {
            "KECAMATAN": "kecamatan",
            "NAMAKCAMATAN": "kecamatan",
            "NAMAKECAMATAN": "kecamatan",
            "WILAYAH": "kecamatan",
            "BULAN": "bulan",
            "MONTH": "bulan",
            "TAHUN": "tahun",
            "YEAR": "tahun",
            "TANGGAL": "tanggal",
            "DATE": "tanggal",
            "TGL": "tanggal",
            "CURAHHUJANMM": "curah_hujan",
            "CURAHHUJAN": "curah_hujan",
            "HUJAN": "curah_hujan",
            "RR": "curah_hujan",
            "SUHURATARATAC": "suhu",
            "SUHURATARATA": "suhu",
            "SUHU": "suhu",
            "TEMPERATUR": "suhu",
            "TAVG": "suhu",
            "TEMP": "suhu",
            "KELEMBAPANRATARATAPERSEN": "kelembapan",
            "KELEMBAPANRATARATA": "kelembapan",
            "KELEMBAPAN": "kelembapan",
            "RHAVG": "kelembapan",
            "HUMIDITY": "kelembapan",
        }
        rename_map = {}
        used_names = set()
        for col in df.columns:
            canonical = climate_aliases.get(normalize_column_token(col))
            if canonical is not None and canonical not in used_names:
                rename_map[col] = canonical
                used_names.add(canonical)
        df = df.rename(columns=rename_map)

        if "tanggal" in df.columns:
            tanggal = pd.to_datetime(df["tanggal"], errors="coerce", dayfirst=True)
            if "tahun" not in df.columns:
                df["tahun"] = tanggal.dt.year
            if "bulan" not in df.columns:
                df["bulan"] = tanggal.dt.month.map(MONTH_NUMBER_TO_NAME)

        if "kecamatan" not in df.columns:
            kecamatan_list = sorted(normalize_text(dbd_df["KECAMATAN"]).dropna().unique().tolist())
            if kecamatan_list:
                df = df.assign(_join_key=1).merge(
                    pd.DataFrame({"kecamatan": kecamatan_list, "_join_key": 1}),
                    on="_join_key",
                    how="inner",
                ).drop(columns=["_join_key"])

        required = {"kecamatan", "bulan", "curah_hujan", "suhu", "kelembapan"}
        if not required.issubset(df.columns):
            st.error(
                "Data iklim harus memiliki kolom wilayah/bulan/tahun dan variabel iklim. "
                "Kolom yang didukung antara lain kecamatan, bulan, tahun, curah_hujan/RR, suhu/Tavg, dan kelembapan/RH_avg."
            )
            return pd.DataFrame()

        if "tahun" not in df.columns:
            if len(CLIMATE_YEARS) > 0:
                repeated_years = np.resize(np.array(CLIMATE_YEARS), len(df))
                df["tahun"] = repeated_years
            else:
                df["tahun"] = 2025

        df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").fillna(2025).astype(int)
        bulan_numeric = pd.to_numeric(df["bulan"], errors="coerce")
        df["bulan"] = df["bulan"].astype(str).str.strip().str.title()
        df.loc[bulan_numeric.notna(), "bulan"] = bulan_numeric.dropna().astype(int).map(MONTH_NUMBER_TO_NAME)
        df["bulan"] = df["bulan"].str.upper().map(MONTH_NAME_TO_NAME).fillna(df["bulan"])
        df["kecamatan"] = normalize_text(df["kecamatan"])
        for col in ["curah_hujan", "suhu", "kelembapan"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace({9999: np.nan})
        df["curah_hujan"] = df["curah_hujan"].replace({8888: 0})

        df = df.dropna(subset=["tahun", "bulan", "kecamatan"]).copy()
        df = df[df["bulan"].isin(CLIMATE_MONTHS)].copy()
        df["KECAMATAN_KEY"] = df["kecamatan"]
        return (
            df.groupby(["tahun", "bulan", "kecamatan", "KECAMATAN_KEY"], as_index=False)
            .agg(
                curah_hujan=("curah_hujan", "sum"),
                suhu=("suhu", "mean"),
                kelembapan=("kelembapan", "mean"),
            )
        )
    except Exception as exc:
        st.error(f"Error loading data iklim: {exc}")
        return pd.DataFrame()


def build_monthly_analysis_dataset(dbd_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    dbd_yearly = dbd_df.melt(
        id_vars=["KECAMATAN", "KELURAHAN", "KECAMATAN_KEY", "KELURAHAN_KEY", "SPATIAL_KEY"]
        + get_optional_health_feature_columns(dbd_df),
        value_vars=[col for col in dbd_df.columns if str(col).isdigit()],
        var_name="tahun",
        value_name="kasus_dbd",
    )
    dbd_yearly["tahun"] = pd.to_numeric(dbd_yearly["tahun"], errors="coerce").astype(int)
    dbd_yearly["kasus_dbd"] = pd.to_numeric(dbd_yearly["kasus_dbd"], errors="coerce").fillna(0).astype(int)
    return dbd_yearly.merge(
        climate_df[["KECAMATAN_KEY", "tahun", "bulan", "curah_hujan", "suhu", "kelembapan"]],
        on=["KECAMATAN_KEY", "tahun"],
        how="left",
    )


def build_yearly_climate_summary(climate_df: pd.DataFrame) -> pd.DataFrame:
    return (
        climate_df.groupby(["KECAMATAN_KEY", "tahun"], as_index=False)
        .agg(
            total_curah_hujan_tahunan=("curah_hujan", "sum"),
            suhu_rata_rata_tahunan=("suhu", "mean"),
            kelembapan_rata_rata_tahunan=("kelembapan", "mean"),
        )
    )


def build_health_facility_df(dbd_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if dbd_df is not None and not dbd_df.empty:
        available_cols = [
            col
            for col in ["jumlah_fasilitas_kesehatan"]
            if col in dbd_df.columns
        ]
        if available_cols:
            agg_map = {col: "median" for col in available_cols}
            return (
                dbd_df[["KECAMATAN_KEY"] + available_cols]
                .groupby("KECAMATAN_KEY", as_index=False)
                .agg(agg_map)
            )
    return load_health_facility_reference()


def build_prediction_dataset(dbd_df: pd.DataFrame, yearly_climate_df: pd.DataFrame) -> pd.DataFrame:
    base = dbd_df.copy()
    health_facility_df = build_health_facility_df(dbd_df)
    base = base.merge(health_facility_df, on="KECAMATAN_KEY", how="left")
    if "jumlah_fasilitas_kesehatan_x" in base.columns:
        base["jumlah_fasilitas_kesehatan"] = pd.to_numeric(base["jumlah_fasilitas_kesehatan_x"], errors="coerce")
        if "jumlah_fasilitas_kesehatan_y" in base.columns:
            base["jumlah_fasilitas_kesehatan"] = base["jumlah_fasilitas_kesehatan"].fillna(base["jumlah_fasilitas_kesehatan_y"])
        drop_cols = [col for col in ["jumlah_fasilitas_kesehatan_x", "jumlah_fasilitas_kesehatan_y"] if col in base.columns]
        base = base.drop(columns=drop_cols)
    year_cols = sorted([col for col in base.columns if str(col).isdigit()])
    if len(year_cols) < 3:
        return pd.DataFrame()

    latest_year = int(year_cols[-1])
    prev_year = int(year_cols[-2])
    prev2_year = int(year_cols[-3])

    climate_latest = yearly_climate_df[yearly_climate_df["tahun"] == latest_year].copy()

    train_df = base.merge(
        climate_latest[
            [
                "KECAMATAN_KEY",
                "total_curah_hujan_tahunan",
                "suhu_rata_rata_tahunan",
                "kelembapan_rata_rata_tahunan",
            ]
        ],
        on="KECAMATAN_KEY",
        how="left",
    )
    train_df["lag_1"] = train_df[str(prev_year)]
    train_df["lag_2"] = train_df[str(prev2_year)]
    train_df["target"] = train_df[str(latest_year)]
    train_df["kasus_t_1"] = train_df["lag_1"]

    predict_2026_df = base.merge(
        climate_latest[
            [
                "KECAMATAN_KEY",
                "total_curah_hujan_tahunan",
                "suhu_rata_rata_tahunan",
                "kelembapan_rata_rata_tahunan",
            ]
        ],
        on="KECAMATAN_KEY",
        how="left",
    )
    predict_2026_df["lag_1"] = predict_2026_df[str(latest_year)]
    predict_2026_df["lag_2"] = predict_2026_df[str(prev_year)]
    predict_2026_df["kasus_t_1"] = predict_2026_df["lag_1"]

    predict_2027_df = base.merge(
        climate_latest[
            [
                "KECAMATAN_KEY",
                "total_curah_hujan_tahunan",
                "suhu_rata_rata_tahunan",
                "kelembapan_rata_rata_tahunan",
            ]
        ],
        on="KECAMATAN_KEY",
        how="left",
        suffixes=("", "_old"),
    )
    for col in [
        "total_curah_hujan_tahunan",
        "suhu_rata_rata_tahunan",
        "kelembapan_rata_rata_tahunan",
    ]:
        if f"{col}_old" in predict_2027_df.columns:
            predict_2027_df[col] = predict_2027_df[col].fillna(predict_2027_df[f"{col}_old"])
            predict_2027_df = predict_2027_df.drop(columns=[f"{col}_old"])
    predict_2027_df["kasus_t_1"] = predict_2027_df[str(latest_year)]

    train_df = add_health_access_features(train_df)
    predict_2026_df = add_health_access_features(predict_2026_df)
    predict_2027_df = add_health_access_features(predict_2027_df)

    return train_df, predict_2026_df, predict_2027_df


def run_prediction(dbd_df: pd.DataFrame, climate_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    year_cols = sorted([col for col in dbd_df.columns if str(col).isdigit()])
    if len(year_cols) < 3:
        return dbd_df.copy(), int(year_cols[-1]) + 1 if year_cols else 2026

    yearly_climate_df = build_yearly_climate_summary(climate_df)
    datasets = build_prediction_dataset(dbd_df, yearly_climate_df)
    if isinstance(datasets, pd.DataFrame) and datasets.empty:
        return dbd_df.copy(), int(year_cols[-1]) + 1

    train_df, predict_2026_df, predict_2027_df = datasets
    feature_cols = [
        "lag_1",
        "lag_2",
        "total_curah_hujan_tahunan",
        "suhu_rata_rata_tahunan",
        "kelembapan_rata_rata_tahunan",
        "jumlah_fasilitas_kesehatan",
    ]
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median(numeric_only=True))
    predict_2026_df[feature_cols] = predict_2026_df[feature_cols].fillna(train_df[feature_cols].median(numeric_only=True))

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(train_df[feature_cols], train_df["target"])

    result_df = dbd_df.copy()
    result_df["Prediksi 2026"] = np.clip(model.predict(predict_2026_df[feature_cols]).round().astype(int), 0, None)
    result_df["prediksi_validasi"] = np.clip(model.predict(train_df[feature_cols]).round().astype(int), 0, None)
    latest_year_col = year_cols[-1]
    result_df["error"] = (result_df[latest_year_col] - result_df["prediksi_validasi"]).abs()

    predict_2027_df[feature_cols[:2]] = 0
    predict_2027_df["lag_1"] = result_df["Prediksi 2026"].values
    predict_2027_df["lag_2"] = result_df[latest_year_col].values
    climate_medians = train_df[feature_cols[2:]].median(numeric_only=True)
    predict_2027_df[feature_cols[2:]] = predict_2027_df[feature_cols[2:]].fillna(climate_medians)
    result_df["Prediksi 2027"] = np.clip(model.predict(predict_2027_df[feature_cols]).round().astype(int), 0, None)

    result_df = attach_future_risk_predictions(result_df, dbd_df, climate_df)
    return result_df, int(latest_year_col) + 1


def build_risk_target(kasus_series: pd.Series) -> tuple[pd.Series, float]:
    threshold = float(RISK_CASE_THRESHOLD) if not kasus_series.empty else 0.0
    return (kasus_series >= threshold).astype(int), threshold


def safe_auc_scorer(estimator: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    if pd.Series(y).nunique() < 2 or len(getattr(estimator, "classes_", [])) < 2:
        return 0.5
    positive_class_index = int(np.where(estimator.classes_ == 1)[0][0]) if 1 in estimator.classes_ else -1
    y_proba = estimator.predict_proba(X)[:, positive_class_index]
    return roc_auc_score(y, y_proba)


def build_model_selection_cv(train_df: pd.DataFrame, y_train: pd.Series) -> tuple[object, pd.Series | None, str]:
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    cv_splits = min(5, min_class_count)
    return StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42), None, f"StratifiedKFold-{cv_splits}"


def build_refined_param_grid(best_params: dict) -> dict:
    param_candidates = {
        "n_estimators": [100, 200, 300, 500, 700, 1000],
        "max_depth": [10, 20, 30, 40, 50],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt"],
        "class_weight": ["balanced"],
    }
    refined_grid = {}
    for param_name, candidates in param_candidates.items():
        best_value = best_params.get(param_name, candidates[0])
        if param_name in {"max_features", "class_weight"}:
            refined_grid[param_name] = [best_value]
            continue

        best_index = candidates.index(best_value) if best_value in candidates else 0
        if param_name in {"n_estimators", "max_depth"}:
            neighbor_indices = {best_index, min(len(candidates) - 1, best_index + 1)}
        else:
            neighbor_indices = {best_index}
        refined_grid[param_name] = [candidates[idx] for idx in sorted(neighbor_indices)]
    return refined_grid


def clean_random_forest_params(params: dict | None) -> dict:
    allowed_params = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
    }
    return {key: value for key, value in (params or {}).items() if key in allowed_params}


def get_display_model_params(params: dict | None) -> dict:
    hidden_params = {
        "randomized_search_auc",
        "grid_search_auc",
        "model_selection_cv",
    }
    return {key: value for key, value in (params or {}).items() if key not in hidden_params}


def fit_risk_classifier(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[RandomForestClassifier, dict]:
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target_risiko"].astype(int)

    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count >= 2:
        cv, groups, cv_name = build_model_selection_cv(train_df, y_train)
        param_distributions = {
            "n_estimators": [100, 200, 300, 500, 700, 1000],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 4, 6, 8, 10],
            "min_samples_leaf": [1, 2, 3, 4],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
        }

        randomized_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=8,
            scoring=safe_auc_scorer,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            error_score=0.5,
        )
        randomized_search.fit(X_train, y_train, groups=groups)

        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=build_refined_param_grid(randomized_search.best_params_),
            scoring=safe_auc_scorer,
            cv=cv,
            n_jobs=-1,
            error_score=0.5,
        )
        grid_search.fit(X_train, y_train, groups=groups)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_.copy()
        best_params["randomized_search_auc"] = float(randomized_search.best_score_)
        best_params["grid_search_auc"] = float(grid_search.best_score_)
        best_params["model_selection_cv"] = cv_name
    else:
        best_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
        )
        best_model.fit(X_train, y_train)
        best_params = best_model.get_params()

    best_model.fit(X_train, y_train)
    return best_model, best_params


def optimize_probability_threshold(
    estimator: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[float, pd.DataFrame]:
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 0
    if min_class_count < 2:
        threshold_df = pd.DataFrame(
            [{"threshold": 0.5, "f1_score": np.nan, "precision": np.nan, "recall": np.nan}]
        )
        return 0.5, threshold_df

    n_splits = min(3, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_train), dtype=float)

    for train_idx, valid_idx in cv.split(X_train, y_train):
        fold_model = clone(estimator)
        fold_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        oof_proba[valid_idx] = fold_model.predict_proba(X_train.iloc[valid_idx])[:, 1]

    threshold_rows = []
    for threshold in np.arange(0.20, 0.81, 0.02):
        preds = (oof_proba >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": float(np.round(threshold, 2)),
                "f1_score": f1_score(y_train, preds, zero_division=0),
                "precision": precision_score(y_train, preds, zero_division=0),
                "recall": recall_score(y_train, preds, zero_division=0),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["f1_score", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    )
    best_threshold = float(threshold_df.iloc[0]["threshold"]) if not threshold_df.empty else 0.5
    return best_threshold, threshold_df


def get_risk_feature_columns(dbd_df: pd.DataFrame | None = None) -> list[str]:
    return [
        "total_curah_hujan_tahunan",
        "rata_rata_suhu_tahunan",
        "rata_rata_kelembapan_tahunan",
        "std_suhu_tahunan",
        "jumlah_bulan_hujan_tinggi",
        "jumlah_fasilitas_kesehatan",
        "kasus_t_1",
        "total_curah_hujan_t_1",
        "trend_perubahan_kasus",
        "interaksi_hujan_kelembapan",
        "interaksi_suhu_kelembapan",
    ]


def get_feature_label_map() -> dict[str, str]:
    return {
        "total_curah_hujan_tahunan": "total curah hujan tahunan",
        "rata_rata_suhu_tahunan": "rata-rata suhu tahunan",
        "rata_rata_kelembapan_tahunan": "rata-rata kelembapan tahunan",
        "std_suhu_tahunan": "variasi suhu tahunan",
        "jumlah_bulan_hujan_tinggi": "jumlah bulan dengan hujan tinggi",
        "jumlah_fasilitas_kesehatan": "jumlah fasilitas kesehatan",
        "kasus_t_1": "jumlah kasus tahun sebelumnya",
        "total_curah_hujan_t_1": "total curah hujan tahun sebelumnya",
        "trend_perubahan_kasus": "tren perubahan kasus",
        "interaksi_hujan_kelembapan": "gabungan curah hujan dan kelembapan",
        "interaksi_suhu_kelembapan": "gabungan suhu dan kelembapan",
    }


def summarize_feature_drivers(importance_df: pd.DataFrame, limit: int = 5) -> list[str]:
    if importance_df.empty:
        return []
    label_map = get_feature_label_map()
    top_rows = importance_df.head(limit).copy()
    return [
        f"{idx}. {label_map.get(str(row['Feature']), str(row['Feature']))}"
        for idx, (_, row) in enumerate(top_rows.iterrows(), start=1)
    ]


def build_local_driver_note(
    row: pd.Series,
    thresholds: dict[str, float],
    predicted_cases: float,
    target_year: int,
    case_threshold: float,
) -> str:
    drivers: list[str] = []
    kasus_t_1 = float(pd.to_numeric(row.get("kasus_t_1"), errors="coerce")) if pd.notna(row.get("kasus_t_1")) else np.nan
    trend_kasus = float(pd.to_numeric(row.get("trend_perubahan_kasus"), errors="coerce")) if pd.notna(row.get("trend_perubahan_kasus")) else np.nan
    curah_hujan = float(pd.to_numeric(row.get("total_curah_hujan_tahunan"), errors="coerce")) if pd.notna(row.get("total_curah_hujan_tahunan")) else np.nan
    kelembapan = float(pd.to_numeric(row.get("rata_rata_kelembapan_tahunan"), errors="coerce")) if pd.notna(row.get("rata_rata_kelembapan_tahunan")) else np.nan
    jumlah_fasilitas = float(pd.to_numeric(row.get("jumlah_fasilitas_kesehatan"), errors="coerce")) if pd.notna(row.get("jumlah_fasilitas_kesehatan")) else np.nan

    if not np.isnan(kasus_t_1) and kasus_t_1 >= thresholds.get("kasus_t_1", np.inf):
        drivers.append("kasus tahun sebelumnya relatif tinggi")
    if not np.isnan(trend_kasus) and trend_kasus > 0:
        drivers.append("tren kasus masih meningkat dibanding tahun sebelumnya")
    if not np.isnan(curah_hujan) and curah_hujan >= thresholds.get("total_curah_hujan_tahunan", np.inf):
        drivers.append("total curah hujan tahunan relatif tinggi")
    if not np.isnan(kelembapan) and kelembapan >= thresholds.get("rata_rata_kelembapan_tahunan", np.inf):
        drivers.append("kelembapan rata-rata relatif tinggi")
    if not np.isnan(jumlah_fasilitas) and jumlah_fasilitas <= thresholds.get("jumlah_fasilitas_kesehatan", -np.inf):
        drivers.append("jumlah fasilitas kesehatan relatif terbatas")

    if predicted_cases >= case_threshold:
        if drivers:
            return (
                f"Prediksi {target_year} masuk kategori tinggi terutama karena "
                + ", ".join(drivers[:3])
                + "."
            )
        return f"Prediksi {target_year} masuk kategori tinggi karena hasil prediksi kasus mencapai atau melampaui ambang risiko."

    if drivers:
        return (
            f"Prediksi {target_year} masih di bawah ambang risiko, tetapi wilayah ini tetap perlu dipantau karena "
            + ", ".join(drivers[:2])
            + "."
        )
    return f"Prediksi {target_year} masih rendah dan tidak ada sinyal dominan yang menonjol dari variabel utama pada wilayah ini."


def build_future_risk_features(
    dbd_df: pd.DataFrame,
    annual_climate: pd.DataFrame,
    future_year: int,
    lag_1_cases: pd.Series,
    lag_2_cases: pd.Series,
) -> pd.DataFrame:
    climate_reference_year = int(annual_climate["tahun"].max()) if not annual_climate.empty else future_year - 1
    climate_features = annual_climate[annual_climate["tahun"] == climate_reference_year][
        [
            "KECAMATAN_KEY",
            "total_curah_hujan_tahunan",
            "rata_rata_suhu_tahunan",
            "rata_rata_kelembapan_tahunan",
            "std_suhu_tahunan",
            "jumlah_bulan_hujan_tinggi",
        ]
    ].copy()
    climate_features = climate_features.merge(build_health_facility_df(dbd_df), on="KECAMATAN_KEY", how="left")

    future_df = dbd_df[
        ["KECAMATAN", "KELURAHAN", "KECAMATAN_KEY", "KELURAHAN_KEY", "SPATIAL_KEY"]
    ].copy()
    future_df["tahun"] = future_year
    future_df = future_df.merge(climate_features, on="KECAMATAN_KEY", how="left")
    if "jumlah_fasilitas_kesehatan_x" in future_df.columns:
        future_df["jumlah_fasilitas_kesehatan"] = pd.to_numeric(
            future_df["jumlah_fasilitas_kesehatan_x"], errors="coerce"
        )
        if "jumlah_fasilitas_kesehatan_y" in future_df.columns:
            future_df["jumlah_fasilitas_kesehatan"] = future_df["jumlah_fasilitas_kesehatan"].fillna(
                future_df["jumlah_fasilitas_kesehatan_y"]
            )
        drop_cols = [
            col for col in ["jumlah_fasilitas_kesehatan_x", "jumlah_fasilitas_kesehatan_y"] if col in future_df.columns
        ]
        future_df = future_df.drop(columns=drop_cols)
    future_df["kasus_t_1"] = pd.to_numeric(lag_1_cases, errors="coerce").fillna(0).astype(float).values
    future_df["total_curah_hujan_t_1"] = future_df["total_curah_hujan_tahunan"]
    future_df["trend_perubahan_kasus"] = (
        pd.to_numeric(lag_1_cases, errors="coerce").fillna(0).astype(float).values
        - pd.to_numeric(lag_2_cases, errors="coerce").fillna(0).astype(float).values
    )
    future_df["interaksi_hujan_kelembapan"] = (
        future_df["total_curah_hujan_tahunan"] * future_df["rata_rata_kelembapan_tahunan"]
    )
    future_df["interaksi_suhu_kelembapan"] = (
        future_df["rata_rata_suhu_tahunan"] * future_df["rata_rata_kelembapan_tahunan"]
    )
    future_df = add_health_access_features(future_df)
    return future_df


def attach_future_risk_predictions(
    predicted_df: pd.DataFrame,
    dbd_df: pd.DataFrame,
    climate_df: pd.DataFrame,
) -> pd.DataFrame:
    model_df, _ = build_classification_dataset(dbd_df, climate_df)
    if model_df.empty or model_df["target_risiko"].nunique() < 2:
        result_df = predicted_df.copy()
        result_df["Risk_Score"] = 0
        result_df["Risiko"] = "0 = Risiko Rendah"
        result_df["Probabilitas_Risiko_Tinggi_2026"] = 0.0
        result_df["Probabilitas_Risiko_Rendah_2026"] = 1.0
        result_df["Probabilitas_Label_2026"] = 1.0
        result_df["Probabilitas_Model_2026"] = "Rendah: 100.0%"
        result_df["Sinyal_RF_2026"] = "Sinyal: Risiko Rendah"
        result_df["Risk_Score_2027"] = 0
        result_df["Risiko_2027"] = "0 = Risiko Rendah"
        result_df["Probabilitas_Risiko_Tinggi_2027"] = 0.0
        result_df["Probabilitas_Risiko_Rendah_2027"] = 1.0
        result_df["Probabilitas_Label_2027"] = 1.0
        result_df["Probabilitas_Model_2027"] = "Rendah: 100.0%"
        result_df["Sinyal_RF_2027"] = "Sinyal: Risiko Rendah"
        result_df["Catatan_Analisis_2026"] = "Risiko utama mengikuti prediksi kasus yang masih di bawah ambang 60 kasus."
        result_df["Catatan_Analisis_2027"] = "Risiko utama mengikuti prediksi kasus yang masih di bawah ambang 60 kasus."
        result_df["Ambang_Risiko"] = 0.0
        return result_df

    feature_cols = get_risk_feature_columns(dbd_df)
    risk_model, _ = fit_risk_classifier(model_df, feature_cols)
    annual_climate, _ = build_annual_climate_features(climate_df)

    year_cols = sorted([col for col in dbd_df.columns if str(col).isdigit()])
    latest_year_col = year_cols[-1]
    prev_year_col = year_cols[-2]
    climate_fill_values = model_df[feature_cols].median(numeric_only=True)
    case_threshold = float(model_df["target_threshold_median"].iloc[0])
    positive_class_index = int(np.where(risk_model.classes_ == 1)[0][0]) if 1 in risk_model.classes_ else 0

    future_2026_df = build_future_risk_features(
        dbd_df=dbd_df,
        annual_climate=annual_climate,
        future_year=int(latest_year_col) + 1,
        lag_1_cases=dbd_df[latest_year_col],
        lag_2_cases=dbd_df[prev_year_col],
    )
    X_future_2026 = future_2026_df[feature_cols].fillna(climate_fill_values)
    proba_2026 = risk_model.predict_proba(X_future_2026)[:, positive_class_index]
    final_risk_2026 = (predicted_df["Prediksi 2026"] >= case_threshold).astype(int)

    future_2027_df = build_future_risk_features(
        dbd_df=dbd_df,
        annual_climate=annual_climate,
        future_year=int(latest_year_col) + 2,
        lag_1_cases=predicted_df["Prediksi 2026"],
        lag_2_cases=dbd_df[latest_year_col],
    )
    X_future_2027 = future_2027_df[feature_cols].fillna(climate_fill_values)
    proba_2027 = risk_model.predict_proba(X_future_2027)[:, positive_class_index]
    final_risk_2027 = (predicted_df["Prediksi 2027"] >= case_threshold).astype(int)
    local_thresholds = {
        "kasus_t_1": float(model_df["kasus_t_1"].median()) if "kasus_t_1" in model_df.columns else np.inf,
        "total_curah_hujan_tahunan": float(model_df["total_curah_hujan_tahunan"].median())
        if "total_curah_hujan_tahunan" in model_df.columns
        else np.inf,
        "rata_rata_kelembapan_tahunan": float(model_df["rata_rata_kelembapan_tahunan"].median())
        if "rata_rata_kelembapan_tahunan" in model_df.columns
        else np.inf,
        "jumlah_fasilitas_kesehatan": float(model_df["jumlah_fasilitas_kesehatan"].median())
        if "jumlah_fasilitas_kesehatan" in model_df.columns
        else -np.inf,
    }

    result_df = predicted_df.copy()
    result_df["Risk_Score"] = final_risk_2026.astype(int)
    result_df["Risiko"] = result_df["Risk_Score"].map(
        {
            0: "0 = Risiko Rendah",
            1: "1 = Risiko Tinggi",
        }
    )
    result_df["Probabilitas_Risiko_Tinggi_2026"] = proba_2026
    result_df["Probabilitas_Risiko_Rendah_2026"] = 1 - proba_2026
    result_df["Probabilitas_Label_2026"] = np.where(
        result_df["Risk_Score"].eq(1),
        result_df["Probabilitas_Risiko_Tinggi_2026"],
        result_df["Probabilitas_Risiko_Rendah_2026"],
    )
    result_df["Probabilitas_Model_2026"] = np.where(
        result_df["Risk_Score"].eq(1),
        "Tinggi: " + (result_df["Probabilitas_Risiko_Tinggi_2026"] * 100).round(1).astype(str) + "%",
        "Rendah: " + (result_df["Probabilitas_Risiko_Rendah_2026"] * 100).round(1).astype(str) + "%",
    )
    result_df["Sinyal_RF_2026"] = result_df["Risk_Score"].map(
        {
            0: "Sinyal: Risiko Rendah",
            1: "Sinyal: Risiko Tinggi",
        }
    )
    result_df["Risk_Score_2027"] = final_risk_2027.astype(int)
    result_df["Risiko_2027"] = result_df["Risk_Score_2027"].map(
        {
            0: "0 = Risiko Rendah",
            1: "1 = Risiko Tinggi",
        }
    )
    result_df["Probabilitas_Risiko_Tinggi_2027"] = proba_2027
    result_df["Probabilitas_Risiko_Rendah_2027"] = 1 - proba_2027
    result_df["Probabilitas_Label_2027"] = np.where(
        result_df["Risk_Score_2027"].eq(1),
        result_df["Probabilitas_Risiko_Tinggi_2027"],
        result_df["Probabilitas_Risiko_Rendah_2027"],
    )
    result_df["Probabilitas_Model_2027"] = np.where(
        result_df["Risk_Score_2027"].eq(1),
        "Tinggi: " + (result_df["Probabilitas_Risiko_Tinggi_2027"] * 100).round(1).astype(str) + "%",
        "Rendah: " + (result_df["Probabilitas_Risiko_Rendah_2027"] * 100).round(1).astype(str) + "%",
    )
    result_df["Sinyal_RF_2027"] = result_df["Risk_Score_2027"].map(
        {
            0: "Sinyal: Risiko Rendah",
            1: "Sinyal: Risiko Tinggi",
        }
    )
    result_df["Catatan_Analisis_2026"] = [
        build_local_driver_note(
            row=future_2026_df.iloc[idx],
            thresholds=local_thresholds,
            predicted_cases=float(result_df.iloc[idx]["Prediksi 2026"]),
            target_year=int(latest_year_col) + 1,
            case_threshold=case_threshold,
        )
        for idx in range(len(result_df))
    ]
    result_df["Catatan_Analisis_2027"] = [
        build_local_driver_note(
            row=future_2027_df.iloc[idx],
            thresholds=local_thresholds,
            predicted_cases=float(result_df.iloc[idx]["Prediksi 2027"]),
            target_year=int(latest_year_col) + 2,
            case_threshold=case_threshold,
        )
        for idx in range(len(result_df))
    ]
    result_df["Ambang_Risiko"] = case_threshold
    return result_df


def build_annual_climate_features(climate_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    high_rain_threshold = float(climate_df["curah_hujan"].quantile(0.75)) if not climate_df.empty else 0.0
    annual_climate = (
        climate_df.groupby(["KECAMATAN_KEY", "tahun"], as_index=False)
        .agg(
            total_curah_hujan_tahunan=("curah_hujan", "sum"),
            rata_rata_suhu_tahunan=("suhu", "mean"),
            rata_rata_kelembapan_tahunan=("kelembapan", "mean"),
            std_suhu_tahunan=("suhu", lambda values: float(np.std(values, ddof=0))),
            jumlah_bulan_hujan_tinggi=("curah_hujan", lambda values: int(np.sum(values >= high_rain_threshold))),
        )
    )
    return annual_climate, high_rain_threshold


def build_classification_dataset(
    dbd_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    spatial_blocks_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    dbd_yearly = dbd_df.melt(
        id_vars=["KECAMATAN", "KELURAHAN", "KECAMATAN_KEY", "KELURAHAN_KEY", "SPATIAL_KEY"]
        + get_optional_health_feature_columns(dbd_df),
        value_vars=[col for col in dbd_df.columns if str(col).isdigit()],
        var_name="tahun",
        value_name="kasus_dbd",
    )
    dbd_yearly["tahun"] = pd.to_numeric(dbd_yearly["tahun"], errors="coerce").astype(int)
    dbd_yearly["kasus_dbd"] = pd.to_numeric(dbd_yearly["kasus_dbd"], errors="coerce").fillna(0).astype(int)

    annual_climate, high_rain_threshold = build_annual_climate_features(climate_df)
    model_df = dbd_yearly.merge(annual_climate, on=["KECAMATAN_KEY", "tahun"], how="left")
    if spatial_blocks_df is not None and not spatial_blocks_df.empty:
        block_cols = ["SPATIAL_KEY", "SPATIAL_BLOCK_ID", "SPATIAL_BLOCK"]
        model_df = model_df.merge(
            spatial_blocks_df[[col for col in block_cols if col in spatial_blocks_df.columns]].drop_duplicates(
                "SPATIAL_KEY"
            ),
            on="SPATIAL_KEY",
            how="left",
        )
    model_df = model_df.merge(build_health_facility_df(dbd_df), on="KECAMATAN_KEY", how="left", suffixes=("", "_kec"))
    for col in ["jumlah_fasilitas_kesehatan"]:
        kec_col = f"{col}_kec"
        if kec_col in model_df.columns:
            if col in model_df.columns:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(model_df[kec_col])
                model_df = model_df.drop(columns=[kec_col])
            else:
                model_df = model_df.rename(columns={kec_col: col})
    model_df = model_df.sort_values(["SPATIAL_KEY", "tahun"]).reset_index(drop=True)

    model_df["kasus_t_1"] = model_df.groupby("SPATIAL_KEY")["kasus_dbd"].shift(1)
    model_df["total_curah_hujan_t_1"] = model_df.groupby("SPATIAL_KEY")["total_curah_hujan_tahunan"].shift(1)
    model_df["trend_perubahan_kasus"] = model_df["kasus_dbd"] - model_df["kasus_t_1"]
    model_df["interaksi_hujan_kelembapan"] = (
        model_df["total_curah_hujan_tahunan"] * model_df["rata_rata_kelembapan_tahunan"]
    )
    model_df["interaksi_suhu_kelembapan"] = (
        model_df["rata_rata_suhu_tahunan"] * model_df["rata_rata_kelembapan_tahunan"]
    )
    model_df = add_health_access_features(model_df)

    model_df["target_risiko"], target_threshold = build_risk_target(model_df["kasus_dbd"])
    model_df["target_label"] = model_df["target_risiko"].map({0: "0 = Risiko Rendah", 1: "1 = Risiko Tinggi"})
    model_df["target_threshold_median"] = target_threshold
    model_df["high_rain_threshold"] = high_rain_threshold

    required_feature_cols = [
        "kasus_t_1",
        "total_curah_hujan_t_1",
        "total_curah_hujan_tahunan",
        "rata_rata_suhu_tahunan",
        "rata_rata_kelembapan_tahunan",
        "std_suhu_tahunan",
        "jumlah_bulan_hujan_tinggi",
        "jumlah_fasilitas_kesehatan",
    ]
    model_df = model_df.dropna(subset=required_feature_cols).copy()
    return model_df, high_rain_threshold


def extract_feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    importance = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    ).sort_values("Importance", ascending=False)
    return importance_df


def run_spatial_cross_validation(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict | None = None,
) -> dict:
    block_col = "SPATIAL_BLOCK_ID"
    if block_col not in model_df.columns:
        return {"status": "missing_spatial_blocks", "block_col": block_col}

    valid_blocks = sorted(pd.to_numeric(model_df[block_col], errors="coerce").dropna().unique().tolist())
    n_splits = len(valid_blocks)
    if n_splits < 2:
        return {"status": "insufficient_spatial_blocks", "block_col": block_col, "n_blocks": int(n_splits)}

    fold_rows = []
    fold_params = clean_random_forest_params(model_params)

    for fold_no, block_id in enumerate(valid_blocks, start=1):
        train_fold_df = model_df[model_df[block_col] != block_id].copy()
        test_fold_df = model_df[model_df[block_col] == block_id].copy()
        block_name = (
            str(test_fold_df["SPATIAL_BLOCK"].dropna().iloc[0])
            if "SPATIAL_BLOCK" in test_fold_df.columns and not test_fold_df["SPATIAL_BLOCK"].dropna().empty
            else f"Blok {int(block_id)}"
        )
        wilayah_uji = ", ".join(sorted(test_fold_df["KELURAHAN"].dropna().astype(str).unique().tolist()))

        if train_fold_df["target_risiko"].nunique() < 2 or test_fold_df["target_risiko"].nunique() < 2:
            fold_rows.append(
                {
                    "Fold": fold_no,
                    "Blok Uji": block_name,
                    "Wilayah Uji": wilayah_uji,
                    "Jumlah Data Uji": int(len(test_fold_df)),
                    "Jumlah Kasus Tinggi Uji": int(test_fold_df["target_risiko"].sum()),
                    "Threshold Fold": np.nan,
                    "AUC": np.nan,
                    "F1-Score": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                }
            )
            continue

        fold_model = RandomForestClassifier(random_state=42, n_jobs=-1, **fold_params)
        fold_model.fit(train_fold_df[feature_cols].copy(), train_fold_df["target_risiko"].astype(int))
        fold_threshold, _ = optimize_probability_threshold(
            fold_model,
            train_fold_df[feature_cols].copy(),
            train_fold_df["target_risiko"].astype(int),
        )
        X_test_fold = test_fold_df[feature_cols].copy()
        y_test_fold = test_fold_df["target_risiko"].astype(int)
        y_proba_fold = (
            fold_model.predict_proba(X_test_fold)[:, 1] if len(fold_model.classes_) > 1 else np.zeros(len(X_test_fold))
        )
        y_pred_fold = (y_proba_fold >= fold_threshold).astype(int)
        fold_rows.append(
            {
                "Fold": fold_no,
                "Blok Uji": block_name,
                "Wilayah Uji": wilayah_uji,
                "Jumlah Data Uji": int(len(test_fold_df)),
                "Jumlah Kasus Tinggi Uji": int(y_test_fold.sum()),
                "Threshold Fold": float(fold_threshold),
                "AUC": roc_auc_score(y_test_fold, y_proba_fold) if y_test_fold.nunique() > 1 else np.nan,
                "F1-Score": f1_score(y_test_fold, y_pred_fold, zero_division=0),
                "Precision": precision_score(y_test_fold, y_pred_fold, zero_division=0),
                "Recall": recall_score(y_test_fold, y_pred_fold, zero_division=0),
            }
        )

    folds_df = pd.DataFrame(fold_rows)
    metric_cols = ["AUC", "F1-Score", "Precision", "Recall"]
    summary = {metric: float(folds_df[metric].mean()) for metric in metric_cols}
    summary["n_splits"] = int(n_splits)
    summary["n_blocks"] = int(n_splits)
    summary["cv_method"] = f"Spatial Cross-Validation k={n_splits}"
    summary["scope"] = "seluruh observasi berlabel yang dibagi menjadi blok spasial berdasarkan urutan koordinat centroid kelurahan"
    return {"status": "ok", "folds_df": folds_df, "summary": summary}


@st.cache_data
def run_risk_classification_pipeline(
    dbd_df: pd.DataFrame,
    climate_df: pd.DataFrame,
    spatial_blocks_df: pd.DataFrame | None = None,
) -> dict:
    model_df, high_rain_threshold = build_classification_dataset(dbd_df, climate_df, spatial_blocks_df)
    if model_df.empty:
        return {"status": "empty"}

    feature_cols = get_risk_feature_columns(dbd_df)

    available_years = sorted(model_df["tahun"].unique().tolist())
    if len(available_years) < 2:
        return {"status": "insufficient_years", "dataset": model_df}

    test_year = int(available_years[-1])
    train_years = [int(year) for year in available_years if year < test_year]
    train_df = model_df[model_df["tahun"].isin(train_years)].copy()
    test_df = model_df[model_df["tahun"] == test_year].copy()

    if train_df.empty or test_df.empty or train_df["target_risiko"].nunique() < 2:
        return {"status": "insufficient_train_data", "dataset": model_df}

    X_test = test_df[feature_cols].copy()
    y_test = test_df["target_risiko"].astype(int)

    best_model, best_params = fit_risk_classifier(train_df, feature_cols)
    spatial_cv_result = run_spatial_cross_validation(model_df, feature_cols, best_params)
    best_threshold, threshold_curve_df = optimize_probability_threshold(
        best_model,
        train_df[feature_cols].copy(),
        train_df["target_risiko"].astype(int),
    )

    y_proba = best_model.predict_proba(X_test)[:, 1] if len(best_model.classes_) > 1 else np.zeros(len(X_test))
    y_pred = (y_proba >= best_threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    test_result_df = test_df.copy()
    test_result_df["predicted_risk_score"] = y_pred.astype(int)
    test_result_df["predicted_risk_label"] = test_result_df["predicted_risk_score"].map(
        {
            0: "0 = Risiko Rendah",
            1: "1 = Risiko Tinggi",
        }
    )
    test_result_df["probabilitas_risiko_tinggi"] = y_proba

    feature_importance_df = extract_feature_importance(best_model, feature_cols)

    return {
        "status": "ok",
        "dataset": model_df,
        "train_df": train_df,
        "test_df": test_result_df,
        "feature_cols": feature_cols,
        "best_model": best_model,
        "best_params": best_params,
        "metrics": {
            "auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        },
        "decision_threshold": best_threshold,
        "threshold_curve": threshold_curve_df,
        "test_year": test_year,
        "train_years": train_years,
        "target_threshold": float(model_df["target_threshold_median"].iloc[0]),
        "high_rain_threshold": high_rain_threshold,
        "feature_importance": feature_importance_df,
        "spatial_cv": spatial_cv_result,
    }


def get_dbd_recommendations(risk_level: str) -> list[str]:
    base = [
        "Lakukan gerakan 3M Plus secara rutin di rumah dan lingkungan sekitar.",
        "Aktifkan kader Jumantik pada tingkat RT/RW untuk pemeriksaan jentik berkala.",
        "Gunakan perlindungan pribadi seperti lotion anti nyamuk dan kawat kasa.",
        "Edukasi warga terkait gejala dini dan kapan harus segera ke fasilitas kesehatan.",
    ]
    tinggi = [
        "Perkuat PSN mingguan dan inspeksi tempat penampungan air.",
        "Laksanakan penyuluhan DBD di sekolah, fasilitas kesehatan, dan forum warga.",
        "Distribusikan larvasida pada area dengan potensi genangan tinggi.",
        "Tingkatkan fogging fokus sesuai hasil surveilans lapangan.",
        "Bentuk posko siaga DBD di kelurahan dan fasilitas kesehatan.",
        "Lakukan pemetaan rumah berisiko dan kunjungan door-to-door.",
    ]
    if "1 =" in risk_level or "Tinggi" in risk_level:
        return base + tinggi
    return base


def get_top_warning_regions(
    df: pd.DataFrame,
    selected_year: str,
    selected_kecamatan: str,
    selected_kelurahan: str,
) -> pd.DataFrame:
    risk_col = "Risiko_Analisis" if "Risiko_Analisis" in df.columns else "Risiko"
    empty_columns = [
        "SPATIAL_KEY",
        "KECAMATAN",
        "KELURAHAN",
        selected_year,
        "peringkat_warning",
        "warning_text",
        risk_col,
    ]
    if selected_year not in df.columns:
        return pd.DataFrame(columns=empty_columns)

    warning_df = df.copy()
    if selected_kecamatan != "Semua":
        warning_df = warning_df[warning_df["KECAMATAN"] == selected_kecamatan]
    if selected_kelurahan != "Semua":
        warning_df = warning_df[warning_df["KELURAHAN"] == selected_kelurahan]
    warning_df = warning_df[warning_df[selected_year] >= WARNING_CASE_THRESHOLD]

    if warning_df.empty:
        return pd.DataFrame(columns=empty_columns)

    warning_df = warning_df.sort_values(
        [selected_year, "KECAMATAN", "KELURAHAN"],
        ascending=[False, True, True],
    ).copy()
    limit = 1 if selected_kelurahan != "Semua" else 5
    warning_df = warning_df.head(limit).copy()
    warning_df["peringkat_warning"] = range(1, len(warning_df) + 1)
    warning_df["warning_text"] = (
        "Top "
        + warning_df["peringkat_warning"].astype(str)
        + ": "
        + warning_df["KELURAHAN"]
        + " ("
        + warning_df["KECAMATAN"]
        + ") - "
        + warning_df[selected_year].astype(int).astype(str)
        + " kasus"
    )
    return warning_df


def build_analysis_risk_columns(
    predicted_df: pd.DataFrame,
    selected_year: int,
    selected_year_option,
) -> pd.DataFrame:
    result_df = predicted_df.copy()
    threshold_value = float(result_df["Ambang_Risiko"].iloc[0]) if "Ambang_Risiko" in result_df.columns else float(RISK_CASE_THRESHOLD)
    historical_year_col = str(selected_year)

    if selected_year_option == "Semua" or historical_year_col in result_df.columns:
        year_col = historical_year_col if historical_year_col in result_df.columns else None
        if year_col is not None:
            result_df["Risk_Score_Analisis"] = (pd.to_numeric(result_df[year_col], errors="coerce").fillna(0) >= threshold_value).astype(int)
            result_df["Risiko_Analisis"] = result_df["Risk_Score_Analisis"].map(
                {
                    0: "0 = Risiko Rendah",
                    1: "1 = Risiko Tinggi",
                }
            )
            result_df["Label_Risiko_Analisis"] = f"Risiko Tahun {selected_year}"
            return result_df

    result_df["Risk_Score_Analisis"] = result_df.get("Risk_Score", 0)
    result_df["Risiko_Analisis"] = result_df.get("Risiko", "0 = Risiko Rendah")
    result_df["Label_Risiko_Analisis"] = "Risiko Prediksi 2026"
    return result_df


def create_map(
    df: gpd.GeoDataFrame,
    selected_year: str,
    selected_month: str,
    warning_df: gpd.GeoDataFrame,
    selected_kecamatan: str,
    selected_kelurahan: str,
) -> folium.Map:
    if df.empty or "geometry" not in df.columns:
        return folium.Map(location=[-6.18, 106.63], zoom_start=11, tiles="CartoDB positron")

    center = [df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()]
    if any(pd.isna(center)):
        center = [-6.18, 106.63]

    fmap = folium.Map(location=center, zoom_start=11, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light").add_to(fmap)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    value_col = str(selected_year)
    if value_col in df.columns:
        folium.Choropleth(
            geo_data=df.to_json(),
            data=df,
            columns=["SPATIAL_KEY", value_col],
            key_on="feature.properties.SPATIAL_KEY",
            fill_color="YlOrRd",
            fill_opacity=0.78,
            line_opacity=0.35,
            line_weight=1,
            nan_fill_color="#eceff1",
            legend_name=f"Jumlah Kasus DBD Tahun {selected_year}",
            highlight=True,
            name=f"Kasus {selected_year}",
        ).add_to(fmap)

    tooltip_fields = ["KECAMATAN", "KELURAHAN"]
    tooltip_aliases = ["Kecamatan", "Kelurahan"]
    for year_col in sorted([col for col in df.columns if str(col).isdigit()]):
        tooltip_fields.append(year_col)
        tooltip_aliases.append(f"Kasus {year_col}")

    climate_fields = []
    climate_aliases = []
    risk_alias = str(df["Label_Risiko_Analisis"].iloc[0]) if "Label_Risiko_Analisis" in df.columns and not df.empty else "Risiko Analisis"
    for field, alias in [
        ("curah_hujan", f"Curah Hujan {selected_month} (mm)"),
        ("suhu", f"Suhu {selected_month} (C)"),
        ("kelembapan", f"Kelembapan {selected_month} (%)"),
        ("Prediksi 2026", "Prediksi 2026"),
        ("Prediksi 2027", "Prediksi 2027"),
        ("Risiko_Analisis", risk_alias),
    ]:
        if field in df.columns:
            climate_fields.append(field)
            climate_aliases.append(alias)

    folium.GeoJson(
        df.to_json(),
        name="Info Wilayah",
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "#1f2937",
            "weight": 1.4 if selected_kecamatan == "Semua" else 2.0,
        },
        highlight_function=lambda _: {
            "fillColor": "#fde68a",
            "fillOpacity": 0.45,
            "color": "#991b1b",
            "weight": 3.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields + climate_fields,
            aliases=tooltip_aliases + climate_aliases,
            localize=True,
            sticky=True,
            style="""
                background-color: rgba(255,255,255,0.96);
                border: 2px solid #b91c1c;
                border-radius: 8px;
                box-shadow: 0 6px 14px rgba(0,0,0,0.18);
                font-size: 13px;
                padding: 8px 10px;
            """,
        ),
    ).add_to(fmap)

    if selected_kecamatan == "Semua":
        label_source = (
            df.dissolve(by="KECAMATAN", as_index=False)
            .rename(columns={"KECAMATAN": "LABEL_NAME"})
        )
        label_font_size = "10px"
        label_padding = "2px 6px"
    elif selected_kelurahan == "Semua":
        label_source = df.copy().rename(columns={"KELURAHAN": "LABEL_NAME"})
        label_font_size = "9px"
        label_padding = "2px 4px"
    else:
        label_source = df.copy().rename(columns={"KELURAHAN": "LABEL_NAME"})
        label_font_size = "11px"
        label_padding = "3px 7px"

    if not label_source.empty:
        label_source["label_point"] = label_source.geometry.representative_point()
        for _, row in label_source.iterrows():
            point = row["label_point"]
            label_html = (
                "<div style='"
                "font-size: " + label_font_size + ";"
                "font-weight: 800;"
                "letter-spacing: 0.2px;"
                "color: #111827;"
                "background: rgba(255,255,255,0.92);"
                "padding: " + label_padding + ";"
                "border-radius: 6px;"
                "border: 1px solid rgba(153,27,27,0.35);"
                "box-shadow: 0 2px 8px rgba(15,23,42,0.12);"
                "white-space: nowrap;"
                "'>"
                f"{row['LABEL_NAME']}"
                "</div>"
            )
            folium.Marker(
                [point.y, point.x],
                icon=folium.DivIcon(html=label_html),
            ).add_to(fmap)

    if selected_kelurahan == "Semua" and not warning_df.empty:
        warning_df = warning_df.copy()
        warning_df["warning_point"] = warning_df.geometry.representative_point()
        for _, row in warning_df.iterrows():
            point = row["warning_point"]
            marker_html = f"""
            <div style="position: relative; width: 30px; height: 42px;">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 1px;
                    width: 28px;
                    height: 28px;
                    background: linear-gradient(180deg, #ef4444 0%, #b91c1c 100%);
                    border: 2px solid #ffffff;
                    border-radius: 50% 50% 50% 0;
                    transform: rotate(-45deg);
                    box-shadow: 0 8px 16px rgba(185, 28, 28, 0.35);
                "></div>
                <div style="
                    position: absolute;
                    top: 5px;
                    left: 11px;
                    color: #ffffff;
                    font-size: 12px;
                    font-weight: 800;
                    z-index: 2;
                ">{int(row['peringkat_warning'])}</div>
            </div>
            """
            popup_lines = [
                f"<b>{row['KELURAHAN']}</b>",
                f"Kecamatan: {row['KECAMATAN']}",
                f"Peringkat: Top {int(row['peringkat_warning'])}",
                f"Kasus {selected_year}: {int(row[value_col])}",
            ]
            if "curah_hujan" in row and pd.notna(row["curah_hujan"]):
                popup_lines.append(f"Curah hujan {selected_month}: {row['curah_hujan']:.1f} mm")
            if "suhu" in row and pd.notna(row["suhu"]):
                popup_lines.append(f"Suhu {selected_month}: {row['suhu']:.1f} C")
            if "kelembapan" in row and pd.notna(row["kelembapan"]):
                popup_lines.append(f"Kelembapan {selected_month}: {row['kelembapan']:.1f}%")
            if "Risiko_Analisis" in row and pd.notna(row["Risiko_Analisis"]):
                label_risiko = row.get("Label_Risiko_Analisis", "Risiko Analisis")
                popup_lines.append(f"{label_risiko}: {row['Risiko_Analisis']}")

            folium.Marker(
                location=[point.y, point.x],
                popup="<br>".join(popup_lines),
                tooltip=f"Peringatan Prioritas {int(row['peringkat_warning'])}",
                icon=folium.DivIcon(html=marker_html, icon_size=(30, 42), icon_anchor=(15, 38)),
            ).add_to(fmap)

    if selected_kelurahan != "Semua":
        bounds = df.total_bounds
        pad_x = max((bounds[2] - bounds[0]) * 0.0003, 0.00001)
        pad_y = max((bounds[3] - bounds[1]) * 0.0003, 0.00001)
        fmap.fit_bounds(
            [
                [bounds[1] - pad_y, bounds[0] - pad_x],
                [bounds[3] + pad_y, bounds[2] + pad_x],
            ]
        )
    elif len(df) > 1:
        bounds = df.total_bounds
        if selected_kecamatan == "Semua":
            pad_ratio = 0.035
            min_pad = 0.002
        elif selected_kelurahan == "Semua":
            pad_ratio = 0.02
            min_pad = 0.0012
        else:
            pad_ratio = 0.0008
            min_pad = 0.00005
        pad_x = max((bounds[2] - bounds[0]) * pad_ratio, min_pad)
        pad_y = max((bounds[3] - bounds[1]) * pad_ratio, min_pad)
        fmap.fit_bounds(
            [
                [bounds[1] - pad_y, bounds[0] - pad_x],
                [bounds[3] + pad_y, bounds[2] + pad_x],
            ]
        )
    else:
        only_point = df.geometry.iloc[0].representative_point()
        fmap.location = [only_point.y, only_point.x]
        fmap.zoom_start = 19

    MiniMap(toggle_display=True, position="bottomleft", width=120, height=120).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def create_prediction_map(
    df: gpd.GeoDataFrame,
    prediction_col: str,
    risk_col: str,
    year_label: str,
    selected_kecamatan: str,
    selected_kelurahan: str,
) -> folium.Map:
    if df.empty or "geometry" not in df.columns:
        return folium.Map(location=[-6.18, 106.63], zoom_start=11, tiles="CartoDB positron")

    df = df.copy()
    prediction_year_col = f"Tahun_Prediksi_{year_label}"
    prediction_display_col = f"{prediction_col}_Label"
    df[prediction_year_col] = year_label
    df[prediction_display_col] = (
        pd.to_numeric(df[prediction_col], errors="coerce")
        .round()
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "-")
        + " kasus"
    )

    center = [df.geometry.centroid.y.mean(), df.geometry.centroid.x.mean()]
    if any(pd.isna(center)):
        center = [-6.18, 106.63]

    fmap = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    def prediction_color(value) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "#e5e7eb"
        if numeric_value >= 80:
            return "#b91c1c"
        if numeric_value >= 60:
            return "#f97316"
        if numeric_value >= 40:
            return "#facc15"
        return "#22c55e"

    def style_prediction(feature: dict) -> dict:
        risk_value = str(feature["properties"].get(risk_col, ""))
        is_high = "Tinggi" in risk_value or risk_value.startswith("1")
        predicted_value = feature["properties"].get(prediction_col)
        return {
            "fillColor": prediction_color(predicted_value),
            "color": "#7f1d1d" if is_high else "#166534",
            "weight": 2.1 if is_high else 1.1,
            "fillOpacity": 0.76,
        }

    tooltip_fields = ["KECAMATAN", "KELURAHAN", prediction_year_col, prediction_display_col, risk_col]
    tooltip_aliases = ["Kecamatan: ", "Kelurahan: ", "Tahun Prediksi: ", "Jumlah Kasus: ", f"Risiko {year_label}: "]

    folium.GeoJson(
        df.to_json(),
        name=f"Prediksi {year_label}",
        style_function=style_prediction,
        highlight_function=lambda _: {
            "fillColor": "#fde68a",
            "fillOpacity": 0.64,
            "color": "#111827",
            "weight": 3.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            style="""
                background-color: rgba(255,255,255,0.96);
                border: 2px solid #374151;
                border-radius: 8px;
                box-shadow: 0 6px 14px rgba(0,0,0,0.18);
                font-size: 13px;
                padding: 8px 10px;
            """,
        ),
    ).add_to(fmap)

    if selected_kecamatan == "Semua":
        label_source = df.dissolve(by="KECAMATAN", as_index=False).rename(columns={"KECAMATAN": "LABEL_NAME"})
        label_font_size = "10px"
        label_padding = "2px 6px"
    elif selected_kelurahan == "Semua":
        label_source = df.copy().rename(columns={"KELURAHAN": "LABEL_NAME"})
        label_font_size = "9px"
        label_padding = "2px 4px"
    else:
        label_source = df.copy().rename(columns={"KELURAHAN": "LABEL_NAME"})
        label_font_size = "11px"
        label_padding = "3px 7px"

    if not label_source.empty:
        label_source["label_point"] = label_source.geometry.representative_point()
        for _, row in label_source.iterrows():
            point = row["label_point"]
            label_html = (
                "<div style='"
                "font-size: " + label_font_size + ";"
                "font-weight: 800;"
                "color: #111827;"
                "background: rgba(255,255,255,0.88);"
                "padding: " + label_padding + ";"
                "border-radius: 6px;"
                "border: 1px solid rgba(31,41,55,0.22);"
                "box-shadow: 0 2px 7px rgba(15,23,42,0.12);"
                "white-space: nowrap;"
                "'>"
                f"{row['LABEL_NAME']}"
                "</div>"
            )
            folium.Marker(
                [point.y, point.x],
                icon=folium.DivIcon(html=label_html),
            ).add_to(fmap)

    top_prediction_df = df.copy()
    top_prediction_df[prediction_col] = pd.to_numeric(top_prediction_df[prediction_col], errors="coerce")
    top_prediction_df = top_prediction_df.dropna(subset=[prediction_col]).sort_values(prediction_col, ascending=False)
    if selected_kelurahan == "Semua":
        for rank, (_, row) in enumerate(top_prediction_df.head(5).iterrows(), start=1):
            point = row.geometry.representative_point()
            marker_html = f"""
            <div style="position: relative; width: 30px; height: 42px;">
                <div style="
                    position: absolute;
                    top: 0;
                    left: 1px;
                    width: 28px;
                    height: 28px;
                    background: linear-gradient(180deg, #ef4444 0%, #b91c1c 100%);
                    border: 2px solid #ffffff;
                    border-radius: 50% 50% 50% 0;
                    transform: rotate(-45deg);
                    box-shadow: 0 8px 16px rgba(185, 28, 28, 0.35);
                "></div>
                <div style="
                    position: absolute;
                    top: 5px;
                    left: 11px;
                    color: #ffffff;
                    font-size: 12px;
                    font-weight: 800;
                    z-index: 2;
                ">{rank}</div>
            </div>
            """
            popup_lines = [
                f"<b>{row['KELURAHAN']}</b>",
                f"Kecamatan: {row['KECAMATAN']}",
                f"Peringkat: Top {rank}",
                f"Tahun prediksi: {year_label}",
                f"Jumlah kasus: {int(row[prediction_col])}",
                f"Risiko: {row.get(risk_col, '-')}",
            ]
            folium.Marker(
                location=[point.y, point.x],
                popup="<br>".join(popup_lines),
                tooltip=f"Peringatan Prediksi Top {rank}",
                icon=folium.DivIcon(html=marker_html, icon_size=(30, 42), icon_anchor=(15, 38)),
            ).add_to(fmap)

    legend_html = f"""
    <div style="
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        padding: 8px 10px;
        border: 1px solid #d1d5db;
        border-radius: 7px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.14);
        font-size: 11px;
        color: #111827;
        line-height: 1.25;
    ">
        <div style="font-weight: 800; margin-bottom: 5px;">Prediksi {year_label}</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#22c55e;margin-right:5px;"></span>&lt; 40</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#facc15;margin-right:5px;"></span>40-59</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#f97316;margin-right:5px;"></span>60-79</div>
        <div><span style="display:inline-block;width:10px;height:10px;background:#b91c1c;margin-right:5px;"></span>&ge; 80</div>
        <div style="margin-top:5px;border-top:1px solid #e5e7eb;padding-top:5px;">Batas merah: tinggi</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    if selected_kelurahan != "Semua":
        bounds = df.total_bounds
        pad_x = max((bounds[2] - bounds[0]) * 0.0003, 0.00001)
        pad_y = max((bounds[3] - bounds[1]) * 0.0003, 0.00001)
        fmap.fit_bounds([[bounds[1] - pad_y, bounds[0] - pad_x], [bounds[3] + pad_y, bounds[2] + pad_x]])
    elif len(df) > 1:
        bounds = df.total_bounds
        pad_ratio = 0.035 if selected_kecamatan == "Semua" else 0.02
        min_pad = 0.002 if selected_kecamatan == "Semua" else 0.0012
        pad_x = max((bounds[2] - bounds[0]) * pad_ratio, min_pad)
        pad_y = max((bounds[3] - bounds[1]) * pad_ratio, min_pad)
        fmap.fit_bounds([[bounds[1] - pad_y, bounds[0] - pad_x], [bounds[3] + pad_y, bounds[2] + pad_x]])
    else:
        only_point = df.geometry.iloc[0].representative_point()
        fmap.location = [only_point.y, only_point.x]
        fmap.zoom_start = 19

    return fmap


def main() -> None:
    st.title("Dashboard Prediksi Risiko Demam Berdarah Dengue Kota Tangerang")
    st.caption(
        "Sistem analisis berbasis GIS dan Random Forest untuk pemantauan kasus DBD, variabel iklim, "
        "klasifikasi risiko, dan prediksi wilayah prioritas."
    )
    st.markdown("### Monitoring Spasial dan Prediksi Risiko DBD")

    st.sidebar.header("Pengaturan Data dan Filter")
    geo_file = st.sidebar.file_uploader("Unggah GeoJSON Kelurahan (opsional)", type=["geojson", "json"])
    dbd_file = st.sidebar.file_uploader("Unggah Data Kasus DBD CSV/Excel (opsional)", type=["csv", "xlsx"])
    climate_file = st.sidebar.file_uploader("Unggah Data Iklim BMKG CSV/Excel (opsional)", type=["csv", "xlsx"])

    with st.spinner("Memuat data dan menyiapkan dashboard..."):
        dbd_df = load_dbd_data(dbd_file)
        if dbd_df.empty:
            st.error("Data DBD tidak dapat dimuat.")
            return
        climate_df = load_climate_data(climate_file, dbd_df)
        gdf = load_geo(geo_file)

    dbd_master_df = dbd_df.copy()

    if climate_df.empty:
        st.error("Data iklim tidak dapat dimuat.")
        return
    if gdf.empty:
        st.error("Data geografis tidak dapat dimuat.")
        return

    available_years = sorted([int(col) for col in dbd_df.columns if str(col).isdigit()])
    available_months = [month for month in CLIMATE_MONTHS if month in climate_df["bulan"].unique()]
    available_kecamatan = ["Semua"] + sorted(dbd_df["KECAMATAN"].dropna().unique().tolist())

    selected_kecamatan = st.sidebar.selectbox("Kecamatan", available_kecamatan)
    kelurahan_source = dbd_df.copy()
    if selected_kecamatan != "Semua":
        kelurahan_source = kelurahan_source[kelurahan_source["KECAMATAN"] == selected_kecamatan]
    available_kelurahan = ["Semua"] + sorted(kelurahan_source["KELURAHAN"].dropna().unique().tolist())

    selected_kelurahan = st.sidebar.selectbox("Kelurahan", available_kelurahan)
    year_options = ["Semua"] + available_years
    selected_year_option = st.sidebar.selectbox("Tahun Analisis", year_options, index=0)
    selected_month = st.sidebar.selectbox("Bulan Iklim", available_months, index=0 if available_months else 0)
    show_only_matching_climate = st.sidebar.checkbox("Terapkan rentang iklim pada peta", value=False)

    if selected_year_option == "Semua":
        active_years = available_years
    else:
        active_years = [int(selected_year_option)]
    selected_year = max(active_years)
    active_year_cols = [str(year) for year in active_years]

    climate_value_cols = ["curah_hujan", "suhu", "kelembapan"]
    climate_years_for_filter = active_years if selected_year_option == "Semua" else [selected_year]
    climate_filter_df = climate_df[
        (climate_df["tahun"].isin(climate_years_for_filter)) & (climate_df["bulan"] == selected_month)
    ].copy()
    if climate_filter_df[climate_value_cols].dropna(how="all").empty:
        valid_climate_years = sorted(
            climate_df.dropna(subset=climate_value_cols, how="all")["tahun"].dropna().astype(int).unique().tolist()
        )
        if valid_climate_years:
            previous_valid_years = [year for year in valid_climate_years if year <= selected_year]
            fallback_climate_year = max(previous_valid_years) if previous_valid_years else max(valid_climate_years)
            climate_years_for_filter = [fallback_climate_year]
            climate_filter_df = climate_df[
                (climate_df["tahun"].isin(climate_years_for_filter)) & (climate_df["bulan"] == selected_month)
            ].copy()
            st.sidebar.info(
                f"Data iklim {selected_year} belum tersedia, filter iklim memakai data {fallback_climate_year}."
            )

    rain_min, rain_max = int(climate_filter_df["curah_hujan"].min()), int(climate_filter_df["curah_hujan"].max())
    temp_min, temp_max = float(climate_filter_df["suhu"].min()), float(climate_filter_df["suhu"].max())
    humid_min, humid_max = int(climate_filter_df["kelembapan"].min()), int(climate_filter_df["kelembapan"].max())

    selected_rain = st.sidebar.slider("Curah Hujan (mm)", rain_min, rain_max, (rain_min, rain_max))
    selected_temp = st.sidebar.slider("Suhu (C)", temp_min, temp_max, (temp_min, temp_max))
    selected_humid = st.sidebar.slider("Kelembapan (%)", humid_min, humid_max, (humid_min, humid_max))

    if selected_kecamatan != "Semua":
        dbd_df = dbd_df[dbd_df["KECAMATAN"] == selected_kecamatan]
    if selected_kelurahan != "Semua":
        dbd_df = dbd_df[dbd_df["KELURAHAN"] == selected_kelurahan]

    keep_dbd_cols = [
        "KECAMATAN",
        "KELURAHAN",
        "KECAMATAN_KEY",
        "KELURAHAN_KEY",
        "SPATIAL_KEY",
    ] + active_year_cols + get_optional_health_feature_columns(dbd_df)
    dbd_df = dbd_df[keep_dbd_cols].copy()

    analysis_df = build_monthly_analysis_dataset(dbd_df, climate_df)
    yearly_climate_df = build_yearly_climate_summary(climate_df)
    predicted_df, predicted_year = run_prediction(dbd_master_df, climate_df)
    if selected_kecamatan != "Semua":
        predicted_df = predicted_df[predicted_df["KECAMATAN"] == selected_kecamatan].copy()
    if selected_kelurahan != "Semua":
        predicted_df = predicted_df[predicted_df["KELURAHAN"] == selected_kelurahan].copy()
    predicted_df = build_analysis_risk_columns(predicted_df, selected_year, selected_year_option)
    spatial_blocks_df = build_spatial_block_lookup(gdf, n_blocks=5)
    risk_pipeline = run_risk_classification_pipeline(dbd_master_df, climate_df, spatial_blocks_df)

    climate_selected = climate_df[
        (climate_df["tahun"].isin(climate_years_for_filter))
        & (climate_df["bulan"] == selected_month)
        & (climate_df["curah_hujan"].between(selected_rain[0], selected_rain[1]))
        & (climate_df["suhu"].between(selected_temp[0], selected_temp[1]))
        & (climate_df["kelembapan"].between(selected_humid[0], selected_humid[1]))
    ].copy()
    if selected_kecamatan != "Semua":
        climate_selected = climate_selected[climate_selected["kecamatan"] == selected_kecamatan]

    prediction_map_df = gdf.merge(
        predicted_df,
        on=["SPATIAL_KEY", "KECAMATAN", "KELURAHAN", "KECAMATAN_KEY", "KELURAHAN_KEY"],
        how="inner",
    )
    map_df = prediction_map_df.copy()
    monthly_map = (
        climate_selected.groupby("KECAMATAN_KEY", as_index=False)[["curah_hujan", "suhu", "kelembapan"]].mean()
    )
    map_df = map_df.merge(monthly_map, on="KECAMATAN_KEY", how="left")

    if show_only_matching_climate:
        map_df = map_df[map_df["KECAMATAN_KEY"].isin(climate_selected["KECAMATAN_KEY"].unique())].copy()

    warning_df = get_top_warning_regions(
        predicted_df,
        str(selected_year),
        selected_kecamatan,
        selected_kelurahan,
    )
    warning_merge_cols = ["SPATIAL_KEY", "peringkat_warning", "warning_text", str(selected_year)]
    warning_merge_on = ["SPATIAL_KEY", str(selected_year)]
    if "Risiko_Analisis" in warning_df.columns and "Risiko_Analisis" in map_df.columns:
        warning_merge_cols.append("Risiko_Analisis")
        warning_merge_on.append("Risiko_Analisis")
    if warning_df.empty:
        warning_map_df = map_df.iloc[0:0].copy()
    else:
        warning_map_df = map_df.merge(
            warning_df[warning_merge_cols],
            on=warning_merge_on,
            how="inner",
        )

    year_cols = sorted([col for col in predicted_df.columns if str(col).isdigit()])
    total_wilayah = len(predicted_df)
    total_actual = int(predicted_df[str(selected_year)].fillna(0).sum())
    total_historical = int(predicted_df[year_cols].fillna(0).sum().sum()) if year_cols else 0
    avg_rain = climate_selected["curah_hujan"].mean() if not climate_selected.empty else np.nan
    avg_temp = climate_selected["suhu"].mean() if not climate_selected.empty else np.nan
    avg_humid = climate_selected["kelembapan"].mean() if not climate_selected.empty else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wilayah Terpantau", total_wilayah)
    year_metric_label = f"Total Kasus {selected_year}" if selected_year_option != "Semua" else "Total Kasus Historis"
    year_metric_value = total_actual if selected_year_option != "Semua" else total_historical
    m2.metric(year_metric_label, f"{year_metric_value:,}")
    m3.metric("Rata-rata Curah Hujan", "-" if pd.isna(avg_rain) else f"{avg_rain:.1f} mm")
    m4.metric("Rata-rata Kelembapan", "-" if pd.isna(avg_humid) else f"{avg_humid:.1f}%")

    st.subheader("Peta Sebaran Kasus DBD")
    if selected_kelurahan != "Semua":
        st.caption(
            "Peta difokuskan pada kelurahan terpilih untuk menampilkan batas administrasi dan informasi wilayah secara lebih rinci."
        )
    elif selected_kecamatan != "Semua":
        st.caption(
            f"Peta menampilkan kelurahan pada kecamatan terpilih. Penanda prioritas ditampilkan untuk wilayah dengan kasus tahun {selected_year} minimal {WARNING_CASE_THRESHOLD} kasus."
        )
    else:
        st.caption(
            f"Peta seluruh wilayah menampilkan ringkasan spasial Kota Tangerang. Penanda prioritas diberikan pada wilayah dengan kasus tahun {selected_year} minimal {WARNING_CASE_THRESHOLD} kasus."
        )
    if selected_year_option == "Semua":
        st.info(
            f"Mode `Semua` tetap memakai tahun historis terakhir, yaitu {selected_year}, untuk pewarnaan peta dan label risiko analisis agar kasus dan risiko dibaca pada konteks tahun yang sama."
        )
    fmap = create_map(
        map_df,
        str(selected_year),
        selected_month,
        warning_map_df,
        selected_kecamatan,
        selected_kelurahan,
    )
    map_height = 780 if selected_kelurahan != "Semua" else 620 if selected_kecamatan != "Semua" else 560
    st_folium(fmap, width=None, height=map_height, returned_objects=[])

    if map_df.empty:
        st.warning("Tidak terdapat wilayah yang sesuai dengan kombinasi filter yang dipilih.")
        return

    if not warning_df.empty:
        warning_display_cols = [
            "peringkat_warning",
            "KECAMATAN",
            "KELURAHAN",
            str(selected_year),
        ]
        warning_rename_map = {
            "peringkat_warning": "Peringkat",
            "KECAMATAN": "Kecamatan",
            "KELURAHAN": "Kelurahan",
            str(selected_year): f"Kasus {selected_year}",
        }
        if "Risiko" in warning_df.columns:
            warning_display_cols.append("Risiko")
            warning_rename_map["Risiko"] = "Prediksi Risiko 2026"
        if "Risiko_Analisis" in warning_df.columns:
            warning_display_cols.append("Risiko_Analisis")
            warning_rename_map["Risiko_Analisis"] = f"Risiko Tahun {selected_year}"
        st.markdown(f"#### Wilayah Prioritas Berdasarkan Kasus {selected_year}")
        st.caption(
            f"Tabel ini diurutkan dari kasus DBD tertinggi pada tahun {selected_year}. "
            f"`Risiko Tahun {selected_year}` menunjukkan kondisi historis pada tahun yang dipilih, sedangkan `Prediksi Risiko 2026` menunjukkan hasil model untuk tahun berikutnya."
        )
        st.dataframe(
            warning_df[warning_display_cols].rename(columns=warning_rename_map),
            use_container_width=True,
            hide_index=True,
        )
    elif selected_kelurahan == "Semua":
        st.info(
            f"Tidak terdapat wilayah yang mencapai ambang prioritas {WARNING_CASE_THRESHOLD} kasus pada tahun {selected_year}."
        )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Tren Kasus Tahunan")
        yearly_data = pd.DataFrame(
            {
                "Tahun": year_cols,
                "Total Kasus": [int(predicted_df[col].fillna(0).sum()) for col in year_cols],
            }
        )
        yearly_data["Perubahan dari Tahun Sebelumnya"] = yearly_data["Total Kasus"].diff().fillna(0).astype(int)
        yearly_data["Persentase Perubahan (%)"] = (
            yearly_data["Total Kasus"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0) * 100
        ).round(2)

        if len(yearly_data) >= 2:
            latest_change = int(yearly_data.iloc[-1]["Perubahan dari Tahun Sebelumnya"])
            trend_label = "naik" if latest_change > 0 else "turun" if latest_change < 0 else "stabil"
            st.caption(
                f"Tren tahunan menunjukkan jumlah kasus DBD {trend_label} sebesar {abs(latest_change):,} kasus dibandingkan tahun sebelumnya."
            )
        else:
            st.caption("Grafik menampilkan total kasus pada tahun analisis yang dipilih.")

        st.line_chart(yearly_data.set_index("Tahun")[["Total Kasus"]], use_container_width=True)
        st.dataframe(
            yearly_data.rename(
                columns={
                    "Tahun": "Tahun",
                    "Total Kasus": "Total Kasus DBD",
                    "Perubahan dari Tahun Sebelumnya": "Selisih Tahunan",
                    "Persentase Perubahan (%)": "Perubahan (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=220,
        )

    with c2:
        st.subheader("Distribusi Kasus per Wilayah")
        latest_year_col = year_cols[-1]
        top_10 = (
            predicted_df[["KECAMATAN", "KELURAHAN", latest_year_col]]
            .sort_values(latest_year_col, ascending=False)
            .head(10)
            .set_index("KELURAHAN")
        )
        st.bar_chart(top_10[[latest_year_col]], use_container_width=True)

    st.markdown("---")
    ik1, ik2 = st.columns(2)

    with ik1:
        st.subheader("Ringkasan Variabel Iklim")
        if climate_selected.empty:
            st.info("Tidak ada data iklim yang sesuai dengan filter yang dipilih.")
        else:
            climate_summary = (
                climate_selected.groupby("kecamatan", as_index=False)
                .agg(
                    curah_hujan=("curah_hujan", "mean"),
                    suhu=("suhu", "mean"),
                    kelembapan=("kelembapan", "mean"),
                )
                .rename(
                    columns={
                        "kecamatan": "Kecamatan",
                        "curah_hujan": "Curah Hujan (mm)",
                        "suhu": "Suhu (C)",
                        "kelembapan": "Kelembapan (%)",
                    }
                )
                .sort_values("Kecamatan")
            )
            st.table(climate_summary.round(2))

    with ik2:
        st.subheader("Korelasi Kasus DBD dan Variabel Iklim")
        corr_df = analysis_df[["kasus_dbd", "curah_hujan", "suhu", "kelembapan"]].dropna()
        if not corr_df.empty:
            corr = corr_df.corr(numeric_only=True)["kasus_dbd"].drop("kasus_dbd").sort_values(key=np.abs, ascending=False)
            corr_table = corr.reset_index()
            corr_table.columns = ["Variabel Iklim", "Korelasi dengan Kasus DBD"]
            st.table(corr_table.round(3))
            strongest = corr.index[0]
            st.info(
                f"Variabel iklim dengan korelasi tertinggi terhadap kasus DBD adalah `{strongest}` dengan nilai korelasi `{corr.iloc[0]:.2f}`."
            )
        else:
            st.info("Data belum mencukupi untuk menghitung korelasi.")

    st.markdown("---")
    st.subheader("Pipeline Machine Learning")

    if risk_pipeline.get("status") == "ok":
        model_year_for_display = None if selected_year_option == "Semua" else int(selected_year_option)
        test_result_df = risk_pipeline["test_df"].copy()
        if selected_kecamatan != "Semua":
            test_result_df = test_result_df[test_result_df["KECAMATAN"] == selected_kecamatan]
        if selected_kelurahan != "Semua":
            test_result_df = test_result_df[test_result_df["KELURAHAN"] == selected_kelurahan]
        show_model_test_table = selected_year_option == "Semua" or model_year_for_display == int(risk_pipeline["test_year"])

        spatial_cv = risk_pipeline.get("spatial_cv", {})
        spatial_cv_context = "gabungan tahun 2024 dan 2025"
        if model_year_for_display == 2023:
            spatial_cv = {"status": "base_year"}
            spatial_cv_context = "tahun 2023"
        elif model_year_for_display in risk_pipeline["dataset"]["tahun"].unique().tolist():
            spatial_source_df = risk_pipeline["dataset"][risk_pipeline["dataset"]["tahun"] == model_year_for_display].copy()
            spatial_cv = run_spatial_cross_validation(
                spatial_source_df,
                risk_pipeline["feature_cols"],
                risk_pipeline["best_params"],
            )
            spatial_cv_context = f"tahun {model_year_for_display}"

        with st.expander("Metodologi Pemodelan", expanded=False):
            st.markdown(
                f"""
1. **Pengumpulan Data Sekunder**
   Data yang digunakan pada aplikasi ini mencakup data kasus DBD per kelurahan tahun 2023, 2024, dan 2025, data iklim dari BMKG, data jumlah fasilitas kesehatan per kecamatan, serta data batas administrasi kelurahan dalam format spasial.
2. **Pra-pemrosesan Data**
   Tahap ini meliputi pembersihan kolom, normalisasi nama kecamatan dan kelurahan, konversi data numerik, standarisasi sistem koordinat peta ke `EPSG:4326`, serta penyelarasan data tabular dan data spasial menggunakan kunci wilayah. Data iklim BMKG yang berbasis tanggal atau bulan diubah terlebih dahulu menjadi data bulanan, kemudian diagregasi menjadi data tahunan agar sesuai dengan data kasus DBD tahunan.
3. **Perekayasaan Fitur**
   Karena data kasus DBD tersedia per tahun, maka data iklim yang awalnya per bulan terlebih dahulu diagregasi menjadi fitur tahunan per kecamatan. Fitur tambahan dibuat secara berurutan antar tahun: data 2023 menjadi `kasus_t-1` untuk baris 2024, dan data 2024 menjadi `kasus_t-1` untuk baris 2025. Jika filter tahun dipilih `Semua`, maka data 2023, 2024, dan 2025 tetap ditampilkan dalam analisis. Fitur iklim yang digunakan meliputi `total_curah_hujan_tahunan`, `rata_rata_suhu_tahunan`, `rata_rata_kelembapan_tahunan`, `std_suhu_tahunan`, dan `jumlah_bulan_hujan_tinggi`. Fitur tambahan yang digunakan meliputi `kasus_t-1`, `total_curah_hujan_t-1`, tren perubahan kasus, dan `jumlah_fasilitas_kesehatan`.
4. **Prediksi Jumlah Kasus**
   Model regresi `RandomForestRegressor` digunakan untuk memprediksi jumlah kasus DBD tahun 2026 dan 2027 berdasarkan histori kasus, variabel iklim, dan jumlah fasilitas kesehatan.
5. **Penentuan Label Risiko**
   Label risiko akhir ditentukan dari hasil prediksi jumlah kasus. Jika prediksi kasus mencapai atau melebihi `60` kasus, maka wilayah diberi label `1 = Risiko Tinggi`; jika di bawah `60` kasus, maka diberi label `0 = Risiko Rendah`.
6. **Model Klasifikasi sebagai Pendukung**
   Model klasifikasi `RandomForestClassifier` tetap digunakan untuk membaca sinyal pendukung dari pola historis dan iklim. Nilai `Probabilitas Model` menunjukkan seberapa kuat pola wilayah tersebut menyerupai kelompok risiko tinggi, tetapi tidak menjadi penentu utama label risiko akhir.
7. **Evaluasi Model**
   Seluruh data historis 2023-2025 tetap digunakan dalam pemodelan. Pada evaluasi berbasis waktu, pembentukan fitur dilakukan secara rolling: baris 2024 memakai informasi tahun 2023 sebagai tahun sebelumnya, sedangkan baris 2025 memakai informasi tahun 2024 sebagai tahun sebelumnya. Setelah fitur terbentuk, tahun `{", ".join(map(str, risk_pipeline['train_years']))}` menjadi data latih dan tahun `{risk_pipeline['test_year']}` menjadi data uji. Evaluasi dilakukan dengan metrik AUC, F1-Score, Precision, Recall, serta Spatial Cross-Validation untuk menilai kualitas sinyal pendukung model.
8. **Feature Importance**
   Kepentingan fitur dihitung menggunakan `feature_importances_` bawaan Random Forest. Hasil ini digunakan untuk melihat variabel yang paling berpengaruh terhadap pembentukan sinyal risiko.
9. **Visualisasi Hasil**
   Hasil analisis divisualisasikan dalam bentuk dashboard GIS interaktif berbasis Streamlit dan Folium, yang menampilkan peta risiko, tabel prediksi kelurahan, ringkasan variabel iklim, evaluasi model, dan catatan analisis per wilayah.
"""
            )
        st.caption(
            f"Ambang 'curah hujan tinggi' ditetapkan dari kuartil atas data bulanan, yaitu {risk_pipeline['high_rain_threshold']:.1f} mm."
        )

        st.markdown("#### Dataset Hasil Feature Engineering")
        preview_cols = [
            "tahun",
            "KECAMATAN",
            "KELURAHAN",
            "SPATIAL_BLOCK",
            "kasus_dbd",
            "total_curah_hujan_tahunan",
            "rata_rata_suhu_tahunan",
            "rata_rata_kelembapan_tahunan",
            "std_suhu_tahunan",
            "jumlah_bulan_hujan_tinggi",
            "jumlah_fasilitas_kesehatan",
            "kasus_t_1",
            "total_curah_hujan_t_1",
            "trend_perubahan_kasus",
            "interaksi_hujan_kelembapan",
            "interaksi_suhu_kelembapan",
            "target_risiko",
        ]
        preview_cols = [col for col in preview_cols if col in risk_pipeline["dataset"].columns]
        engineered_preview = risk_pipeline["dataset"].copy()
        if selected_kecamatan != "Semua":
            engineered_preview = engineered_preview[engineered_preview["KECAMATAN"] == selected_kecamatan]
        if selected_kelurahan != "Semua":
            engineered_preview = engineered_preview[engineered_preview["KELURAHAN"] == selected_kelurahan]
        if selected_year_option != "Semua":
            engineered_preview = engineered_preview[engineered_preview["tahun"] == int(selected_year_option)]
        else:
            engineered_preview = engineered_preview[engineered_preview["tahun"].isin(active_years)]
        engineered_preview = engineered_preview[preview_cols].sort_values(
            ["tahun", "KECAMATAN", "KELURAHAN"],
            ascending=[False, True, True],
        )
        if engineered_preview.empty:
            st.info(
                "Tahun 2023 digunakan sebagai data dasar, sehingga belum ditampilkan sebagai baris feature engineering karena belum memiliki data tahun sebelumnya."
            )
        else:
            engineered_display = engineered_preview.rename(
                columns={
                    "tahun": "Tahun",
                    "KECAMATAN": "Kecamatan",
                    "KELURAHAN": "Kelurahan",
                    "SPATIAL_BLOCK": "Blok Spasial",
                    "kasus_dbd": "Kasus DBD",
                    "total_curah_hujan_tahunan": "Total Curah Hujan Tahunan",
                    "rata_rata_suhu_tahunan": "Rata-rata Suhu Tahunan",
                    "rata_rata_kelembapan_tahunan": "Rata-rata Kelembapan Tahunan",
                    "std_suhu_tahunan": "Variasi Suhu Tahunan",
                    "jumlah_bulan_hujan_tinggi": "Jumlah Bulan Hujan Tinggi",
                    "jumlah_fasilitas_kesehatan": "Jumlah Fasilitas Kesehatan",
                    "kasus_t_1": "Kasus t-1",
                    "total_curah_hujan_t_1": "Curah Hujan t-1",
                    "trend_perubahan_kasus": "Tren Perubahan Kasus",
                    "interaksi_hujan_kelembapan": "Interaksi Hujan-Kelembapan",
                    "interaksi_suhu_kelembapan": "Interaksi Suhu-Kelembapan",
                    "target_risiko": "Target Risiko",
                }
            )
            st.dataframe(
                engineered_display,
                use_container_width=True,
                hide_index=True,
                height=280,
            )

        st.markdown("#### Evaluasi Utama Berbasis Waktu")
        ml1, ml2, ml3, ml4 = st.columns(4)
        ml1.metric("AUC", "-" if pd.isna(risk_pipeline["metrics"]["auc"]) else f"{risk_pipeline['metrics']['auc']:.3f}")
        ml2.metric("F1-Score", f"{risk_pipeline['metrics']['f1']:.3f}")
        ml3.metric("Precision", f"{risk_pipeline['metrics']['precision']:.3f}")
        ml4.metric("Recall", f"{risk_pipeline['metrics']['recall']:.3f}")
        st.caption(
            f"Label risiko utama ditentukan dari jumlah kasus: `>= {RISK_CASE_THRESHOLD}` = Risiko Tinggi "
            f"dan `< {RISK_CASE_THRESHOLD}` = Risiko Rendah. Ambang probabilitas model "
            f"`{risk_pipeline['decision_threshold']:.2f}` hanya dipakai untuk evaluasi klasifikasi historis "
            "sebagai sinyal pendukung, bukan sebagai penentu label risiko akhir 2026/2027."
        )
        if risk_pipeline["metrics"]["auc"] >= 0.85 and risk_pipeline["metrics"]["f1"] >= 0.80:
            st.success("Target evaluasi utama tercapai: AUC > 0.85 dan F1-Score > 0.80.")
        else:
            st.warning("Target evaluasi utama belum sepenuhnya tercapai.")

        if spatial_cv.get("status") == "ok":
            spatial_summary = spatial_cv.get("summary", {})
            st.markdown("#### Evaluasi Tambahan: Spatial Cross-Validation")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric(
                "AUC Spasial",
                "-" if pd.isna(spatial_summary.get("AUC")) else f"{spatial_summary['AUC']:.3f}",
            )
            sc2.metric("F1 Spasial", "-" if pd.isna(spatial_summary.get("F1-Score")) else f"{spatial_summary['F1-Score']:.3f}")
            sc3.metric(
                "Precision Spasial",
                "-" if pd.isna(spatial_summary.get("Precision")) else f"{spatial_summary['Precision']:.3f}",
            )
            sc4.metric(
                "Recall Spasial",
                "-" if pd.isna(spatial_summary.get("Recall")) else f"{spatial_summary['Recall']:.3f}",
            )
            st.info(
                "Spatial Cross-Validation ini dipakai untuk melihat kemampuan Random Forest melakukan generalisasi pada wilayah yang berbeda dan mengurangi bias akibat autokorelasi spasial."
            )
            st.dataframe(spatial_cv["folds_df"], use_container_width=True, hide_index=True, height=260)
        elif spatial_cv.get("status") == "base_year":
            st.markdown("#### Evaluasi Tambahan: Spatial Cross-Validation")
            st.info(
                "Tahun 2023 digunakan sebagai data dasar untuk membentuk fitur tahun berikutnya, sehingga Spatial Cross-Validation belum ditampilkan untuk tahun 2023."
            )
        elif selected_year_option != "Semua":
            st.markdown("#### Evaluasi Tambahan: Spatial Cross-Validation")
            st.info(
                f"Spatial Cross-Validation belum dapat ditampilkan untuk tahun {selected_year_option} karena data pada tahun tersebut belum memadai untuk membentuk dua kelas risiko pada setiap fold."
            )

        col_ml1, col_ml2 = st.columns(2)
        with col_ml1:
            st.markdown("#### Hyperparameter Terbaik")
            display_best_params = get_display_model_params(risk_pipeline["best_params"])
            best_params_df = pd.DataFrame(
                {
                    "Parameter": list(display_best_params.keys()),
                    "Nilai": [str(value) for value in display_best_params.values()],
                }
            )
            st.dataframe(best_params_df, use_container_width=True, hide_index=True)
        with col_ml2:
            st.markdown("#### Kepentingan Fitur Model")
            importance_df = risk_pipeline["feature_importance"].head(10).copy()
            st.bar_chart(importance_df.set_index("Feature"), use_container_width=True)
            top_driver_lines = summarize_feature_drivers(risk_pipeline["feature_importance"], limit=5)
            if top_driver_lines:
                st.caption("Faktor yang paling berpengaruh terhadap kenaikan risiko menurut model:")
                for line in top_driver_lines:
                    st.markdown(line)

        st.markdown("#### Hasil Pengujian Model per Wilayah")
        if selected_year_option == "Semua":
            st.caption(
                f"Mode `Semua` menampilkan hasil pengujian tahun {risk_pipeline['test_year']}. Tahun 2023 digunakan sebagai data dasar, tahun 2024 sebagai data latih, dan tahun 2025 sebagai data uji."
            )
        elif model_year_for_display == 2023:
            st.caption(
                "Tahun 2023 digunakan sebagai data dasar untuk membentuk fitur tahun 2024, sehingga belum memiliki tabel hasil pengujian model."
            )
        elif model_year_for_display == int(risk_pipeline["train_years"][0]):
            st.caption(
                f"Tahun {model_year_for_display} digunakan sebagai data latih model klasifikasi, sehingga tidak ditampilkan sebagai tabel hasil pengujian. Hasil pengujian utama ditampilkan pada tahun {risk_pipeline['test_year']}."
            )
        else:
            st.caption(
                "Cara membaca tabel ini: `Kasus DBD Aktual` adalah data asli pada tahun uji, `Risiko Aktual` adalah label asli berdasarkan ambang 60 kasus, `Prediksi Risiko Klasifikasi` adalah hasil evaluasi model klasifikasi historis, dan `Probabilitas Risiko Tinggi` menunjukkan tingkat keyakinan model pada data uji."
            )
        if not show_model_test_table:
            if model_year_for_display == 2023:
                st.info("Tidak ada hasil pengujian model untuk tahun 2023 karena tahun ini menjadi data dasar.")
            else:
                st.info(
                    f"Tidak ada hasil pengujian model untuk tahun {selected_year_option} karena tahun tersebut digunakan sebagai data latih. Pilih tahun {risk_pipeline['test_year']} atau `Semua` untuk melihat hasil pengujian."
                )
        elif test_result_df.empty:
            st.info("Tidak terdapat data uji yang sesuai dengan filter wilayah yang dipilih.")
        else:
            test_display_cols = [
                "tahun",
                "KECAMATAN",
                "KELURAHAN",
                "kasus_dbd",
                "target_label",
                "predicted_risk_score",
                "predicted_risk_label",
                "probabilitas_risiko_tinggi",
            ]
            st.dataframe(
                test_result_df[test_display_cols]
                .rename(
                    columns={
                        "tahun": "Tahun",
                        "KECAMATAN": "Kecamatan",
                        "KELURAHAN": "Kelurahan",
                        "kasus_dbd": "Kasus DBD Aktual",
                        "target_label": "Risiko Aktual",
                        "predicted_risk_score": "Prediksi Skor",
                        "predicted_risk_label": "Prediksi Risiko Klasifikasi",
                        "probabilitas_risiko_tinggi": "Probabilitas Risiko Tinggi",
                    }
                )
                .sort_values(["Probabilitas Risiko Tinggi", "Kasus DBD Aktual"], ascending=[False, False]),
                use_container_width=True,
                hide_index=True,
                height=360,
            )
    elif risk_pipeline.get("status") == "insufficient_train_data":
        st.info("Pipeline klasifikasi belum dapat dijalankan karena data latih belum memadai untuk membentuk dua kelas risiko.")
    else:
        st.info("Pipeline klasifikasi belum dapat dijalankan karena jumlah tahun historis belum memadai.")

    if len(year_cols) >= 3 and "Prediksi 2026" in predicted_df.columns:
        st.markdown("---")
        st.subheader("Prediksi Kasus DBD Tahun 2026 dan 2027")
        st.caption("Prediksi dilakukan menggunakan Random Forest berdasarkan histori kasus dan variabel iklim tahunan per kecamatan.")

        mae = predicted_df["error"].mean() if "error" in predicted_df.columns else 0
        baseline_mean = predicted_df[latest_year_col].mean() if latest_year_col in predicted_df.columns else 0
        accuracy = 100 - (mae / baseline_mean * 100) if baseline_mean else 0
        total_pred_26 = int(predicted_df["Prediksi 2026"].sum())
        total_pred_27 = int(predicted_df["Prediksi 2027"].sum())

        p1, p2, p3, p4 = st.columns(4)
        p1.metric(f"Total Aktual {latest_year_col}", f"{int(predicted_df[latest_year_col].sum()):,}")
        p2.metric("Prediksi 2026", f"{total_pred_26:,}", delta=f"{total_pred_26 - int(predicted_df[latest_year_col].sum()):+,}")
        p3.metric("Prediksi 2027", f"{total_pred_27:,}", delta=f"{total_pred_27 - total_pred_26:+,}")
        p4.metric("Akurasi Aproksimasi", f"{accuracy:.1f}%", help=f"MAE in-sample: {mae:.2f}")

        st.markdown("#### Peta Prediksi Risiko 2026 dan 2027")
        st.markdown("##### Prediksi 2026")
        pred_2026_map = create_prediction_map(
            prediction_map_df,
            "Prediksi 2026",
            "Risiko",
            "2026",
            selected_kecamatan,
            selected_kelurahan,
        )
        st_folium(pred_2026_map, width=None, height=560, returned_objects=[], key="prediction_map_2026")

        st.markdown("##### Prediksi 2027")
        pred_2027_map = create_prediction_map(
            prediction_map_df,
            "Prediksi 2027",
            "Risiko_2027",
            "2027",
            selected_kecamatan,
            selected_kelurahan,
        )
        st_folium(pred_2027_map, width=None, height=560, returned_objects=[], key="prediction_map_2027")

        threshold_value = float(predicted_df["Ambang_Risiko"].iloc[0]) if "Ambang_Risiko" in predicted_df.columns else 0.0
        st.info(
            f"Label risiko utama pada detail kelurahan sekarang mengikuti hasil prediksi kasus dengan ambang tetap `{threshold_value:.1f}` kasus. "
            "Tabel utama menampilkan angka prediksi dari Random Forest Regressor, serta sinyal classifier sebagai informasi pendukung."
        )
        st.caption(
            "Cara membaca tabel detail: angka prediksi adalah estimasi jumlah kasus dari Random Forest Regressor, sedangkan sinyal model adalah indikator pendukung dari Random Forest Classifier."
        )

        st.markdown("#### Detail Prediksi per Kelurahan")
        st.caption(
            "Label risiko mengikuti aturan prediksi kasus `>= 60` = tinggi. Sinyal model dan catatan digunakan sebagai penjelas pendukung."
        )
        prediction_summary_cols = [
            "KECAMATAN",
            "KELURAHAN",
            *year_cols,
            "Prediksi 2026",
            "Risiko",
            "Sinyal_RF_2026",
            "Catatan_Analisis_2026",
            "Prediksi 2027",
            "Risiko_2027",
            "Sinyal_RF_2027",
            "Catatan_Analisis_2027",
        ]
        prediction_summary_df = (
            predicted_df[prediction_summary_cols]
            .rename(
                columns={
                    "KECAMATAN": "Kecamatan",
                    "KELURAHAN": "Kelurahan",
                    "Prediksi 2026": "Prediksi Kasus 2026",
                    "Risiko": "Risiko 2026",
                    "Sinyal_RF_2026": "Sinyal Model 2026",
                    "Catatan_Analisis_2026": "Catatan 2026",
                    "Prediksi 2027": "Prediksi Kasus 2027",
                    "Risiko_2027": "Risiko 2027",
                    "Sinyal_RF_2027": "Sinyal Model 2027",
                    "Catatan_Analisis_2027": "Catatan 2027",
                }
            )
            .sort_values(["Prediksi Kasus 2026", "Prediksi Kasus 2027"], ascending=[False, False])
        )
        st.dataframe(
            prediction_summary_df,
            use_container_width=True,
            hide_index=True,
            height=360,
        )

        st.markdown("#### Fitur Iklim Tahunan yang Digunakan Model")
        model_feature_df = yearly_climate_df[yearly_climate_df["tahun"] == int(latest_year_col)].copy()
        model_feature_df = model_feature_df.merge(build_health_facility_df(dbd_master_df), on="KECAMATAN_KEY", how="left")
        model_feature_df = model_feature_df.rename(
            columns={
                "KECAMATAN_KEY": "Kecamatan",
                "total_curah_hujan_tahunan": "Total Curah Hujan Tahunan (mm)",
                "suhu_rata_rata_tahunan": "Suhu Rata-rata Tahunan (C)",
                "kelembapan_rata_rata_tahunan": "Kelembapan Rata-rata Tahunan (%)",
                "jumlah_fasilitas_kesehatan": "Jumlah Fasilitas Kesehatan",
            }
        )
        st.caption(
            "`Jumlah Fasilitas Kesehatan` digunakan sebagai fitur kesehatan pada model."
        )
        st.dataframe(model_feature_df, use_container_width=True, hide_index=True, height=260)

        trend_data = {str(col): int(predicted_df[col].fillna(0).sum()) for col in year_cols}
        trend_data["2026 (prediksi)"] = total_pred_26
        trend_data["2027 (prediksi)"] = total_pred_27
        trend_df = pd.DataFrame({"Tahun": list(trend_data.keys()), "Total Kasus": list(trend_data.values())})
        st.markdown("#### Tren Historis dan Prediksi")
        st.bar_chart(trend_df.set_index("Tahun"), use_container_width=True)

        top_left, top_right = st.columns(2)
        with top_left:
            st.markdown("#### Sepuluh Wilayah dengan Prediksi Tertinggi Tahun 2026")
            st.caption("Fokus utama: angka `Prediksi 2026`, lalu baca `Label Risiko 2026` sebagai kategori hasil prediksi.")
            st.dataframe(
                predicted_df[
                    [
                        "KECAMATAN",
                        "KELURAHAN",
                        "Prediksi 2026",
                        "Risiko",
                    ]
                ]
                .rename(
                    columns={
                        "KECAMATAN": "Kecamatan",
                        "KELURAHAN": "Kelurahan",
                        "Risiko": "Label Risiko 2026",
                    }
                )
                .sort_values("Prediksi 2026", ascending=False)
                .head(10),
                use_container_width=True,
                hide_index=True,
            )
        with top_right:
            st.markdown("#### Sepuluh Wilayah dengan Prediksi Tertinggi Tahun 2027")
            st.caption("Fokus utama: angka `Prediksi 2027`, lalu baca `Label Risiko 2027` sebagai kategori hasil prediksi.")
            st.dataframe(
                predicted_df[
                    [
                        "KECAMATAN",
                        "KELURAHAN",
                        "Prediksi 2027",
                        "Risiko_2027",
                    ]
                ]
                .rename(
                    columns={
                        "KECAMATAN": "Kecamatan",
                        "KELURAHAN": "Kelurahan",
                        "Risiko_2027": "Label Risiko 2027",
                    }
                )
                .sort_values("Prediksi 2027", ascending=False)
                .head(10),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.subheader("Klasifikasi Risiko DBD dan Rekomendasi Intervensi")
        st.caption("Klasifikasi risiko digunakan untuk mendukung penentuan prioritas wilayah dalam upaya pencegahan dan pengendalian DBD.")

        risk_counts = predicted_df["Risiko"].value_counts()
        r1, r2 = st.columns(2)
        r1.metric("0 = Risiko Rendah", int(risk_counts.get("0 = Risiko Rendah", 0)))
        r2.metric("1 = Risiko Tinggi", int(risk_counts.get("1 = Risiko Tinggi", 0)))

        for level in ["1 = Risiko Tinggi", "0 = Risiko Rendah"]:
            level_df = predicted_df[predicted_df["Risiko"] == level]
            if level_df.empty:
                continue
            with st.expander(f"{level} - {len(level_df)} kelurahan", expanded=level.startswith("1")):
                kel_list = ", ".join(level_df["KELURAHAN"].sort_values().tolist())
                st.markdown(f"**Kelurahan:** {kel_list}")
                st.markdown("**Rekomendasi Intervensi:**")
                for idx, rec in enumerate(get_dbd_recommendations(level), start=1):
                    st.markdown(f"{idx}. {rec}")
    elif len(year_cols) < 3:
        st.markdown("---")
        st.info(
            f"Prediksi belum ditampilkan karena histori aktif baru mencakup {len(year_cols)} tahun ({', '.join(year_cols)}). "
            "Prediksi 2026-2027 akan muncul jika histori minimal 3 tahun tersedia."
        )

    st.markdown("---")
    st.subheader("Ringkasan Hasil Analisis")

    s1, s2, s3, s4 = st.columns(4)
    total_cases_all = int(predicted_df[year_cols].sum().sum()) if year_cols else 0
    avg_cases = round(total_cases_all / total_wilayah, 2) if total_wilayah else 0
    max_cases = int(predicted_df[year_cols].max().max()) if year_cols else 0
    s1.metric("Jumlah Wilayah", total_wilayah)
    s2.metric("Total Kasus Historis", f"{total_cases_all:,}")
    s3.metric("Rata-rata Kasus", avg_cases)
    s4.metric("Kasus Tertinggi", max_cases)

    st.markdown("---")
    st.subheader("Kesimpulan Sementara")
    if len(year_cols) >= 2:
        trend_delta = int(predicted_df[year_cols[-1]].sum() - predicted_df[year_cols[-2]].sum())
        trend_text = "meningkat" if trend_delta > 0 else "menurun" if trend_delta < 0 else "stabil"
        pred_2026_total = int(predicted_df["Prediksi 2026"].sum()) if "Prediksi 2026" in predicted_df.columns else 0
        pred_2027_total = int(predicted_df["Prediksi 2027"].sum()) if "Prediksi 2027" in predicted_df.columns else 0
        high_risk = int((predicted_df["Risk_Score"] == 1).sum()) if "Risk_Score" in predicted_df.columns else 0
        recommendation_text = (
            "Rekomendasi penyuluhan dan intervensi telah disusun berdasarkan tingkat risiko."
            if "Risk_Score" in predicted_df.columns
            else "Rekomendasi berbasis risiko belum tersedia karena hasil klasifikasi belum terbentuk."
        )
        dominant_factors = summarize_feature_drivers(
            risk_pipeline["feature_importance"] if risk_pipeline.get("status") == "ok" else pd.DataFrame(),
            limit=3,
        )
        dominant_factor_text = (
            "Faktor yang paling konsisten memengaruhi risiko pada model saat ini adalah "
            + ", ".join([line.split(". ", 1)[1] for line in dominant_factors])
            + "."
            if dominant_factors
            else "Faktor dominan model belum dapat diringkas karena hasil interpretasi fitur belum tersedia."
        )
        st.success(
            f"""
Berdasarkan hasil analisis dashboard:
- Jumlah wilayah kelurahan yang dipantau sebanyak **{total_wilayah}** wilayah.
- Tren kasus DBD **{trend_text}** sebesar **{abs(trend_delta):,}** kasus dibandingkan tahun sebelumnya.
- Model prediksi berbasis **Random Forest** telah dijalankan untuk tahun **2026** dan **2027**.
- Prediksi total kasus tahun **2026** sebesar **{pred_2026_total:,}** kasus.
- Prediksi total kasus tahun **2027** sebesar **{pred_2027_total:,}** kasus.
- Wilayah dengan kategori **1 = Risiko Tinggi** berjumlah **{high_risk}** kelurahan.
- {dominant_factor_text}
- {recommendation_text}
"""
        )
    else:
        st.success("Dashboard berhasil menampilkan peta sebaran kasus DBD dan variabel iklim pendukung.")

    st.markdown("---")
    d1, d2 = st.columns(2)
    export_climate_df = (
        climate_selected.groupby("KECAMATAN_KEY", as_index=False)[["curah_hujan", "suhu", "kelembapan"]].mean()
    )
    export_df = predicted_df.merge(
        export_climate_df,
        on="KECAMATAN_KEY",
        how="left",
    )
    with d1:
        csv_data = export_df.drop(columns=["SPATIAL_KEY"], errors="ignore").to_csv(index=False).encode("utf-8")
        st.download_button(
            "Unduh Data Hasil Analisis (CSV)",
            data=csv_data,
            file_name="hasil_dashboard_dbd_iklim.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        if st.button("Muat Ulang Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
