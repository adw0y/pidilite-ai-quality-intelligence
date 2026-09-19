"""
Data Pipeline — ETL module (Project 15).
Loads cleaned Excel files, engineers features, returns ready-to-use DataFrames.
"""
import pandas as pd
import numpy as np
from functools import lru_cache
import config as cfg


# ─────────────────────────────────────────────────────────────────────────
# 1. Raw loaders
# ─────────────────────────────────────────────────────────────────────────
def _clean_excel_dates(series: pd.Series) -> pd.Series:
    """Convert mixed string / serial integer dates from Excel into datetime."""
    def _parse(val):
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, (int, float)):
            # Excel base date is 1899-12-30
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(val, unit="D")
        return pd.to_datetime(val, errors="coerce")
    return series.apply(_parse)


@lru_cache(maxsize=1)
def load_feature_table() -> pd.DataFrame:
    """Load the model-ready batch feature table (200 × 24)."""
    df = pd.read_excel(cfg.BATCH_FEATURE_TABLE)
    if "reactor_date" in df.columns:
        df["reactor_date"] = _clean_excel_dates(df["reactor_date"])
    if "blender_date" in df.columns:
        df["blender_date"] = _clean_excel_dates(df["blender_date"])
    return df


@lru_cache(maxsize=1)
def load_reactor_bct() -> pd.DataFrame:
    """Load cleaned reactor BCT step durations (200 × 29)."""
    df = pd.read_excel(cfg.REACTOR_BCT)
    return df


@lru_cache(maxsize=1)
def load_blender_bct() -> pd.DataFrame:
    """Load cleaned blender BCT step durations (200 × 39)."""
    df = pd.read_excel(cfg.BLENDER_BCT)
    return df


@lru_cache(maxsize=1)
def load_qc_results() -> pd.DataFrame:
    """Load cleaned QC results in long format (400 × 10)."""
    df = pd.read_excel(cfg.QC_RESULTS)
    return df


@lru_cache(maxsize=1)
def load_tags() -> dict[str, pd.DataFrame]:
    """Load tag file — returns dict with 'bct_tags' and 'process_tags'."""
    bct  = pd.read_excel(cfg.TAG_FILE, sheet_name="Batch_BCT_Tag", header=1)
    proc = pd.read_excel(cfg.TAG_FILE, sheet_name="Process_Tag",   header=1)
    return {"bct_tags": bct, "process_tags": proc}


# ─────────────────────────────────────────────────────────────────────────
# 2. Unified step column resolver
# ─────────────────────────────────────────────────────────────────────────
def _unify_step_name(col: str) -> str:
    """Map product-specific step names to a unified name."""
    mapping = {
        "Seed operation":           "Seeding",
        "Continous feeding- mono":  "Continous feed -mono",
        "RM-7 Charge.1":            "RM-8 Charge",
    }
    return mapping.get(col, col)


def get_numeric_step_cols(df: pd.DataFrame) -> list[str]:
    """Return the numeric step-duration columns (exclude meta cols)."""
    meta = {"product", "date", "product_name", "batch_no"}
    return [c for c in df.columns
            if c not in meta and df[c].dtype in ("float64", "int64")
            and c not in cfg.DROP_COLS]


# ─────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────────────────
def build_reactor_features(product: str | None = None) -> pd.DataFrame:
    """
    Build an enriched reactor feature table for modeling.

    Merges step durations from cleaned_reactor_bct with quality targets
    from batch_feature_table, then engineers additional features.
    """
    bct = load_reactor_bct().copy()
    ft  = load_feature_table().copy()

    # Combine / coalesce alternate step names so both products have common columns
    if "Seed operation" in bct.columns:
        if "Seeding" in bct.columns:
            bct["Seeding"] = bct["Seeding"].fillna(bct["Seed operation"])
            bct.drop(columns=["Seed operation"], inplace=True)
        else:
            bct.rename(columns={"Seed operation": "Seeding"}, inplace=True)

    if "Continous feeding- mono" in bct.columns:
        if "Continous feed -mono" in bct.columns:
            bct["Continous feed -mono"] = bct["Continous feed -mono"].fillna(bct["Continous feeding- mono"])
            bct.drop(columns=["Continous feeding- mono"], inplace=True)
        else:
            bct.rename(columns={"Continous feeding- mono": "Continous feed -mono"}, inplace=True)

    if "RM-7 Charge.1" in bct.columns:
        if "RM-8 Charge" in bct.columns:
            bct["RM-8 Charge"] = bct["RM-8 Charge"].fillna(bct["RM-7 Charge.1"])
            bct.drop(columns=["RM-7 Charge.1"], inplace=True)
        else:
            bct.rename(columns={"RM-7 Charge.1": "RM-8 Charge"}, inplace=True)

    # Merge quality targets onto BCT
    merged = bct.merge(
        ft[["product", "batch_no"] + cfg.IPQC_TARGETS + cfg.SFG_TARGETS],
        on=["product", "batch_no"],
        how="left",
    )

    # ── Engineered features ──
    merged["yield_ratio"] = merged["Act Production"] / merged["Std Production\n(Batch Size)"]

    # Total holding time
    hold_cols = [c for c in merged.columns if "Holding" in c and c not in cfg.DROP_COLS]
    merged["holding_total"] = merged[hold_cols].sum(axis=1)

    # Total charge time
    charge_cols = [c for c in merged.columns if c.startswith("RM-")]
    merged["charge_total"] = merged[charge_cols].sum(axis=1)

    # Continuous feed column (unified)
    feed_col = "Continous feed -mono"
    if feed_col in merged.columns:
        merged["reaction_phase_pct"] = merged[feed_col] / merged["Total BCT"]
        merged["transfer_to_bct_ratio"] = merged["Transfer to blender"] / merged["Total BCT"]

    # BCT z-score (per product)
    for prod in merged["product"].unique():
        mask = merged["product"] == prod
        bct_vals = merged.loc[mask, "Total BCT"]
        merged.loc[mask, "bct_z_score"] = (bct_vals - bct_vals.mean()) / bct_vals.std()

    # Filter by product if requested
    if product:
        merged = merged[merged["product"] == product].reset_index(drop=True)

    return merged


def get_modeling_xy(product: str, target: str = "reactor_ipqc_viscosity"):
    """
    Return X (features) and y (target) ready for sklearn.

    Drops non-feature columns (dates, product name, quality targets, etc.)
    and returns only numeric feature columns.
    """
    df = build_reactor_features(product)

    # Columns to exclude from X
    exclude = (
        {"product", "date", "product_name", "batch_no", "Std Production\n(Batch Size)"}
        | set(cfg.IPQC_TARGETS)
        | set(cfg.SFG_TARGETS)
        | set(cfg.DROP_COLS)
        | {"reactor_appearance"}
    )

    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and df[c].dtype in ("float64", "int64")]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Drop rows with NaN targets
    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # Fill NaN features with 0 (product-specific columns like Seed operation)
    X = X.fillna(0)

    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────
# 4. Convenience
# ─────────────────────────────────────────────────────────────────────────
def load_all():
    """Load all datasets into a dict for convenience."""
    return {
        "feature_table": load_feature_table(),
        "reactor_bct":   load_reactor_bct(),
        "blender_bct":   load_blender_bct(),
        "qc_results":    load_qc_results(),
        "tags":          load_tags(),
    }


if __name__ == "__main__":
    print("Loading all datasets...")
    data = load_all()
    for name, df in data.items():
        if isinstance(df, dict):
            for k, v in df.items():
                print(f"  {name}.{k}: {v.shape}")
        else:
            print(f"  {name}: {df.shape}")

    print("\nBuilding reactor features for Product-A...")
    rf = build_reactor_features("Product-A")
    print(f"  Shape: {rf.shape}")
    print(f"  Columns: {list(rf.columns)}")

    print("\nGetting modeling X, y for Product-A viscosity...")
    X, y, cols = get_modeling_xy("Product-A", "reactor_ipqc_viscosity")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Features: {cols}")
