"""Configuration — paths, feature lists, constants."""
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent                # …/pidilite_ai/
LOCAL_DATA  = PROJECT_DIR / "data"

if LOCAL_DATA.exists() and (LOCAL_DATA / "batch_feature_table.xlsx").exists():
    DATA_DIR = LOCAL_DATA
    BASE_DIR = LOCAL_DATA
else:
    BASE_DIR = Path(__file__).resolve().parent.parent          # …/Pidilite/
    DATA_DIR = BASE_DIR / "Pidilite_Data pipeline"

# ── Cleaned data files ───────────────────────────────────────────────────
BATCH_FEATURE_TABLE = DATA_DIR / "batch_feature_table.xlsx"
REACTOR_BCT         = DATA_DIR / "cleaned_reactor_bct.xlsx"
BLENDER_BCT         = DATA_DIR / "cleaned_blender_bct.xlsx"
QC_RESULTS          = DATA_DIR / "cleaned_qc_results.xlsx"
TAG_FILE            = (DATA_DIR / "Vizag_1_Tags_Batch.xlsx") if (DATA_DIR / "Vizag_1_Tags_Batch.xlsx").exists() else (BASE_DIR / "Vizag_1_Tags_Batch.xlsx")

# ── Reactor step columns (union of Product-A and Product-B) ──────────────
# These are the step-duration columns present in cleaned_reactor_bct.xlsx.
# Product-A uses "Seeding" / "Continous feed -mono";
# Product-B uses "Seed operation" / "Continous feeding- mono" / "RM-8 Charge".
REACTOR_STEP_COLS = [
    "Start of Batch",
    "Charge Pre Intermediate",
    "Temprature adjustment",
    "RM-1 (L) charge in reactor",
    "RM-2 (S) charge in reactor",
    "RM-3 (S) charge in reactor",
    "RM-4 (S) charge in reactor",
    "Seeding",                       # Product-A name
    "Seed operation",                # Product-B name
    "Holding-1",
    "RM-5 charge",
    "Continous feed -mono",          # Product-A name
    "Continous feeding- mono",       # Product-B name
    "Wait for set  temperature reach",
    "RM-6 charge",
    "Holding-2",
    "RM-7 Charge",
    "RM-7 Charge.1",                 # Product-A RM-8 (mislabeled)
    "RM-8 Charge",                   # Product-B RM-8
    "Holding-3",
    "Sampling",
    "Transfer to blender",
]

# Columns to drop (zero variance or 100% empty)
DROP_COLS = [
    "Start of Batch",   # constant 5 min
    "Holding-3",        # constant 30 min
    "Sampling",         # 100% null
]

# Summary columns (not individual steps)
SUMMARY_COLS = ["Total BCT", "Std Production\n(Batch Size)", "Act Production"]

# ── Quality target columns (from batch_feature_table) ────────────────────
IPQC_TARGETS = ["reactor_ipqc_solid_pct", "reactor_ipqc_viscosity"]
SFG_TARGETS  = ["reactor_sfg_solid_pct", "reactor_sfg_viscosity",
                "reactor_sfg_residual_monomer_pct", "reactor_ph"]

# ── Spec limits (approximate — derived from data distributions) ──────────
SPEC_LIMITS = {
    "Product-A": {
        "reactor_ipqc_solid_pct":  (50.0, 52.0),
        "reactor_ipqc_viscosity":  (230, 380),
    },
    "Product-B": {
        "reactor_ipqc_solid_pct":  (52.0, 55.0),
        "reactor_ipqc_viscosity":  (600, 1150),
    },
}

# ── MLflow ───────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"file:///{(PROJECT_DIR / 'mlruns').as_posix()}"
MLFLOW_EXPERIMENT   = "pidilite_quality_prediction"

# ── Model artifacts ─────────────────────────────────────────────────────
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
