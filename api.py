"""
FastAPI Prediction Endpoint — Project 15.
Serves trained quality prediction models via REST API.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import quality_models as qm
import config as cfg

app = FastAPI(
    title="Pidilite AI — Quality Prediction API",
    description="Real-time quality prediction for PVAc emulsion polymerization batches.",
    version="1.0.0",
)

# ── Cached models ────────────────────────────────────────────────────
_models: dict = {}


def _get_or_train_model(product: str, target: str, model_type: str = "xgboost"):
    """Get cached model or train a new one."""
    key = f"{product}_{target}_{model_type}"
    if key not in _models:
        _models[key] = qm.train_model(product, target, model_type)
    return _models[key]


# ── Request / Response schemas ───────────────────────────────────────
class PredictionRequest(BaseModel):
    product: str = "Product-A"
    target: str = "reactor_ipqc_viscosity"
    model_type: str = "xgboost"
    features: dict

    class Config:
        json_schema_extra = {
            "example": {
                "product": "Product-A",
                "target": "reactor_ipqc_viscosity",
                "model_type": "xgboost",
                "features": {
                    "Charge Pre Intermediate": 27,
                    "Temprature adjustment": 17,
                    "Holding-1": 16,
                    "Continous feed -mono": 289,
                    "Transfer to blender": 225,
                    "Total BCT": 724,
                    "Act Production": 29500,
                },
            }
        }


class PredictionResponse(BaseModel):
    prediction: float
    risk_level: str
    top_drivers: list
    product: str
    target: str
    model_type: str


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Pidilite AI Quality Prediction API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Predict quality for a single batch given step durations."""
    try:
        model_result = _get_or_train_model(req.product, req.target, req.model_type)
        result = qm.predict_single(model_result, req.features)
        return PredictionResponse(
            prediction=round(float(result["prediction"]), 4),
            risk_level=result["risk_level"],
            top_drivers=result["top_drivers"],
            product=req.product,
            target=req.target,
            model_type=req.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List all currently cached (trained) models."""
    return {
        "models": [
            {
                "key": k,
                "product": v["product"],
                "target": v["target"],
                "model_type": v["model_type"],
                "metrics": v["metrics"],
            }
            for k, v in _models.items()
        ]
    }


if __name__ == "__main__":
    print("Starting Pidilite AI API on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
