# Quickstart Guide: Pidilite AI Quality Intelligence Platform

This project implements an end-to-end industrial ML intelligence platform combining 9 targeted projects across quality modeling, cycle time optimization, transfer learning, root cause analysis, process mining, and AI-driven explanations.

---

## 1. Directory Structure (`pidilite_ai/`)

- [`app.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/app.py): **Streamlit Web Application** hosting all 9 interactive tabs.
- [`api.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/api.py): **FastAPI REST Service** serving real-time predictions and SHAP driver contributions.
- [`data_pipeline.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/data_pipeline.py): Production ETL pipeline, coalescing features across products and engineering domain indicators.
- [`quality_models.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/quality_models.py): XGBoost, LightGBM, Random Forest, Ridge regression + SHAP explainability + MLflow tracking.
- [`bct_analysis.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/bct_analysis.py): Cycle-time variance ranking, correlation heatmaps, variance decomposition.
- [`golden_batch.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/golden_batch.py): Reference golden profiling, Euclidean similarity scores, 2D PCA & t-SNE embeddings.
- [`transfer_analysis.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/transfer_analysis.py): Transfer to blender bottleneck diagnosis, fast/slow k-means segmentation, survival curves.
- [`transfer_learning.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/transfer_learning.py): Cross-product transfer learning (Zero-shot Product-A to B vs few-shot fine-tuning).
- [`process_mining.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/process_mining.py): Step sequence Gantt analysis, bottleneck ranking, what-if capacity simulations.
- [`explanation_agent.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/explanation_agent.py): Gemini LLM operator agent (with deterministic rule-based fallback).
- [`config.py`](file:///C:/Users/AJhaIIT/Downloads/Pidilite-20260917T170505Z-1-001/Pidilite/pidilite_ai/config.py): Target specs, column taxonomies, and paths.

---

## 2. Launching the Streamlit Interactive Dashboard

In PowerShell/terminal:
```powershell
cd "C:\Users\AJhaIIT\Downloads\Pidilite-20260917T170505Z-1-001\Pidilite\pidilite_ai"
py -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Features by Tab:
1. **📊 Overview & EDA**: Data summary distributions, box plots, batch timelines for Product-A & Product-B.
2. **📈 BCT Variance**: Rank steps by CV%, variance decomposition pie charts, step vs. quality correlation matrix.
3. **🏆 Golden Batch**: Compare golden profiles, similarity distribution, and PCA / t-SNE 2D cluster maps.
4. **🤖 Quality Prediction**: Train XGBoost/LightGBM/RF/Ridge, evaluate test metrics (MAPE, RMSE, R²), view SHAP importance, and test interactive single-batch simulations.
5. **🔄 Transfer Learning**: Benchmark zero-shot vs few-shot fine-tuning across products.
6. **⏱️ Transfer Time**: Statistical distribution and root-cause analysis on transfer bottleneck.
7. **⚙️ Process Mining**: Batch step Gantt charts, bottleneck impact scoring, and what-if cycle-time reduction simulations.
8. **💬 AI Explainer**: Enter your Gemini API key in the sidebar to get instant natural language operator explanations with root-cause recommendations.
9. **🔧 Pipeline & API**: MLflow run registry, architecture diagrams, and REST usage instructions.

---

## 3. Running the FastAPI REST Server

In a separate terminal window:
```powershell
cd "C:\Users\AJhaIIT\Downloads\Pidilite-20260917T170505Z-1-001\Pidilite\pidilite_ai"
py api.py
```
- Swagger UI Documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Predict endpoint: `POST http://localhost:8000/predict`
