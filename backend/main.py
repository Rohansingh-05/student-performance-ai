from pathlib import Path
import json, os
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "student_performance_dataset.csv"
FEATURES = ["attendance_percentage", "assignment_score", "midterm_score", "practical_score", "final_exam_score"]

app = FastAPI(title="Student Performance AI API", version="1.0.0")
origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(CORSMiddleware, allow_origins=["*"] if origin == "*" else [origin],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

_artifacts = None
def load_artifacts():
    global _artifacts
    if _artifacts is None:
        required = ["decision_tree.pkl", "random_forest.pkl", "imputer.pkl", "evaluation_results.json"]
        missing = [name for name in required if not (MODELS / name).exists()]
        if missing:
            raise RuntimeError("Copy existing files into backend/models/: " + ", ".join(missing))
        _artifacts = {
            "decision_tree": joblib.load(MODELS / "decision_tree.pkl"),
            "random_forest": joblib.load(MODELS / "random_forest.pkl"),
            "imputer": joblib.load(MODELS / "imputer.pkl"),
            "evaluation": json.loads((MODELS / "evaluation_results.json").read_text(encoding="utf-8"))
        }
    return _artifacts

class Student(BaseModel):
    student_id: str = Field(default="NEW001", max_length=64)
    attendance_percentage: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)
    midterm_score: float = Field(ge=0, le=100)
    practical_score: float = Field(ge=0, le=100)
    final_exam_score: float = Field(ge=0, le=100)

class Batch(BaseModel):
    students: list[Student] = Field(min_length=1, max_length=500)

def row_for(student):
    return pd.DataFrame([{k: getattr(student, k) for k in FEATURES}])

def infer(model, row, imputer):
    x = imputer.transform(row[FEATURES])
    label = str(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]
    return label, {str(c): round(float(p) * 100, 2) for c, p in zip(model.classes_, probabilities)}

@app.get("/")
def root():
    return {"message": "Student Performance AI API", "health": "/health", "docs": "/docs"}

@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok", "artifacts_loaded": True}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/metrics")
def metrics():
    try: return load_artifacts()["evaluation"]
    except Exception as exc: raise HTTPException(status_code=503, detail=str(exc))

@app.get("/feature-importance")
def feature_importance():
    try:
        a = load_artifacts()
        output = {}
        for key in ["decision_tree", "random_forest"]:
            model = a[key]
            values = getattr(model, "feature_importances_", [])
            output[key] = [{"feature": f, "importance": round(float(v), 6)}
                           for f, v in sorted(zip(FEATURES, values), key=lambda x: x[1], reverse=True)]
        return output
    except Exception as exc: raise HTTPException(status_code=503, detail=str(exc))

@app.get("/dataset-summary")
def dataset_summary():
    if not DATA.exists(): return {"available": False}
    df = pd.read_csv(DATA)
    return {"available": True, "rows": int(len(df)), "columns": int(len(df.columns)),
            "average_attendance": round(float(df.attendance_percentage.mean()), 2),
            "average_academic_score": round(float(df[FEATURES[1:]].mean().mean()), 2),
            "class_distribution": {str(k): int(v) for k, v in df.performance_label.value_counts().items()}}

@app.post("/predict")
def predict(student: Student):
    try:
        a = load_artifacts()
        row = row_for(student)
        dt, dtp = infer(a["decision_tree"], row, a["imputer"])
        rf, rfp = infer(a["random_forest"], row, a["imputer"])
        scores = {f: float(row.iloc[0][f]) for f in FEATURES}
        weak = [f.replace("_", " ").title() for f, v in scores.items() if v < 50]
        tips = []
        if scores["attendance_percentage"] < 75: tips.append("Review attendance barriers and agree on a realistic attendance improvement plan.")
        if scores["assignment_score"] < 50: tips.append("Provide assignment feedback, guided practice, and a short follow-up review.")
        if scores["midterm_score"] < 50 or scores["final_exam_score"] < 50: tips.append("Identify topic-level gaps and provide targeted revision or tutoring.")
        if scores["practical_score"] < 50: tips.append("Offer additional supervised practical exercises.")
        if not tips: tips.append("Continue monitoring progress and reinforce effective study habits.")
        return {"student_id": student.student_id,
                "decision_tree": {"prediction": dt, "probabilities": dtp},
                "random_forest": {"prediction": rf, "probabilities": rfp},
                "models_agree": dt == rf, "weak_areas": weak, "support_recommendations": tips,
                "notice": "Decision-support only. Probabilities are model estimates, not guarantees or final academic judgments."}
    except Exception as exc: raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

@app.post("/batch-predict")
def batch_predict(batch: Batch):
    try:
        a = load_artifacts()
        rows = []
        for student in batch.students:
            row = row_for(student)
            dt, _ = infer(a["decision_tree"], row, a["imputer"])
            rf, probs = infer(a["random_forest"], row, a["imputer"])
            rows.append({"student_id": student.student_id, "decision_tree_prediction": dt,
                         "random_forest_prediction": rf, "models_agree": dt == rf,
                         "random_forest_confidence": max(probs.values())})
        return {"count": len(rows), "predictions": rows}
    except Exception as exc: raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}")
