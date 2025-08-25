import os, json, joblib
import numpy as np
from typing import Dict, List, Tuple
from ..utils import normalize_text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "text_clf.pkl")
KB_PATH = os.path.join(PROJECT_ROOT, "src", "knowledge_base", "guidelines_demo.json")

class Predictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Please run: python src/model/train.py")
        self.pipeline = joblib.load(MODEL_PATH)
        with open(KB_PATH, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    def predict(self, symptom_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        proba = None
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            proba = self.pipeline.predict_proba([symptom_text])[0]
            classes = self.pipeline.named_steps["clf"].classes_
            ranked = sorted(list(zip(classes, proba)), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        # fallback: use decision_function-like ranking via predict_log_proba if available
        if hasattr(self.pipeline.named_steps["clf"], "predict_log_proba"):
            logp = self.pipeline.named_steps["clf"].predict_log_proba([symptom_text])[0]
            classes = self.pipeline.named_steps["clf"].classes_
            ranked = sorted(list(zip(classes, np.exp(logp))), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]
        # last resort: return predicted class with prob=1.0
        c = self.pipeline.predict([symptom_text])[0]
        return [(c, 1.0)]

    def explanation_keywords(self, symptom_text: str, top_n: int = 8) -> List[str]:
        # crude "matched keywords": intersect input tokens with KB keywords for top diseases
        toks = set(normalize_text(symptom_text).split())
        keywords = set()
        for d, info in self.kb.get("disease_keywords", {}).items():
            for k in info:
                if k in toks:
                    keywords.add(k)
        return list(sorted(keywords))[:top_n]

    def meds_for_disease(self, disease: str) -> List[Dict]:
        return self.kb.get("guidelines", {}).get(disease, {}).get("recommended_meds", [])

    def compute_dose(self, dose_obj: Dict, weight_kg: float | None):
        if dose_obj is None:
            return None
        if dose_obj.get("demo_only", True):
            return None
        per_kg = dose_obj.get("per_kg_mg")
        fixed_mg = dose_obj.get("fixed_mg")
        freq = dose_obj.get("frequency")
        dur = dose_obj.get("duration_days")

        if per_kg is not None and weight_kg:
            total_mg = round(per_kg * float(weight_kg), 1)
        elif fixed_mg is not None:
            total_mg = fixed_mg
        else:
            total_mg = None
        return {
            "dose_mg": total_mg,
            "frequency": freq,
            "duration_days": dur
        }
