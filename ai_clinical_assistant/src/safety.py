from typing import List, Dict
from .utils import normalize_text

RED_FLAG_PATTERNS = [
    "chest pain",
    "shortness of breath",
    "severe headache",
    "unconscious",
    "confusion",
    "stiff neck",
    "fever 40",    # temp >=40C mentioned
    "bloody stool",
    "vomiting blood",
    "seizure",
    "pregnant with pain",
]

def detect_red_flags(symptoms_text: str) -> List[str]:
    t = normalize_text(symptoms_text)
    matches = []
    for pat in RED_FLAG_PATTERNS:
        if pat in t:
            matches.append(pat)
    return matches

def allergy_filter(med_list: List[Dict], allergies_csv: str) -> List[Dict]:
    if not allergies_csv:
        return med_list
    allergies = [a.strip().lower() for a in allergies_csv.split(",") if a.strip()]
    filtered = []
    for med in med_list:
        name = med.get("name", "").lower()
        # if any allergy token appears in med name or class notes, flag it
        med_copy = dict(med)
        med_copy["allergy_flag"] = any(a in name or a in (med.get("class","").lower()) for a in allergies)
        filtered.append(med_copy)
    return filtered
