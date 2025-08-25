# AI Diagnostic & Prescription Assistant — Demo (Pure Software)

> ⚠️ **Safety & Ethics**  
> This demo is for **educational and hackathon purposes only**. It is **not medical advice**.  
> The medication entries and dosage fields in the knowledge base are **placeholders** and must be replaced with values from trusted clinical guidelines (e.g., WHO, national formularies) and reviewed by a licensed clinician.  
> The app includes *refer-to-provider* guardrails that suppress suggestions for critical symptoms.

## What you get

- Simple NLP classifier (Multinomial Naive Bayes + TF-IDF) trained on a tiny, toy dataset.
- A local knowledge base (`src/knowledge_base/guidelines_demo.json`) with **placeholder** first-line medication entries and dosage *templates*.
- Safety checks: red-flag symptom detection, allergy checks, and a "do not show meds" mode for critical inputs.
- Streamlit UI with explainability (keywords that influenced the prediction) and a feedback logger.

## Quick Start

### 1) Create a virtual environment and install
```bash
# from inside the project folder
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Train the toy model
```bash
python src/model/train.py
```
This will create `models/vectorizer.pkl` and `models/classifier.pkl`.

### 3) Run the app
```bash
streamlit run src/app/streamlit_app.py
```
Open the provided local URL in your browser.

## Folder Structure
```
ai_clinical_assistant/
├─ data/
│  └─ toy_symptoms.csv
├─ models/                # created after training
├─ src/
│  ├─ app/
│  │  └─ streamlit_app.py
│  ├─ knowledge_base/
│  │  └─ guidelines_demo.json
│  ├─ model/
│  │  ├─ __init__.py
│  │  ├─ inference.py
│  │  └─ train.py
│  ├─ safety.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## How to replace placeholder dosing
Edit `src/knowledge_base/guidelines_demo.json` and fill each `dose` object with values from **trusted sources**.  
The UI clearly labels demo entries. When you set `demo_only: false` for an entry, the UI will show calculated doses (if weight-based) and details.

## Disclaimers
- Do **not** deploy without clinical review and proper compliance checks.
- The tiny dataset is for demo. Replace it with curated data and re-train.
