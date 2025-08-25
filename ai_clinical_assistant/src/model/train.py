import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "toy_symptoms.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = df["symptoms_text"].astype(str).tolist()
    y = df["disease"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    # Save the full pipeline to keep it simple
    model_path = os.path.join(MODEL_DIR, "text_clf.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline to {model_path}")

if __name__ == "__main__":
    main()
