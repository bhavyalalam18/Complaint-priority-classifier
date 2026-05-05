import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from deep_translator import GoogleTranslator

def translate_to_english(text: str) -> str:
    """Translate any language to English."""
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception:
        return text

# ---------------------------------------------------------------------------
# Priority keywords
# ---------------------------------------------------------------------------
PRIORITY_KEYWORDS = {
    "high": [
        "down", "crash", "breach", "critical", "urgent", "failure",
        "outage", "data loss", "broken", "emergency", "production",
        "corrupted", "locked out", "unresponsive", "billing", "exposed",
        "unavailable", "bouncing", "recovery", "freeze", "missing data",
        "not accessible", "cannot access", "service down", "all users",
        "complete", "entire", "major", "severe", "immediately",
        "everyone", "no one", "all customers", "total", "wiped"
    ],
    "medium": [
        "slow", "delayed", "intermittent", "sometimes", "occasional",
        "error", "incorrect", "issue", "problem", "not working",
        "freezes", "fails", "unable", "missing", "timeout", "inconsistent",
        "some users", "certain", "for some", "not arriving", "not loading",
        "not received", "not saving", "not refreshing", "not generating",
        "stale", "blank screen", "occasionally"
    ],
    "low": [
        "typo", "color", "cosmetic", "minor", "suggestion", "request",
        "alignment", "tooltip", "dark mode", "label", "spacing", "icon",
        "polish", "onboarding", "improve", "add option", "feature request",
        "font", "padding", "placeholder", "text overlap", "brand",
        "guideline", "animation", "keyboard shortcut", "empty state",
        "help text", "footer", "copyright", "reorder", "grammatical",
        "blurry", "contrast", "hover", "tab order", "confirmation"
    ]
}

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------------------------
# Keyword scorer transformer
# ---------------------------------------------------------------------------
class KeywordScorer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        results = []
        for text in X:
            t = preprocess(text)
            scores = [
                sum(kw in t for kw in PRIORITY_KEYWORDS["high"]),
                sum(kw in t for kw in PRIORITY_KEYWORDS["medium"]),
                sum(kw in t for kw in PRIORITY_KEYWORDS["low"]),
            ]
            total = sum(scores) or 1
            results.append([s / total for s in scores])
        return np.array(results)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(min_class_count: int = 5) -> Pipeline:
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            max_features=8000,
            sublinear_tf=True,
            stop_words="english",
            min_df=1
        )),
        ("keywords", KeywordScorer())
    ])

    calibration_cv = max(2, min(5, min_class_count))

    svm = CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced", max_iter=2000),
        cv=calibration_cv
    )
    return Pipeline([("features", features), ("clf", svm)])

# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------
def keyword_priority(text: str) -> tuple[str, float]:
    t = preprocess(text)
    scores = {p: sum(kw in t for kw in kws) for p, kws in PRIORITY_KEYWORDS.items()}
    total = sum(scores.values())
    best = max(scores, key=scores.get)
    if total == 0:
        return "medium", 0.40
    return best, round(min(scores[best] / total, 0.75), 3)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(csv_path: str, model_path: str = "models/model.joblib"):
    # ✅ FIX 1: explicit header=0 so pandas never treats header as data
    df = pd.read_csv(csv_path, header=0)
    df.dropna(subset=["text", "priority"], inplace=True)
    df["priority"] = df["priority"].str.lower().str.strip()

    # ✅ FIX 2: drop any row where priority is not a valid label
    df = df[df["priority"].isin(["high", "medium", "low"])]

    X, y = df["text"], df["priority"]
    print(f"\n📦 Dataset: {len(df)} samples | Classes: {dict(y.value_counts().sort_index())}")

    min_class_count = int(y.value_counts().min())

    # ✅ FIX 3: calibration_cv must be safe within each outer fold
    # Each fold's train split is (n_splits-1)/n_splits of data
    # so effective min samples per class in a fold is min_class_count * (n_splits-1)/n_splits
    n_splits = max(2, min(5, min_class_count))
    effective_min = int(min_class_count * (n_splits - 1) / n_splits)
    calibration_cv = max(2, min(5, effective_min))

    pipeline = build_pipeline(min_class_count=calibration_cv)

    print(f"   Using n_splits={n_splits} (smallest class has {min_class_count} samples)")
    print(f"   Calibration cv={calibration_cv} (effective min per fold={effective_min})")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"\n📈 Cross-validation accuracy : {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
    print(f"   Per-fold scores           : {[f'{s:.2%}' for s in cv_scores]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n✅ Hold-out Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    labels = ["high", "medium", "low"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("🔢 Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':10}  {'high':>6}  {'medium':>6}  {'low':>6}")
    for lbl, row in zip(["high", "medium", "low"], cm):
        print(f"  {lbl:<8}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}")

    # Refit on full data before saving
    pipeline.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Model saved to {model_path}")
    return pipeline

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(text: str, model_path: str = "models/model.joblib", translate: bool = False) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")

    translated_text = translate_to_english(text) if translate else text

    pipeline = joblib.load(model_path)
    proba    = pipeline.predict_proba([translated_text])[0]
    classes  = pipeline.classes_

    ml_label      = classes[np.argmax(proba)]
    ml_confidence = float(np.max(proba))
    kw_label, kw_confidence = keyword_priority(translated_text)

    if ml_confidence >= 0.6:
        final_label, final_confidence, source = ml_label, ml_confidence, "model"
    elif kw_confidence >= 0.5:
        final_label, final_confidence, source = kw_label, kw_confidence, "keyword_fallback"
    else:
        final_label, final_confidence, source = ml_label, ml_confidence, "model_low_confidence"

    return {
        "text"           : text,
        "translated_text": translated_text if translate else None,
        "priority"       : final_label,
        "confidence"     : round(final_confidence, 3),
        "source"         : source,
        "scores"         : dict(zip(classes, map(float, proba)))
    }