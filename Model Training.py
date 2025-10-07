import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

SUSPICIOUS_KEYWORDS = {"urgent", "verify", "confirm", "click", "immediately", "locked",
                       "password", "bank", "account", "suspended", "risk", "security",
                       "claim", "winner"}
TRUSTED_DOMAINS = ["bank", "facebook", "google", "microsoft"]
URL_REGEX = re.compile(r"https?://[^\s\"'>]+")

def extract_urls(text: str):
    return URL_REGEX.findall(text or "")

def get_domain(email: str):
    if not email or "@" not in email:
        return None
    return email.split("@")[-1].lower()

def is_domain_trusted(domain: str):
    if not domain:
        return False
    return any(td in domain for td in TRUSTED_DOMAINS)

class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key: str):
        self.key = key
    def fit(self, X, y=None): 
        return self
    def transform(self, X): 
        return X[self.key].fillna("").values

class NumericFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        features = []
        for _, row in X.iterrows():
            text = (row.get("subject") or "") + " " + (row.get("email_text") or "")
            urls = extract_urls(text)
            url_count = len(urls)
            words = re.findall(r"\b[a-z0-9]+\b", text.lower())
            keyword_count = sum(1 for w in words if w in SUSPICIOUS_KEYWORDS)
            sender_domain = get_domain(row.get("sender"))
            sender_trust = 1 if is_domain_trusted(sender_domain) else 0
            features.append([url_count, keyword_count, sender_trust])
        return np.array(features)

def load_sample_data():
    data = [
        {"sender": "alerts@secure-bank.com", "subject": "Your account locked",
         "email_text": "Click http://fakebank.com to verify", "label": 1},
        {"sender": "friend@gmail.com", "subject": "Lunch plans",
         "email_text": "Hey, wanna grab lunch today?", "label": 0},
        {"sender": "team@college.edu", "subject": "Project meeting",
         "email_text": "Reminder: meeting at 3 PM", "label": 0},
        {"sender": "support@microsoft.com", "subject": "Virus detected",
         "email_text": "Urgent! Call +1-800-123 immediately", "label": 1}
    ]
    return pd.DataFrame(data)

def build_pipeline():
    text_pipeline = Pipeline([
        ("selector", TextSelector("text_for_model")),
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2000))
    ])
    numeric_pipeline = Pipeline([("num_features", NumericFeatureExtractor())])
    combined_features = FeatureUnion([
        ("text_features", text_pipeline),
        ("numeric_features", numeric_pipeline)
    ])
    pipeline = Pipeline([
        ("features", combined_features),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    return pipeline

if __name__ == "__main__":
    df = load_sample_data()
    df["text_for_model"] = df["subject"] + " " + df["email_text"]
    X = df[["sender", "subject", "email_text", "text_for_model"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds)*100)
    print("Report:\n", classification_report(y_test, preds))

    with open("phish_model.pickle", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as phish_model.pickle")

