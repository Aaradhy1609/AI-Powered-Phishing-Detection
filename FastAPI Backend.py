from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pickle
import re
import pandas as pd

with open("phish_model.pickle", "rb") as f:
    model = pickle.load(f)

SUSPICIOUS_KEYWORDS = {"urgent","verify","confirm","click","immediately","locked",
                       "password","bank","account","suspended","risk","security",
                       "claim","winner"}
TRUSTED_DOMAINS = ["bank","facebook","google","microsoft"]
URL_REGEX = re.compile(r"https?://[^\s]+")

app = FastAPI(title="AI Phishing Detection")

class EmailInput(BaseModel):
    sender: str
    subject: str
    email_text: str
    threshold: float = 0.6

def extract_urls(text): return URL_REGEX.findall(text or "")
def get_domain(email):
    if not email or "@" not in email:
        return None
    return email.split("@")[-1].lower()

def is_domain_trusted(domain):
    if not domain:
        return False
    return any(domain == td or domain.endswith("." + td) for td in TRUSTED_DOMAINS)

def predict_email(sender, subject, email_text, threshold=0.6):
    df = pd.DataFrame([{"sender": sender, "subject": subject, "email_text": email_text}])
    df["text_for_model"] = df["subject"] + " " + df["email_text"]
    proba = model.predict_proba(df)[:,1][0]
    label = "Phishing" if proba >= threshold else "Safe"

    reasons = []
    if extract_urls(subject + email_text):
        reasons.append("Contains suspicious URL(s)")
    if not is_domain_trusted(get_domain(sender)):
        reasons.append("Untrusted sender domain")
    if any(k in (subject + email_text).lower() for k in SUSPICIOUS_KEYWORDS):
        reasons.append("Suspicious keywords detected")

    return {"label": label, "probability": float(proba), "reasons": reasons}

@app.post("/predict")
def predict(email: EmailInput):
    result = predict_email(email.sender, email.subject, email.email_text, email.threshold)
    return JSONResponse(content=result)

@app.get("/")
def home():
    return {"message": "Welcome to AI Phishing Detection API!"}
