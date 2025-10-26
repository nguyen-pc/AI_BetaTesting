from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from keybert import KeyBERT
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = FastAPI(title="Survey AI Analysis Service")

# ✅ Bật CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc giới hạn domain frontend của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
kw_model = KeyBERT()

class TextInput(BaseModel):
    texts: list[str]

@app.post("/analyze")
def analyze_text(data: TextInput):
    texts = [BeautifulSoup(t, "html.parser").get_text() for t in data.texts if t.strip()]
    if not texts:
        return {"error": "No valid text input"}

    results = [sentiment_model(t[:512])[0] for t in texts]
    pos = sum(1 for r in results if r["label"].upper() == "POSITIVE")
    neg = sum(1 for r in results if r["label"].upper() == "NEGATIVE")
    neu = len(texts) - pos - neg
    total = len(texts)

    sentiment_summary = {
        "positive": round((pos / total) * 100, 1),
        "neutral": round((neu / total) * 100, 1),
        "negative": round((neg / total) * 100, 1),
        "total": total,
    }

    all_text = " ".join(texts)
    keywords = kw_model.extract_keywords(all_text, top_n=10)
    key_phrases = [{"text": k, "score": round(s, 3)} for k, s in keywords]

    words = re.findall(r"\b\w+\b", all_text.lower())
    words = [w for w in words if w not in stop_words and len(w) > 2]
    freq = Counter(words).most_common(30)
    word_cloud = [{"text": w, "value": c} for w, c in freq]

    return {
        "sentiment_summary": sentiment_summary,
        "key_phrases": key_phrases,
        "word_cloud": word_cloud,
    }
