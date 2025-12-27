from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from keybert import KeyBERT
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from typing import List, Optional, Literal

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import google.generativeai as genai
from dotenv import load_dotenv
import traceback
import os
import json

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

app = FastAPI(title="Survey AI Analysis Service")

#  Bật CORS
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


# AIzaSyCN3nue1noVLjt__X9KW9om8IseqzC64Gk

# Load env & configure Gemini
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", API_KEY)
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

genai.configure(api_key="AIzaSyAzs4oGmPM7sTGguD3ZtPLphBaDouvE5_U")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    sessionId: str | None = None
    mode: str | None = "general"
    message: str
    history: list[ChatMessage] | None = [] 
    contextData: dict | None = None

PROMPT_SYSTEM = (
    "You are BetaBot, an AI assistant for the BetaTesting platform. "
    "You analyze bug reports, tester feedback, and generate use cases, test cases, and survey questions. "
    "Respond clearly and concisely."
)

MODE_TEMPLATES = {
    "bug": (
        "Analyze this bug report and return a **strict JSON** with fields: "
        "`severity`, `priority`, `root_cause`, and `fix_suggestion`. "
        "Bug Report: {content}"
    ),
    "feedback": (
        "Analyze this feedback and respond with: "
        "- Sentiment (positive/neutral/negative)\n"
        "- Three main insights as bullet points.\n"
        "Feedback: {content}"
    ),
    "usecase": (
       "Generate 4–6 use cases in **strict JSON array** format, each object has fields: "
        "`name` and `description`. No markdown or text outside JSON. "
        "Context: {content}"
    ),

     "testcase": (
        "Generate 5–10 detailed **test cases** in strict JSON array format. "
        "Each object must have: `title`, `description`, `preCondition`, `steps`, `expectedResult`. "
        "No markdown or text outside JSON. Context: {content}"
    ),
    "testscenario": (
        "Generate 3–6 **test scenarios** in JSON array format. "
        "Each object must have: `title`, `description`, and `precondition`. "
        "Context: {content}"
    ),
    "survey": (
        "Create 6 concise survey questions (Likert or multiple choice) "
        "related to: {content}. Return as a numbered list."
    ),
    "general": "{content}",
}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        content = BeautifulSoup(req.message, "html.parser").get_text().strip()
        if not content:
            raise HTTPException(status_code=400, detail="Empty message")

        template = MODE_TEMPLATES.get(req.mode or "general", MODE_TEMPLATES["general"])
        user_prompt = template.format(content=content)

        if hasattr(req, "contextData") and req.contextData:
            user_prompt += (
                "\n\nHere is real project data in JSON:\n"
                f"{json.dumps(req.contextData, ensure_ascii=False)}\n"
                "Use ONLY this data to answer the developer's question."
            )

        print("User Prompt:", user_prompt)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Sửa: tạo context đúng chuẩn Gemini
        context = [
            {"role": "user", "parts": [{"text": PROMPT_SYSTEM}]}
        ]

        # if req.history:
        #     for h in req.history:
        #         role = "user" if h.role == "user" else "model"
        #         context.append({"role": role, "parts": [{"text": h.content}]})

        # Thêm message hiện tại
        context.append({"role": "user", "parts": [{"text": user_prompt}]})

        # Gọi Gemini
        response = model.generate_content(context)
        text = (response.text or "").strip()

        return {
            "sessionId": req.sessionId,
            "mode": req.mode,
            "response": text,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Service Error: {e}")
    

@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.5-flash"}


class BugAnalysisCampaign(BaseModel):
    id: int
    title: Optional[str] = ""
    description: Optional[str] = ""
    instructions: Optional[str] = ""

class BugAnalysisItem(BaseModel):
    id: int
    title: str
    description: Optional[str] = ""
    stepsToReproduce: Optional[str] = ""
    expectedResult: Optional[str] = ""
    actualResult: Optional[str] = ""
    severity: Optional[str] = ""
    priority: Optional[str] = ""
    status: Optional[str] = ""
    testerId: Optional[int] = None
    testerUserName: Optional[str] = None
    campaignId: Optional[int] = None

class BugAnalysisRequest(BaseModel):
    campaign: BugAnalysisCampaign
    bugs: List[BugAnalysisItem]

@app.post("/bug/analyze")
def analyze_bug(req: BugAnalysisRequest):
    try:
        if not req.bugs or len(req.bugs) == 0:
            raise HTTPException(status_code=400, detail="No bug data provided")

        def clean(text: str):
            return BeautifulSoup(text or "", "html.parser").get_text().strip()

        bug_list = []
        for b in req.bugs:
            bug_list.append({
                "id": b.id,
                "title": clean(b.title),
                "description": clean(b.description),
                "steps": clean(b.stepsToReproduce),
                "expected": clean(b.expectedResult),
                "actual": clean(b.actualResult),
                "severity": b.severity or "UNKNOWN",
                "priority": b.priority or "NONE",
                "status": b.status or "NONE",
            })

        severity_count = {}
        for b in bug_list:
            sev = b["severity"]
            severity_count[sev] = severity_count.get(sev, 0) + 1

        texts = [f"{b['title']} {b['description']} {b['steps']}" for b in bug_list]
        vectorizer = CountVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(vectors)

        duplicates = []
        for i in range(len(bug_list)):
            for j in range(i + 1, len(bug_list)):
                if sim_matrix[i][j] > 0.65:
                    duplicates.append({
                        "bugA": bug_list[i]["id"],
                        "bugB": bug_list[j]["id"],
                        "similarity": round(sim_matrix[i][j], 3)
                    })

        # FIXED PROMPT (escaped braces)
        prompt = f"""
        Bạn là chuyên gia QA Lead. Hãy phân tích toàn bộ danh sách bug trong chiến dịch này.

        Chiến dịch:
        ID: {req.campaign.id}
        Tiêu đề: {req.campaign.title}
        Mô tả: {clean(req.campaign.description)}

        Hãy trả về **STRICT JSON (KHÔNG MARKDOWN)** với các key sau (bằng tiếng Việt):

        {{
        "tong_quan": "Tóm tắt tình trạng lỗi và mô hình chung",
        "muc_do_rui_ro": "Low/Medium/High/Critical",
        "muc_do_severity_hang_dau": ["MAJOR", "MINOR"],
        "nguyen_nhan_kha_nang": ["...", "..."],
        "goi_y_uu_tien_sua_loi": [{{ "bugId": 1, "suggestedPriority": "HIGH" }}],
        "de_xuat": ["...", "..."],
        "danh_gia_chat_luong_bao_cao": "..."
        }}

        Dữ liệu bug:
        {bug_list}
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        ai_result = (model.generate_content(prompt).text or "").strip()

        return {
            "campaignId": req.campaign.id,
            "campaignTitle": req.campaign.title,
            "total_bugs": len(bug_list),
            "severity_overview": severity_count,
            "duplicates": duplicates,
            "ai_analysis": ai_result,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Bug Analysis Error: {str(e)}")


# Gợi ý cho người dùng


class UserProfile(BaseModel):
    id: int
    country: Optional[str] = ""
    zipcode: Optional[str] = ""
    gender: Optional[str] = ""
    languages: Optional[str] = ""
    browsers: Optional[str] = ""
    computer: Optional[str] = ""
    smartPhone: Optional[str] = ""
    otherDevice: Optional[str] = ""
    employment: Optional[str] = ""
    gamingGenres: Optional[str] = ""
    householdIncome: Optional[str] = ""
    children: Optional[bool] = None

class CampaignProfile(BaseModel):
    id: int
    campaignId: int
    gender: Optional[str] = ""
    country: Optional[str] = ""
    zipcode: Optional[str] = ""
    householdIncome: Optional[str] = ""
    isChildren: Optional[bool] = None
    employment: Optional[str] = ""
    gamingGenres: Optional[str] = ""
    browsers: Optional[str] = ""
    languages: Optional[str] = ""
    ownedDevices: Optional[str] = ""
    devices: Optional[str] = ""

class RecommendRequest(BaseModel):
    user: UserProfile
    campaigns: List[CampaignProfile]

class RecommendResult(BaseModel):
    campaignId: int
    score: float

class Interaction(BaseModel):
    user_id: int
    campaign_id: int
    interaction_type: Literal['view', 'apply']


def profile_to_text(p: dict):
    parts = []
    for key in [
        "gender",
        "country",
        "zipcode",
        "employment",
        "gamingGenres",
        "browsers",
        "languages",
        "ownedDevices",
        "devices",
        "webExpertise",
        "householdIncome",
    ]:
        val = p.get(key)
        if val:
            parts.append(str(val))
    return " ".join(parts)


def user_to_text(user: dict):
    devices = ", ".join(
        [x for x in [user.get("computer"), user.get("smartPhone"), user.get("otherDevice")] if x]
    )
    tmp = dict(user)
    tmp["devices"] = devices
    return profile_to_text(tmp)


def content_based_scores(user, campaigns):
    campaign_texts = [profile_to_text(c) for c in campaigns]
    user_text = user_to_text(user)

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([user_text] + campaign_texts)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

    scores = {}
    for cam, sim_score in zip(campaigns, similarities):
        country_score = 1.0 if not cam.get("country") or cam.get("country") == user.get("country") else 0.0
        zipcode_score = 1.0 if not cam.get("zipcode") or cam.get("zipcode") == user.get("zipcode") else 0.0
        final_score = 0.7 * sim_score + 0.2 * country_score + 0.1 * zipcode_score
        scores[cam["campaignId"]] = round(final_score, 3)
    return scores


def get_collaborative_scores(user_id: int, interactions_path="interactions.json"):
    if not os.path.exists(interactions_path):
        return {}
    with open(interactions_path, "r", encoding="utf-8") as f:
        interactions = json.load(f)
    if not interactions:
        return {}

    df = pd.DataFrame(interactions)
    df["interaction_value"] = df["interaction_type"].apply(lambda x: 2 if x == "apply" else 1)
    df = df.sort_values(by="interaction_value", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "campaign_id"], keep="first")

    matrix = df.pivot_table(index="user_id", columns="campaign_id", values="interaction_value", fill_value=0)
    if user_id not in matrix.index:
        return {}

    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    similar_users = sim_df[user_id].drop(user_id).sort_values(ascending=False).head(3)

    scores = {}
    for sim_user, sim_score in similar_users.items():
        row = matrix.loc[sim_user]
        for campaign_id, val in row.items():
            if matrix.loc[user_id, campaign_id] == 0 and val > 0:
                scores[campaign_id] = scores.get(campaign_id, 0) + sim_score * val
    return scores


@app.post("/recommend", response_model=List[RecommendResult])
def recommend(req: RecommendRequest):
    try:
        content_scores = content_based_scores(req.user.dict(), [c.dict() for c in req.campaigns])
        cf_scores = get_collaborative_scores(req.user.id)

        results = []
        for c in req.campaigns:
            cid = c.campaignId
            content = content_scores.get(cid, 0)
            cf = cf_scores.get(cid, 0)
            final = round(0.7 * content + 0.3 * cf, 3)
            results.append(RecommendResult(campaignId=cid, score=final))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:5]

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recommend Error: {e}")


@app.post("/interactions")
def log_interaction(data: Interaction):
    print("Logging interaction:", data)
    interaction = data.dict()
    interaction["timestamp"] = datetime.now().isoformat()

    try:
        with open("interactions.json", "r", encoding="utf-8") as f:
            interactions = json.load(f)
    except:
        interactions = []

    interactions.append(interaction)
    with open("interactions.json", "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)

    return {"message": "Interaction logged successfully"}