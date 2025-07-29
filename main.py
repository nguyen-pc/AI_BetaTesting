# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List
# from data import candidates, jobs
# from recommender import recommend_jobs

# from typing import Literal
# import json
# from datetime import datetime


# app = FastAPI()


# class Interaction(BaseModel):
#     user_id: int
#     job_id: int
#     interaction_type: Literal['view', 'apply']

# class CandidateRequest(BaseModel):
#     candidate_id: int

# @app.post("/recommendations")
# def get_recommendations(req: CandidateRequest):
#     candidate = next((c for c in candidates if c["id"] == req.candidate_id), None)
#     if not candidate:
#         return {"error": "Candidate not found"}
    
#     recommendations = recommend_jobs(candidate, jobs)
#     return {"recommendations": recommendations}


# # Lưu vào file interactions.json (tạm thời)
# @app.post("/interactions")
# def log_interaction(data: Interaction):
#     interaction = data.dict()
#     interaction["timestamp"] = datetime.now().isoformat()

#     try:
#         with open("interactions.json", "r", encoding="utf-8") as f:
#             interactions = json.load(f)
#     except:
#         interactions = []

#     interactions.append(interaction)
#     with open("interactions.json", "w", encoding="utf-8") as f:
#         json.dump(interactions, f, ensure_ascii=False, indent=2)
    
#     return {"message": "Interaction logged successfully"}

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Literal
import json
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import fitz  # PyMuPDF
import os
import cohere
from dotenv import load_dotenv
# from data import candidates, jobs  # Assumed to be available
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware

# Load biến từ file .env
load_dotenv()

# Lấy khóa từ biến môi trường
api_key = os.getenv("COHERE_API_KEY")

co = cohere.Client(api_key)




app = FastAPI()

origins = [
    "http://localhost:5173",
    # Bạn có thể thêm các origin khác nếu cần, hoặc dùng "*" cho phép tất cả
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Cho phép các origin trong danh sách
    allow_credentials=True,
    allow_methods=["*"],          # Cho phép tất cả các phương thức
    allow_headers=["*"],          # Cho phép tất cả các headers
)


# ----- Models -----
class Interaction(BaseModel):
    user_id: int
    job_id: int
    interaction_type: Literal['view', 'apply']

class CandidateRequest(BaseModel):
    candidate: Dict  # hoặc tạo model Candidate nếu muốn
    jobs: List[Dict]

# ----- Utility Functions -----
def get_text(skills):
    return " ".join(skills)

def content_based_scores(candidate, jobs):
    job_texts = [get_text(job["skills"]) for job in jobs]
    candidate_text = get_text(candidate["skills"])

    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([candidate_text] + job_texts)
    similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

    content_scores = {}
    for job, score in zip(jobs, similarities):
        location_score = 1.0 if candidate["address"] == job.get("location", "") else 0.0
        final_score = 0.7 * score + 0.3 * location_score
        content_scores[job["id"]] = round(final_score, 3)

    return content_scores  # job_id -> score

def get_collaborative_scores(user_id, interactions_path="interactions.json"):
    if not os.path.exists(interactions_path):
        return {}

    with open(interactions_path, "r", encoding="utf-8") as f:
        interactions = json.load(f)

    if not interactions:
        return {}

    df = pd.DataFrame(interactions)

    # Giữ giá trị cao nhất nếu user_id - job_id trùng (apply > view)
    df["interaction_value"] = df["interaction_type"].apply(lambda x: 2 if x == "apply" else 1)
    df = df.sort_values(by="interaction_value", ascending=False)
    df = df.drop_duplicates(subset=["user_id", "job_id"], keep="first")

    # Pivot thành ma trận tiện ích
    matrix = df.pivot_table(index="user_id", columns="job_id", values="interaction_value", fill_value=0)

    if user_id not in matrix.index:
        return {}

    # Tính cosine similarity
    sim = cosine_similarity(matrix)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    # Lấy 3 người dùng tương tự nhất (có thể chỉnh K tuỳ ý)
    similar_users = sim_df[user_id].drop(user_id).sort_values(ascending=False).head(3)

    scores = {}
    for sim_user, sim_score in similar_users.items():
        jobs_interacted = matrix.loc[sim_user]
        for job_id, interaction_value in jobs_interacted.items():
            # Nếu người dùng hiện tại chưa tương tác với job_id
            if matrix.loc[user_id, job_id] == 0 and interaction_value > 0:
                scores[job_id] = scores.get(job_id, 0) + sim_score * interaction_value

    return scores  # job_id -> score
    

def recommend_jobs(candidate, jobs, top_n=4):
    content_scores = content_based_scores(candidate, jobs)
    cf_scores = get_collaborative_scores(candidate["id"])

    final_scores = {}
    for job in jobs:
        job_id = job["id"]
        content_score = content_scores.get(job_id, 0)
        cf_score = cf_scores.get(job_id, 0)
        final_score = round(0.7 * content_score + 0.3 * cf_score, 3)
        final_scores[job_id] = final_score

    scored_jobs = sorted(
        [{"job": job, "score": final_scores[job["id"]]} for job in jobs],
        key=lambda x: x["score"], reverse=True
    )
    return scored_jobs[:top_n]




def extract_text_from_pdf(file_bytes):
    doc = fitz.open("pdf", file_bytes)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def analyze_cv_with_ai(cv_text: str, job_description: str ):
    prompt = f"""
Bạn là chuyên gia nhân sự IT.

Dưới đây là mô tả công việc:

{job_description}

Dưới đây là nội dung CV ứng viên:

{cv_text}

Vui lòng phân tích và đưa ra:

1. Mức độ phù hợp giữa CV và công việc (theo thang điểm 0-10).
2. Những điểm mạnh của ứng viên so với yêu cầu công việc.
3. Những điểm còn thiếu hoặc cần cải thiện trong CV để phù hợp hơn.
4. Đề xuất cụ thể giúp ứng viên cải thiện CV.

Trả lời bằng tiếng Việt.
    """


    response = co.generate(
        model="command",  # mô hình mạnh nhất của Cohere
        prompt=prompt,
        max_tokens=300,
        temperature=0.5
    )

    return response.generations[0].text.strip()


# ----- API Endpoints -----
@app.post("/recommendations")
def get_recommendations(req: CandidateRequest):
    print(req)

    candidate = req.candidate
    jobs = req.jobs

    recommendations = recommend_jobs(candidate, jobs)

    print("Recommendations:", recommendations)
    return {"recommendations": recommendations}

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


@app.post("/analyze-cv")
async def analyze_cv(
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    contents = await file.read()
    try:
        text = extract_text_from_pdf(contents)
    except Exception as e:
        return {"error": f"Không thể đọc file PDF: {str(e)}"}

    if not text.strip():
        return {"error": "Không tìm thấy nội dung trong file PDF"}

    try:
        # Gọi hàm phân tích truyền cả CV và mô tả công việc
        # print(text, job_description)
        feedback = analyze_cv_with_ai(text, job_description)
        return {"analysis": feedback}
    except Exception as e:
        return {"error": f"Lỗi khi gọi AI: {str(e)}"}
