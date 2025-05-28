# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# def get_text(skills):
#     return " ".join(skills)

# def recommend_jobs(candidate, jobs, top_n=3):
#     job_texts = [get_text(job["skills"]) for job in jobs]
#     candidate_text = get_text(candidate["skills"])
    
#     vectorizer = CountVectorizer()
#     vectors = vectorizer.fit_transform([candidate_text] + job_texts)
    
#     similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]

#     scored_jobs = []
#     for job, score in zip(jobs, similarities):
#         location_score = 1.0 if candidate["address"] == job.get("location", "") else 0.0
#         final_score = 0.7 * score + 0.3 * location_score
#         scored_jobs.append({"job": job, "score": round(final_score, 3)})

#     return sorted(scored_jobs, key=lambda x: x["score"], reverse=True)[:top_n]



# # Tạo DataFrame từ dữ liệu tương tác
# df = pd.DataFrame(interactions)

# # Tạo ma trận người dùng - công việc
# interaction_matrix = df.pivot_table(index='user_id', columns='job_id', values='interaction', fill_value=0)

# # Tính độ tương đồng giữa người dùng
# user_sim = cosine_similarity(interaction_matrix)

# user_sim_df = pd.DataFrame(user_sim, index=interaction_matrix.index, columns=interaction_matrix.index)

# def get_collaborative_scores(target_user_id):
#     similar_users = user_sim_df[target_user_id].drop(target_user_id)
#     top_sim_users = similar_users.sort_values(ascending=False).head(3)

#     weighted_scores = {}
#     for sim_user, sim_score in top_sim_users.items():
#         jobs_applied = interaction_matrix.loc[sim_user]
#         for job_id, applied in jobs_applied.items():
#             if applied == 1 and interaction_matrix.loc[target_user_id, job_id] == 0:
#                 weighted_scores[job_id] = weighted_scores.get(job_id, 0) + sim_score

#     return weighted_scores  # job_id -> score


