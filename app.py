import streamlit as st
import re
import  fitz # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import pandas as pd

# ------------------ PAGE CONFIG -----------------o-
st.set_page_config(
    page_title="Resume Ranker AI",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ PDF TEXT EXTRACTION ------------------
def extract_text_from_pdf(pdf_bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

# ------------------ SKILL EXTRACTION ------------------
def extract_skills(text):
    SKILLS = [
        "python","java","javascript","c++","sql","mongodb","postgresql",
        "machine learning","deep learning","nlp","llm","generative ai",
        "pytorch","tensorflow","docker","kubernetes","aws","gcp","azure",
        "fastapi","flask","django","react","streamlit","data science",
        "mlops","rag","transformers","bert","gpt","embeddings",
        "chromadb","pinecone","pandas","numpy"
    ]
    text = text.lower()
    found = []
    for skill in SKILLS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found.append(skill.title())
    return list(dict.fromkeys(found))

# ------------------ COSINE SIMILARITY ------------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ------------------ RANKING FUNCTION ------------------
def rank_resumes(job_desc, resumes, model):

    # Encode job description
    jd_embedding = model.encode(job_desc, normalize_embeddings=True)

    # Encode resumes
    resume_embeddings = [
        model.encode(r["text"][:3000], normalize_embeddings=True)
        for r in resumes
    ]

    # Create ChromaDB collection
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"resumes_{datetime.now().strftime('%H%M%S')}"
    )

    # Store embeddings
    for i, (r, emb) in enumerate(zip(resumes, resume_embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb.tolist()],
            metadatas=[{"name": r["name"]}],
            documents=[r["text"][:1000]]
        )

    # Query using embedding (IMPORTANT FIX)
    results = collection.query(
        query_embeddings=[jd_embedding.tolist()],
        n_results=len(resumes)
    )

    ranked = []
    for i, (idx, dist) in enumerate(zip(results["ids"][0], results["distances"][0])):
        
        resume = resumes[int(idx)]

        # Better similarity conversion
        similarity = round((1 - dist) * 100, 2)

        ranked.append({
            "rank": i + 1,
            "name": resume["name"],
            "score": similarity,
            "skills": resume["skills"],
            "snippet": resume["text"][:200].replace("\n", " ")
        })

    # Sort properly
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Re-rank index
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return ranked

# ------------------ UI ------------------
st.title("📄 Semantic Resume Ranking System")

col1, col2 = st.columns(2)

with col1:
    job_desc = st.text_area("Job Description", height=250)

with col2:
    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )

if st.button("🚀 Rank Resumes"):

    if not job_desc:
        st.warning("Enter job description")
    elif not uploaded_files:
        st.warning("Upload resumes")
    else:
        model = load_model()

        resumes = []
        for f in uploaded_files:
            text = extract_text_from_pdf(f.read())
            resumes.append({
                "name": f.name,
                "text": text,
                "skills": extract_skills(text)
            })

        ranked = rank_resumes(job_desc, resumes, model)

        st.subheader("🏆 Results")

        for r in ranked:
            st.write(f"### #{r['rank']} - {r['name']}")
            st.write(f"Score: **{r['score']}%**")
            st.write(f"Skills: {', '.join(r['skills']) if r['skills'] else 'None'}")
            st.write(f"Snippet: {r['snippet']}")
            st.markdown("---")

        # Table
        df = pd.DataFrame(ranked)
        st.dataframe(df)