import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="ML/AI Paper QA (RAG)")

GEN_MODEL_NAME = "google/flan-t5-base"


@st.cache_resource
def load_artifacts():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dense_index = faiss.read_index("rag_dense_index.faiss")
    chunks_df = pd.read_csv("rag_chunks.csv")
    with open("rag_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(device)

    return dense_index, chunks_df, bm25, embed_model, cross_encoder, gen_tokenizer, gen_model, device


dense_index, chunks_df, bm25, embed_model, cross_encoder, gen_tokenizer, gen_model, device = load_artifacts()


def hybrid_retrieve(query, top_n=20):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    dense_scores, dense_idx = dense_index.search(q_emb, top_n)
    dense_scores, dense_idx = dense_scores[0], dense_idx[0]

    bm25_scores_all = bm25.get_scores(query.lower().split())
    bm25_top_idx = np.argsort(bm25_scores_all)[::-1][:top_n]

    candidate_idx = list(set(dense_idx.tolist()) | set(bm25_top_idx.tolist()))

    dense_score_map = {i: s for i, s in zip(dense_idx, dense_scores)}
    d_vals = np.array([dense_score_map.get(i, 0.0) for i in candidate_idx])
    b_vals = np.array([bm25_scores_all[i] for i in candidate_idx])

    d_norm = (d_vals - d_vals.min()) / (d_vals.max() - d_vals.min() + 1e-9)
    b_norm = (b_vals - b_vals.min()) / (b_vals.max() - b_vals.min() + 1e-9)
    hybrid_scores = 0.5 * d_norm + 0.5 * b_norm

    order = np.argsort(hybrid_scores)[::-1]
    return [candidate_idx[i] for i in order]


def rerank(query, candidate_idx, final_k=3):
    texts = chunks_df.iloc[candidate_idx]["text"].tolist()
    ce_scores = cross_encoder.predict([(query, t) for t in texts])
    order = np.argsort(ce_scores)[::-1][:final_k]
    top_idx = [candidate_idx[i] for i in order]
    result = chunks_df.iloc[top_idx].copy()
    result["rerank_score"] = [ce_scores[i] for i in order]
    return result


def retrieve(query, top_n=20, final_k=3):
    candidates = hybrid_retrieve(query, top_n=top_n)
    return rerank(query, candidates, final_k=final_k)


def generate_answer(query, context, max_new_tokens=100, min_new_tokens=25):
    prompt = (
        "Answer the question in a complete, detailed sentence using only the context below. "
        "Do not answer with a single word.\n\n"
        f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    output_ids = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        early_stopping=True
    )
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


st.title("ML/AI Paper Question Answering (RAG)")
st.caption("Hybrid retrieval (BM25 + dense) with cross-encoder re-ranking, answers generated from retrieved context only.")

query = st.text_input("Your question")
final_k = st.slider("Chunks to use as context", min_value=1, max_value=5, value=3)

if query:
    with st.spinner("Retrieving context and generating answer..."):
        retrieved = retrieve(query, top_n=20, final_k=final_k)
        context = " ".join(retrieved["text"].tolist())
        answer = generate_answer(query, context)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for _, row in retrieved.iterrows():
        st.markdown(f"**{row['title']}**  (relevance score: {row['rerank_score']:.3f})")