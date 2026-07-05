# Week 7 - Document Question Answering System (RAG)

This week's assignment was to build a Retrieval-Augmented Generation (RAG) system that can answer questions from a custom document collection instead of relying only on a language model's own knowledge.

## Live Demo
The project can be live interacted from the streamlit community cloud on this [link](https://hybrid-rag-mlarxiv.streamlit.app/).

## Overview

The project retrieves relevant chunks from a corpus of ML/AI arXiv paper abstracts and uses those chunks as context for a language model to generate an answer. The idea is that answers should be grounded in the retrieved text rather than made up from the model's own memory.

## Dataset

Used the **[ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)** dataset from Hugging Face, which has paper titles and abstracts tagged under the cs.LG category. A random sample of 4,000 papers was used to keep things fast while still covering a wide range of ML/AI topics.

## Pipeline

1. Combine title + abstract into a single chunk per paper
2. Generate embeddings with `all-MiniLM-L6-v2`, split across both GPUs
3. Build a dense FAISS index and a BM25 keyword index over the corpus
4. Retrieve candidates using both (hybrid search), then re-rank them with a cross-encoder
5. Pass the top chunks as context to `google/flan-t5-base` to generate the final answer

A grounding check is also included that compares answers generated with retrieved context vs. with no context, to confirm the answers are actually coming from retrieval and not just the model guessing.

## Files

- `Week7_RAG_Advanced.ipynb` - full notebook with EDA, pipeline, and evaluation
- `app.py` - Streamlit UI to query the RAG system interactively
- `requirements.txt` - dependencies for running the app
- `rag_dense_index.faiss` - saved FAISS vector index
- `rag_bm25.pkl` - saved BM25 keyword index
- `rag_chunks.csv` - chunk text and paper metadata used by the app

## Running the app

```
pip install -r requirements.txt
streamlit run app.py
```

## Notes

Retrieval quality was checked with a self-retrieval hit-rate test (using paper titles as queries) since the dataset doesn't come with labeled QA pairs. It's more of a sanity check than a real benchmark, since a title is a close paraphrase of its own abstract. The more meaningful test was running actual ML questions through the pipeline and checking the answers make sense given the retrieved sources.
