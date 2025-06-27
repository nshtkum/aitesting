# app.py
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_vertexai import VertexAIEmbeddings
import vertexai
import openai

st.set_page_config(page_title="Semantic Rank Aligner", layout="wide")
st.title("📊 Semantic Alignment Analyzer for AI Ranking")

# --- Sidebar: Embedding Config ---
st.sidebar.header("Embedding Settings")
embed_source = st.sidebar.radio("Choose Embedding Provider", ["Local (Free)", "OpenAI", "Vertex AI"])

openai_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
vertex_project = st.sidebar.text_input("🧠 Vertex Project ID")
vertex_region = st.sidebar.text_input("📍 Vertex Region", value="us-central1")

# --- Tokenizer Setup ---
ENCODER = tiktoken.get_encoding("cl100k_base")
def count_tokens(text): return len(ENCODER.encode(text))

# --- Embedding Functions ---
def embed_local(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

def embed_openai(texts):
    openai.api_key = openai_key
    emb = []
    for t in texts:
        r = openai.Embedding.create(input=[t], model="text-embedding-3-small")
        emb.append(r['data'][0]['embedding'])
    return np.array(emb)

def embed_vertex(texts):
    vertexai.init(project=vertex_project, location=vertex_region)
    model = vertexai.language_models.TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    return np.array([model.get_embeddings([t])[0].values for t in texts], dtype='float32')

def get_embedding_fn():
    if embed_source == "Local (Free)": return embed_local
    if embed_source == "OpenAI": return embed_openai
    if embed_source == "Vertex AI": return embed_vertex

# --- Chunking Methods ---
def html_chunk(text):
    segments = [s for s in text.split("\n") if len(s.split()) >= 3]
    chunks, buffer = [], ""
    for seg in segments:
        combined = (buffer + " " + seg).strip()
        if count_tokens(combined) <= 100:
            buffer = combined
        else:
            if buffer: chunks.append(buffer)
            buffer = seg
    if buffer: chunks.append(buffer)
    return chunks

def token_chunk(text):
    splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20, encoding_name="cl100k_base")
    return splitter.split_text(text)

def recursive_chunk(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    return splitter.split_text(text)

def semantic_chunk(text):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
    return chunker.split_text(text)

chunkers = {
    "HTML": html_chunk,
    "Token": token_chunk,
    "Recursive": recursive_chunk,
    "Semantic": semantic_chunk
}

# --- PDF/Manual Upload ---
with st.expander("📄 Upload Content (PDF or Text)"):
    keyword = st.text_input("Main Keyword for Comparison")
    uploaded_files = st.file_uploader("Upload Screenshots as PDF", accept_multiple_files=True, type=['pdf'])
    pasted_texts = st.text_area("Or Paste Page Texts (separate by ---)", height=200)

# --- Analyze Button ---
if st.button("🔍 Run Semantic Analysis"):
    embed_fn = get_embedding_fn()
    inputs = []

    if uploaded_files:
        for file in uploaded_files:
            doc = fitz.open(file)
            txt = " ".join([page.get_text() for page in doc])
            inputs.append((file.name, txt))

    elif pasted_texts:
        parts = pasted_texts.split("---")
        for i, p in enumerate(parts):
            if len(p.strip()) > 30:
                inputs.append((f"Text_{i+1}", p.strip()))

    results = []

    for label, content in inputs:
        row = {"Variant": label}
        for method, func in chunkers.items():
            try:
                chunks = func(content)
                emb_chunks = embed_fn(chunks)
                emb_query = embed_fn([keyword])[0].reshape(1, -1)
                sims = cosine_similarity(emb_query, emb_chunks)[0]
                avg_sim = round(float(np.mean(sims)), 4)
                row[method] = avg_sim
            except Exception as e:
                row[method] = "Error"
                print(f"Chunker {method} failed: {e}")
        row["Overall Average"] = round(np.mean([v for v in row.values() if isinstance(v, float)]), 4)
        results.append(row)

    st.success("Analysis Complete")
    df_result = pd.DataFrame(results)
    st.dataframe(df_result, use_container_width=True)
    st.bar_chart(df_result.set_index("Variant")[["HTML", "Recursive", "Semantic", "Token"]])
