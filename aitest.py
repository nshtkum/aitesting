# aitest.py
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter

# Try to import SemanticChunker
has_semantic = True
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_google_vertexai import VertexAIEmbeddings
except ImportError:
    has_semantic = False

st.set_page_config(page_title="IntentAlign", layout="wide")
st.title("📌 Intent Alignment Checker")

# --- Sidebar: Embedding Provider ---
st.sidebar.header("Embedding Model")
model_choice = st.sidebar.radio("Embedding Source", ["Local (Free)", "OpenAI", "Vertex AI"])
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
vertex_project = st.sidebar.text_input("Vertex Project ID")
vertex_region = st.sidebar.text_input("Vertex Region", value="us-central1")

ENCODER = tiktoken.get_encoding("cl100k_base")
def count_tokens(text): return len(ENCODER.encode(text))

# --- Embedding Methods ---
def embed_local(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts)

def embed_openai(texts):
    import openai
    openai.api_key = openai_key
    return np.array([openai.Embedding.create(input=[t], model="text-embedding-3-small")['data'][0]['embedding'] for t in texts])

def embed_vertex(texts):
    import vertexai
    vertexai.init(project=vertex_project, location=vertex_region)
    model = vertexai.language_models.TextEmbeddingModel.from_pretrained("gemini-embedding-001")
    return np.array([model.get_embeddings([t])[0].values for t in texts])

def get_embedding_fn():
    if model_choice == "Local (Free)": return embed_local
    if model_choice == "OpenAI": return embed_openai
    if model_choice == "Vertex AI": return embed_vertex

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

chunkers = {
    "HTML": html_chunk,
    "Token": lambda text: TokenTextSplitter(chunk_size=100, chunk_overlap=20).split_text(text),
    "Recursive": lambda text: RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_text(text)
}

if has_semantic:
    def semantic_chunk(text):
        embeddings = VertexAIEmbeddings(model_name="text-embedding-005")
        chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
        return chunker.split_text(text)
    chunkers["Semantic"] = semantic_chunk
else:
    st.warning("SemanticChunker not available. Only standard chunkers will be used.")

# --- Upload & Input ---
st.subheader("Step 1: Add Your Inputs")
keyword = st.text_input("Enter Main Keyword or Intent")
uploaded_files = st.file_uploader("Upload up to 3 PDF screenshots", accept_multiple_files=True, type=["pdf"])
pasted_text = st.text_area("Or paste full page text (separate by --- for multiple pages)", height=200)

# --- Run Button ---
if st.button("🔍 Check Intent Alignment"):
    embed_fn = get_embedding_fn()
    docs = []

    if uploaded_files:
        for file in uploaded_files:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = " ".join([page.get_text() for page in doc])
            docs.append((file.name, text))
    elif pasted_text:
        for i, block in enumerate(pasted_text.split("---")):
            if len(block.strip()) > 20:
                docs.append((f"Text_{i+1}", block.strip()))

    if not keyword or not docs:
        st.warning("Please provide both keyword and content.")
    else:
        st.subheader("🔍 Alignment Score Table")
        embed_query = embed_fn([keyword])[0].reshape(1, -1)
        rows = []

        for label, content in docs:
            for method, splitter in chunkers.items():
                try:
                    chunks = splitter(content)
                    vectors = embed_fn(chunks)
                    sims = cosine_similarity(embed_query, vectors)[0]
                    avg_score = round(np.mean(sims), 4)
                    rows.append({"Page": label, "Chunking": method, "Similarity": avg_score})
                except Exception as e:
                    rows.append({"Page": label, "Chunking": method, "Similarity": "Error"})

        df = pd.DataFrame(rows)
        st.dataframe(df)

        aligned_pages = df[df["Similarity"] != "Error"].groupby("Page")["Similarity"].max().reset_index()
        aligned_pages["Intent Match"] = aligned_pages["Similarity"] >= 0.80
        st.subheader("✅ Summary")
        st.write(aligned_pages)
        st.success("Pages marked ✔️ are well-aligned with your keyword intent")
