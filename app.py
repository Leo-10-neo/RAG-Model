import sys
import time
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

# Deployment-safe import path setup for Streamlit Cloud/local folder layouts.
BASE_DIR = Path(__file__).resolve().parent
for candidate in [BASE_DIR, BASE_DIR / "Multimodal_RAG-main", BASE_DIR.parent]:
    if (candidate / "rag").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from rag.embeddings import get_jina_embeddings
from rag.vision import describe_image
from rag.chunking import chunk_text
from rag.retriever import FAISSRetriever
from rag.reranker import simple_rerank
from rag.llm import ask_llm


# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Enterprise Multimodal RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------ UI STYLING ------------------ #
st.markdown("""
<style>
body { font-family: Inter, sans-serif; }

.title { font-size: 38px; font-weight: 700; }
.subtitle { color: #6b7280; margin-bottom: 25px; }

.panel {
    padding: 18px;
    border-radius: 14px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

.section-header {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ------------------ HEADER ------------------ #
st.markdown("<div class='title'>🚀 Enterprise Multimodal RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Document + Vision based Retrieval-Augmented Generation</div>", unsafe_allow_html=True)


# ------------------ SESSION STATE ------------------ #
if "history" not in st.session_state:
    st.session_state.history = []


# ------------------ SIDEBAR ------------------ #
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_key = st.text_input("Groq API Key", type="password")
    jina_key = st.text_input("Jina API Key", type="password")

    model = st.selectbox(
        "Select LLM Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )

    filter_type = st.radio(
        "Retrieval Mode",
        ["all", "text", "image"],
        horizontal=True
    )

    st.divider()
    st.info("Upload document + image → Ask questions → Get AI-powered answers")


# ------------------ FILE UPLOAD ------------------ #
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📄 Upload Document</div>", unsafe_allow_html=True)
    txt_file = st.file_uploader("TXT or PDF", type=["txt", "pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🖼️ Upload Image</div>", unsafe_allow_html=True)
    img_file = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ PROCESS DOCUMENT ------------------ #
@st.cache_data(show_spinner=False)
def process_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    else:
        text = file.read().decode("utf-8")
    return text


@st.cache_data(show_spinner=False)
def generate_embeddings(chunks, jina_key):
    return get_jina_embeddings(chunks, jina_key)


# ------------------ RAG PIPELINE ------------------ #
if txt_file and groq_key and jina_key:

    with st.spinner("🔄 Processing knowledge base..."):

        raw_text = process_text(txt_file)
        chunks = chunk_text(raw_text)
        metadata = [{"type": "text"} for _ in chunks]

        if img_file:
            vision_text = describe_image(img_file.read(), groq_key)
            if vision_text:
                chunks.append("Image Context: " + vision_text)
                metadata.append({"type": "image"})

        embeddings = generate_embeddings(chunks, jina_key)
        retriever = FAISSRetriever(embeddings, metadata)

    st.success("✅ Knowledge base ready!")


    # ------------------ QUERY ------------------ #
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💬 Ask a Question</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Enter your query",
        placeholder="Example: Summarize the key points from the document and image"
    )

    run = st.button("🔍 Retrieve & Generate", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # ------------------ INFERENCE ------------------ #
    if run and query:

        start = time.time()

        query_emb = generate_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type
        ids = retriever.search(query_emb, top_k=5, filter_type=f)

        retrieved_docs = [chunks[i] for i in ids]
        reranked_docs = simple_rerank(query, retrieved_docs)

        context = "\n\n".join(reranked_docs[:3])
        answer = ask_llm(context, query, groq_key, model)

        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))

        # ------------------ OUTPUT ------------------ #
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🧠 Answer</div>", unsafe_allow_html=True)
            st.write(answer)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>⚡ Performance</div>", unsafe_allow_html=True)
            st.metric("Latency (sec)", latency)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📚 Retrieved Context"):
            st.text(context)

        with st.expander("🕘 Recent Queries"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.divider()

else:
    st.info("👆 Upload document + provide API keys to begin.")
