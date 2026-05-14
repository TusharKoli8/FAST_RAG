"""
RAG Streamlit Frontend
Run:  streamlit run frontend/app.py
Make sure the FastAPI backend is running on http://localhost:8000
"""

import streamlit as st
import requests

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="📄 RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
    .sub-header  { color: #555; font-size: 1rem; margin-bottom: 1.5rem; }
    .answer-box  {
        background: #f8f9fa; border-left: 4px solid #4a90e2;
        border-radius: 6px; padding: 1rem 1.2rem;
        margin-top: 0.5rem; font-size: 1rem; line-height: 1.6;
    }
    .error-detail {
        background: #fff3cd; border-left: 4px solid #ffc107;
        border-radius: 6px; padding: 0.8rem 1rem;
        font-family: monospace; font-size: 0.85rem;
        white-space: pre-wrap; word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_backend() -> bool:
    try:
        return requests.get(f"{API_URL}/", timeout=3).status_code == 200
    except Exception:
        return False

def get_stats() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def upload_pdf(file_bytes: bytes, filename: str) -> dict:
    r = requests.post(f"{API_URL}/upload",
                      files={"file": (filename, file_bytes, "application/pdf")},
                      timeout=120)
    if not r.ok:
        # Pull the real detail out of the JSON error body
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()

def ask_question(question: str) -> str:
    r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(detail)
    return r.json()["answer"]

def clear_store() -> str:
    r = requests.delete(f"{API_URL}/clear", timeout=10)
    r.raise_for_status()
    return r.json()["message"]


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "uploaded_files"  not in st.session_state: st.session_state.uploaded_files  = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload PDFs and ask questions about their content.</div>',
            unsafe_allow_html=True)

backend_ok = check_backend()
if backend_ok:
    st.success("✅ Backend connected", icon="🟢")
else:
    st.error("❌ Cannot reach the FastAPI backend at `http://localhost:8000`. "
             "Start it with: `uvicorn app:app --reload`", icon="🔴")
    st.stop()

st.divider()

left_col, right_col = st.columns([1, 2], gap="large")

# ════════════════════════════════════════════════════════════════════════════
# LEFT — Upload & Stats
# ════════════════════════════════════════════════════════════════════════════
with left_col:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("⬆️ Process & Store", use_container_width=True, type="primary"):
            with st.spinner(f"Processing **{uploaded_file.name}** …"):
                try:
                    result = upload_pdf(uploaded_file.read(), uploaded_file.name)
                    st.success(f"✅ {result['message']}  \nChunks created: **{result['chunks_created']}**")
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                    st.rerun()
                except RuntimeError as e:
                    st.error(f"❌ Upload failed")
                    st.markdown(f'<div class="error-detail">{e}</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_files:
        st.markdown("**Uploaded this session:**")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- 📄 `{name}`")

    st.divider()

    st.subheader("🗄️ Knowledge Base")
    stats = get_stats()
    if stats:
        c1, c2 = st.columns(2)
        with c1: st.metric("Total Chunks", stats.get("total_chunks", 0))
        with c2: st.metric("Index Size",   stats.get("index_size",   0))
        st.caption(f"Embedding dimension: {stats.get('dimension', '—')}")
    else:
        st.info("Could not fetch stats.")

    st.divider()

    with st.expander("⚠️ Danger Zone"):
        st.warning("This will permanently delete the vector store and all stored chunks.")
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True, type="secondary"):
            try:
                msg = clear_store()
                st.session_state.uploaded_files = []
                st.session_state.chat_history   = []
                st.success(msg)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")

    # ── Groq debug helper ─────────────────────────────────────────────────
    st.divider()
    with st.expander("🔧 Debug: Test Groq connection"):
        st.markdown("Run this command in your **`backend/`** terminal to see the exact Groq error:")
        st.code(
            'python -c "\n'
            'from dotenv import load_dotenv; load_dotenv()\n'
            'from groq import Groq; import os\n'
            'client = Groq(api_key=os.getenv(\'GROQ_API_KEY\'))\n'
            'r = client.chat.completions.create(\n'
            '    model=\'llama3-70b-8192\',\n'
            '    messages=[{\'role\':\'user\',\'content\':\'hi\'}]\n'
            ')\n'
            'print(r.choices[0].message.content)\n"',
            language="bash"
        )
        st.markdown("Also check your `.env` file in `backend/`:")
        st.code("GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx", language="bash")
        st.markdown("Get a free key at: https://console.groq.com/keys")


# ════════════════════════════════════════════════════════════════════════════
# RIGHT — Chat
# ════════════════════════════════════════════════════════════════════════════
with right_col:
    st.subheader("💬 Ask a Question")

    if st.session_state.chat_history:
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.markdown(f'<div class="answer-box">{a}</div>', unsafe_allow_html=True)
        st.divider()

    stats      = get_stats()
    no_chunks  = not stats or stats.get("total_chunks", 0) == 0

    question = st.text_area(
        "Your question",
        placeholder="e.g. What is the name defined in this PDF?",
        height=100,
        disabled=no_chunks,
        help="Upload and process a PDF first." if no_chunks else "",
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_btn = st.button("🔍 Ask", use_container_width=True, type="primary",
                            disabled=no_chunks or not question.strip())
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if no_chunks:
        st.info("📥 Upload and process a PDF on the left to enable Q&A.")

    if ask_btn and question.strip():
        with st.spinner("Thinking …"):
            try:
                answer = ask_question(question.strip())
                st.session_state.chat_history.append((question.strip(), answer))
                st.rerun()
            except RuntimeError as e:
                st.error("❌ The backend returned an error:")
                # Show the full detail — this is where you'll see the real Groq error
                st.markdown(f'<div class="error-detail">{e}</div>', unsafe_allow_html=True)
                st.info("💡 Check the **🔧 Debug** section on the left for next steps.")
