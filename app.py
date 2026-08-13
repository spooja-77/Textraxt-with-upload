"""DocBot — Document Intelligence & Database Chatbot UI.

Features:
- Tab 1: 💬 Chat (Zero-upload chat interface, answers questions from database).
- Tab 2: 📤 Upload Documents (Upload new files to database).
- Tab 3: 📁 Stored Files (View catalog of indexed documents & delete files).
"""

import itertools
import os
from datetime import datetime
import requests
import streamlit as st
from dotenv import load_dotenv

from backend.config import CHAT_MODEL, get_groq_client
from backend.database import (
    delete_document,
    get_all_documents_text,
    get_relevant_context,
    init_db,
    list_documents,
    save_document,
)
from backend.extractor import extract_file_content

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Initialize SQLite database on load
init_db()

# Sample questions for Quick Ask buttons
SAMPLE_QUESTIONS = [
    "What is the total bill amount in BillFor5?",
    "List all the dates mentioned in the documents.",
    "Summarize the key details from the uploaded files.",
    "What is the total amount of all bills?",
]

# ---------------------------------------------------------------- Theme & Styling

LIGHT = {
    "bg": "#E9EAF6",
    "panel": "#FFFFFF",
    "panel-2": "#F4F5FB",
    "sidebar": "#191D7A",
    "sidebar-2": "#232888",
    "sidebar-ink": "#EDEEFF",
    "sidebar-muted": "#9FA6E8",
    "ink": "#171A45",
    "muted": "#666B99",
    "navy": "#1D218F",
    "accent": "#3B4BF6",
    "accent-ink": "#FFFFFF",
    "line": "rgba(23,26,69,.14)",
    "led": "#22C55E",
    "shadow": "rgba(23,26,69,.10)",
}

DARK = {
    "bg": "#0E0F1E",
    "panel": "#181A2E",
    "panel-2": "#1F2138",
    "sidebar": "#12132B",
    "sidebar-2": "#1B1D3D",
    "sidebar-ink": "#DDDFF6",
    "sidebar-muted": "#7B80B8",
    "ink": "#E7E8F6",
    "muted": "#8A8EB8",
    "navy": "#232680",
    "accent": "#7C86FF",
    "accent-ink": "#0E0F1E",
    "line": "rgba(230,232,255,.12)",
    "led": "#34D399",
    "shadow": "rgba(0,0,0,.35)",
}

STATIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: transparent; }
.stMarkdown p, .stMarkdown li {
  font-family: 'Archivo', sans-serif;
  color: var(--ink);
  font-size: 1.0rem;
  line-height: 1.6;
}

/* --- Hero Banner (Premium Glassmorphism) --- */
.hero {
  display: flex;
  align-items: center;
  gap: 1.4rem;
  background: linear-gradient(135deg, #1a1f8e 0%, #2d34c8 40%, #4f56e8 75%, #6c72ff 100%);
  border-radius: 18px;
  padding: 2rem 2.2rem;
  margin-bottom: 1.4rem;
  box-shadow: 0 12px 40px rgba(59,75,246,.25), inset 0 1px 0 rgba(255,255,255,.12);
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -50%; right: -30%;
  width: 400px; height: 400px;
  background: radial-gradient(circle, rgba(255,255,255,.08) 0%, transparent 70%);
  border-radius: 50%;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -40%; left: -20%;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(124,134,255,.15) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-icon {
  font-size: 2.4rem;
  background: rgba(255,255,255,.12);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 16px;
  padding: .7rem .85rem;
  position: relative;
  z-index: 1;
}
.hero-title {
  font-family: 'Archivo', sans-serif;
  font-weight: 900;
  font-size: 2.1rem;
  letter-spacing: -.02em;
  color: #FFFFFF;
  line-height: 1.15;
  position: relative;
  z-index: 1;
}
.hero-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .82rem;
  color: rgba(195,199,255,.85);
  margin-top: .4rem;
  position: relative;
  z-index: 1;
}
.hero-badge {
  display: inline-block;
  background: rgba(255,255,255,.15);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,.2);
  border-radius: 20px;
  padding: .2rem .7rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .65rem;
  font-weight: 600;
  color: rgba(255,255,255,.9);
  letter-spacing: .08em;
  text-transform: uppercase;
  margin-top: .5rem;
  position: relative;
  z-index: 1;
}

/* --- Tab Bar (Equal-Width Pill Tabs) --- */
[data-testid="stTabs"] {
  margin-top: .2rem;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  display: flex !important;
  gap: 6px !important;
  background: var(--panel);
  padding: 6px;
  border-radius: 16px;
  border: 1px solid var(--line);
  box-shadow: 0 2px 12px var(--shadow);
  margin-bottom: 1.4rem;
  width: 100% !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] > div {
  flex: 1 1 33.33% !important;
  width: 33.33% !important;
  min-width: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  flex: 1 1 0% !important;
  width: 100% !important;
  height: 48px;
  border-radius: 12px;
  padding: 0 20px !important;
  font-family: 'Archivo', sans-serif;
  font-weight: 600;
  font-size: 0.92rem;
  color: var(--muted);
  background: transparent;
  border: none;
  transition: all 0.25s cubic-bezier(.4,0,.2,1);
  display: flex !important;
  align-items: center;
  justify-content: center;
  white-space: nowrap;
  cursor: pointer;
}
[data-testid="stTabs"] [data-baseweb="tab"] p {
  font-weight: 600 !important;
  font-size: 0.92rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--ink);
  background: var(--panel-2);
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: linear-gradient(135deg, #3B4BF6 0%, #5C65FF 100%) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 16px rgba(59,75,246,.3);
  font-weight: 700 !important;
}
[data-testid="stTabs"] [aria-selected="true"] p {
  color: #FFFFFF !important;
  font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  display: none !important;
}

/* --- Section Labels --- */
.sec-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .75rem;
  font-weight: 600;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.0rem 0 .5rem;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
  background: var(--sidebar);
  border-right: none;
}
.side-brand {
  font-family: 'Archivo', sans-serif;
  font-weight: 900;
  font-size: 1.45rem;
  color: var(--sidebar-ink);
  margin-bottom: .6rem;
}

/* Fix sidebar text visibility */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: var(--sidebar-ink) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: .78rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: var(--sidebar-2);
  color: var(--sidebar-ink);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .85rem;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
[data-testid="stSidebar"] [data-testid="stToggle"] p {
  text-transform: none;
  letter-spacing: .04em;
}

/* --- Chat Styling --- */
[data-testid="stChatMessage"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  box-shadow: 0 3px 10px var(--shadow);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--panel-2);
  border-left: 4px solid var(--accent);
  box-shadow: none;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stMarkdown p {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .88rem;
}

/* Chat input bar styling */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
  background: var(--panel) !important;
}
[data-testid="stChatInput"] {
  border: 1.5px solid var(--line);
  border-radius: 12px;
  box-shadow: 0 6px 18px var(--shadow);
}
[data-testid="stChatInput"] textarea {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: .88rem !important;
  color: var(--ink) !important;
}

/* --- Document Cards --- */
.doc-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1.3rem 1.5rem;
  margin-bottom: .9rem;
  box-shadow: 0 4px 16px var(--shadow);
  transition: transform .18s ease, box-shadow .18s ease;
  position: relative;
  overflow: hidden;
}
.doc-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  background: linear-gradient(180deg, var(--accent), #6c72ff);
  border-radius: 4px 0 0 4px;
}
.doc-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px var(--shadow);
}
.doc-card-header {
  display: flex;
  align-items: center;
  gap: .7rem;
  margin-bottom: .8rem;
}
.doc-card-icon {
  font-size: 1.6rem;
  background: var(--panel-2);
  border-radius: 12px;
  padding: .45rem .55rem;
  line-height: 1;
}
.doc-card-title {
  font-family: 'Archivo', sans-serif;
  font-weight: 700;
  font-size: 1.05rem;
  color: var(--ink);
  line-height: 1.25;
}
.doc-card-date {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem;
  color: var(--muted);
  margin-top: .15rem;
}
.doc-card-chips {
  display: flex;
  flex-wrap: wrap;
  gap: .5rem;
  margin-top: .1rem;
}
.doc-chip {
  display: inline-flex;
  align-items: center;
  gap: .3rem;
  background: var(--panel-2);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: .3rem .65rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem;
  font-weight: 500;
  color: var(--muted);
}
.doc-chip-val {
  color: var(--ink);
  font-weight: 600;
}
.doc-chip-accent {
  background: rgba(59,75,246,.1);
  border-color: rgba(59,75,246,.2);
}
.doc-chip-accent .doc-chip-val {
  color: var(--accent);
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--muted) !important;
  -webkit-text-fill-color: var(--muted);
  opacity: 1;
}
[data-testid="stBottom"] > div {
  background: transparent !important;
  padding: .5rem 0 !important;
}
"""


def inject_theme() -> None:
    palette = LIGHT
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in palette.items()) + "}"
    st.markdown(f"<style>{root}{STATIC_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- Application Init

st.set_page_config(page_title="DocBot — Document Intelligence", page_icon="📄", layout="wide")

st.session_state.setdefault("messages", [])
inject_theme()

# Fetch documents list from SQLite DB
stored_docs = list_documents()

with st.sidebar:
    st.markdown('<div class="side-brand">📄 Document Intelligence</div>', unsafe_allow_html=True)

# Hero Header
st.markdown(
    '<div class="hero">'
    '<div class="hero-icon">🔍</div>'
    "<div>"
    '<div class="hero-title">Document Intelligence</div>'
    '<div class="hero-sub">Ask anything about your indexed files — zero uploads required</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

def format_timestamp(iso_str: str) -> str:
    """Convert raw ISO timestamp to a clean, human-readable format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except (ValueError, TypeError):
        return iso_str


def get_file_icon(file_type: str) -> str:
    """Return emoji icon based on file type."""
    ft = file_type.lower()
    if "pdf" in ft:
        return "📕"
    elif any(x in ft for x in ["png", "jpg", "jpeg", "webp", "image"]):
        return "🖼️"
    elif "txt" in ft or "text" in ft:
        return "📝"
    return "📄"


# Navigation Bar (Native 33.3% Equal Width Layout)
st.session_state.setdefault("current_tab", "chat")

t_col1, t_col2, t_col3 = st.columns(3)
with t_col1:
    if st.button(
        "💬 Chat",
        use_container_width=True,
        type="primary" if st.session_state.current_tab == "chat" else "secondary",
        key="btn_chat_tab",
    ):
        st.session_state.current_tab = "chat"
        st.rerun()

with t_col2:
    if st.button(
        "📤 Upload Documents",
        use_container_width=True,
        type="primary" if st.session_state.current_tab == "upload" else "secondary",
        key="btn_upload_tab",
    ):
        st.session_state.current_tab = "upload"
        st.rerun()

with t_col3:
    if st.button(
        "📁 Stored Files",
        use_container_width=True,
        type="primary" if st.session_state.current_tab == "list" else "secondary",
        key="btn_list_tab",
    ):
        st.session_state.current_tab = "list"
        st.rerun()

# ================================================================= TAB 1: Chatbot
if st.session_state.current_tab == "chat":
    if not stored_docs:
        st.warning("⚠️ No documents in the database yet. Go to the **Upload Documents** tab to add files!")

    st.markdown('<div class="sec-label">Quick Ask</div>', unsafe_allow_html=True)
    q_cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if q_cols[i % 2].button(q, key=f"quick_{i}"):
            if stored_docs:
                st.session_state.pending_q = q
            else:
                st.toast("Please upload documents first!", icon="⚠️")

    # Display Chat History
    for m in st.session_state.messages:
        avatar = "🤖" if m["role"] == "assistant" else None
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # Chat Input
    user_prompt = st.chat_input(
        "Ask a question about your documents...",
        disabled=not stored_docs,
    )
    if not user_prompt:
        user_prompt = st.session_state.pop("pending_q", None)

    if user_prompt:
        api_key = os.getenv("GROQ_API_KEY", "")
        client = get_groq_client(api_key)
        if client is None:
            st.error("Enter your Groq API key in the sidebar.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")

            docs_text = get_relevant_context(user_prompt)
            if not docs_text:
                docs_text = "(No matching document sections found in database for your query.)"
                
            system_prompt = (
                "You are a precise document assistant. Answer the user's question using ONLY the document contents below.\n\n"
                "Rules:\n"
                "- Quote values (dates, IDs, amounts, locations, names) exactly as written.\n"
                "- Mention which document the information came from.\n"
                "- If not found in any document, say so plainly.\n\n"
                f"RELEVANT DOCUMENTS IN DATABASE:\n{docs_text}"
            )

            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
            ]

            try:
                stream = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                )
                
                def delta_generator():
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta

                placeholder.empty()
                answer = st.write_stream(delta_generator())
            except Exception as e:
                placeholder.empty()
                st.error(f"Groq request failed: {e}")
                st.stop()

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(
            """
            <script>
                setTimeout(() => {
                    const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
                    if (chatMessages.length > 0) {
                        chatMessages[chatMessages.length - 1].scrollIntoView({behavior: 'smooth', block: 'start'});
                    }
                }, 100);
            </script>
            """,
            unsafe_allow_html=True,
        )

# ================================================================= TAB 2: Upload Document
elif st.session_state.current_tab == "upload":
    st.subheader("📤 Upload New Document")

    uploaded_file = st.file_uploader(
        "Select or drag & drop a file to index into Database",
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
        key="admin_uploader",
    )
    if uploaded_file:
        if st.button("💾 Save & Index to Database", type="primary", use_container_width=True):
            api_key = os.getenv("GROQ_API_KEY", "")
            client = get_groq_client(api_key)
            content = uploaded_file.getvalue()
            with st.spinner("Parsing text layer & Vision OCR..."):
                text, kind = extract_file_content(client, content, uploaded_file.name, uploaded_file.type)
                doc = save_document(
                    filename=uploaded_file.name,
                    file_type=kind,
                    file_size_kb=len(content) / 1024.0,
                    extracted_text=text,
                )
                st.success(f"✓ Saved '{doc['filename']}' ({doc['char_count']:,} characters) to Database!")
                st.toast(f"Indexed {doc['filename']}", icon="✅")
                st.rerun()

# ================================================================= TAB 3: Stored Files
elif st.session_state.current_tab == "list":
    if not stored_docs:
        st.info("The database is currently empty. Use the **Upload Documents** tab to add files.")
    else:
        # Summary metrics row
        m1, m2, m3 = st.columns(3)
        total_chars = sum(d['char_count'] for d in stored_docs)
        total_size = sum(d['file_size_kb'] for d in stored_docs)
        m1.metric("📚 Documents", len(stored_docs))
        m2.metric("📊 Total Characters", f"{total_chars:,}")
        m3.metric("💾 Total Size", f"{total_size:.1f} KB")

        st.markdown("<br>", unsafe_allow_html=True)

        # Render each document as a styled card
        for doc in stored_docs:
            icon = get_file_icon(doc['file_type'])
            nice_date = format_timestamp(doc['created_at'])

            card_html = f"""
            <div class="doc-card">
              <div class="doc-card-header">
                <div class="doc-card-icon">{icon}</div>
                <div>
                  <div class="doc-card-title">{doc['filename']}</div>
                  <div class="doc-card-date">📅 {nice_date}</div>
                </div>
              </div>
              <div class="doc-card-chips">
                <span class="doc-chip doc-chip-accent">
                  🏷️ <span class="doc-chip-val">{doc['file_type'].upper()}</span>
                </span>
                <span class="doc-chip">
                  💾 <span class="doc-chip-val">{doc['file_size_kb']:.1f} KB</span>
                </span>
                <span class="doc-chip">
                  ✏️ <span class="doc-chip-val">{doc['char_count']:,} chars</span>
                </span>
                <span class="doc-chip">
                  🔑 ID <span class="doc-chip-val">#{doc['id']}</span>
                </span>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Delete button (Streamlit widget, placed outside HTML)
            if st.button(f"🗑️ Delete {doc['filename']}", key=f"del_{doc['id']}", type="secondary"):
                delete_document(doc['id'])
                st.toast(f"Deleted {doc['filename']}", icon="🗑️")
                st.rerun()
