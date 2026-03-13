"""
StudyBot — RAG-powered Question Paper Understanding Chatbot
Built with LangChain + Streamlit + Google Gemini
Deployable as a public website — supports server-side API key via env var
"""

import streamlit as st
import os
from rag_engine import RAGPipeline

# ─── Server-side API key support ────────────────────────────────────────────
# If GOOGLE_API_KEY is set in the deployment env (Railway/Render secrets),
# visitors use the app for free without entering their own key.
SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
PUBLIC_MODE = bool(SERVER_API_KEY)

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kakarot Study — AI Question Paper Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* Root variables */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #161d2e;
    --accent-blue: #4f9cf9;
    --accent-purple: #a855f7;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: #1e293b;
    --glow: rgba(79, 156, 249, 0.15);
}

/* Global reset */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Main container */
.main .block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1100px !important;
}

/* ── Header ── */
.studybot-header {
    background: linear-gradient(135deg, #0f172a 0%, #1c1308 50%, #0f172a 100%);
    border: 1px solid rgba(249, 115, 22, 0.2);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.studybot-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(249,115,22,0.09) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(250,204,21,0.07) 0%, transparent 60%);
    pointer-events: none;
}
.studybot-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #f97316, #facc15);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 6px 0 !important;
    line-height: 1.1 !important;
}
.studybot-header p {
    color: var(--text-secondary) !important;
    margin: 0 !important;
    font-size: 0.95rem !important;
    font-weight: 300 !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed rgba(79, 156, 249, 0.35) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79, 156, 249, 0.7) !important;
}
[data-testid="stFileUploaderDropzone"] label {
    color: var(--text-secondary) !important;
}

/* ── Chat messages ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}

.msg-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
    border: 1px solid rgba(79, 156, 249, 0.25);
    border-radius: 16px 16px 4px 16px;
    padding: 16px 20px;
    margin-left: 10%;
    color: var(--text-primary) !important;
    font-size: 0.95rem;
    line-height: 1.6;
    position: relative;
}
.msg-user::before {
    content: '👤';
    position: absolute;
    top: 12px;
    right: -36px;
    font-size: 1.3rem;
}

.msg-bot {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 20px 24px;
    margin-right: 5%;
    color: var(--text-primary) !important;
    font-size: 0.93rem;
    line-height: 1.75;
    position: relative;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}
.msg-bot::before {
    content: '🎓';
    position: absolute;
    top: 12px;
    left: -38px;
    font-size: 1.3rem;
}
.msg-bot strong { color: var(--accent-blue) !important; }
.msg-bot code {
    background: rgba(79,156,249,0.1) !important;
    color: var(--accent-emerald) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.88rem !important;
}

/* ── Importance badges ── */
.badge-high {
    display: inline-block;
    background: linear-gradient(90deg, #f43f5e22, #f43f5e44);
    border: 1px solid #f43f5e66;
    color: #f43f5e !important;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 100px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-medium {
    display: inline-block;
    background: linear-gradient(90deg, #f59e0b22, #f59e0b33);
    border: 1px solid #f59e0b55;
    color: #f59e0b !important;
    font-size: 0.73rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 100px;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Stats cards ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    transition: transform 0.15s, border-color 0.15s;
}
.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(79,156,249,0.3);
}
.stat-card .stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--accent-blue);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-card .stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Input box ── */
[data-testid="stTextInput"] input, 
[data-testid="stChatInput"] textarea,
textarea {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,156,249,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Tips section */
.tip-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03));
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--accent-emerald);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.87rem;
    color: var(--text-secondary) !important;
}

/* Source context expander */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Sidebar content */
.sidebar-doc-info {
    background: rgba(79,156,249,0.06);
    border: 1px solid rgba(79,156,249,0.15);
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
    font-size: 0.84rem;
    color: var(--text-secondary) !important;
    line-height: 1.6;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

/* Success / error alerts */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Welcome card */
.welcome-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    margin: 20px 0;
}
.welcome-card h3 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 12px !important;
}
.welcome-card p {
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 24px;
    text-align: left;
}
.feature-item {
    background: rgba(79,156,249,0.05);
    border: 1px solid rgba(79,156,249,0.12);
    border-radius: 10px;
    padding: 14px;
    font-size: 0.84rem;
    color: var(--text-secondary) !important;
}
.feature-item span {
    font-size: 1.2rem;
    margin-right: 8px;
}
.feature-item strong {
    color: var(--text-primary) !important;
    display: block;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "pipeline": None,
        "chat_history": [],
        "doc_info": None,
        "api_key_set": False,
        "processing": False,
        "pending_suggestion": None,   # ← stores button click across rerun
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-init pipeline if server key is available
    if PUBLIC_MODE and not st.session_state.api_key_set:
        st.session_state.pipeline = RAGPipeline(api_key=SERVER_API_KEY)
        st.session_state.api_key_set = True

init_session()

# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-size:2.5rem;'>⚡</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800; 
             background:linear-gradient(90deg,#f97316,#facc15); 
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Kakarot Study
        </div>
        <div style='font-size:0.72rem; color:#475569; margin-top:4px;'>
            AI-Powered Question Paper Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API Key — only shown when NOT in public/hosted mode
    if not PUBLIC_MODE:
        st.markdown('<div class="sidebar-title">🔑 Gemini API Key</div>', unsafe_allow_html=True)
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="AIzaSy...",
            label_visibility="collapsed",
            help="Free key at aistudio.google.com"
        )
        if api_key_input:
            if not st.session_state.api_key_set or api_key_input != getattr(st.session_state, '_last_key', ''):
                st.session_state._last_key = api_key_input
                st.session_state.pipeline = RAGPipeline(api_key=api_key_input)
                st.session_state.api_key_set = True
            st.success("✓ API key set")
        else:
            st.warning("Enter your API key to start")
        st.markdown("---")
    else:
        # Public mode — show a friendly badge instead
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(74,222,128,0.10),rgba(250,204,21,0.07));
             border:1px solid rgba(74,222,128,0.25); border-radius:10px;
             padding:10px 14px; font-size:0.82rem; color:#94a3b8; margin-bottom:12px;">
            ⚡ <strong style="color:#4ade80;">Free to use</strong> — No API key needed!<br/>
            <span style="font-size:0.75rem; color:#475569;">Powered by Google Gemini (Free)</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("---")

    # PDF Upload
    st.markdown('<div class="sidebar-title">📁 Upload Document or Photo</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="Upload a question paper PDF or a photo of a question paper"
    )

    if uploaded_file:
        # Show image preview if it's an image
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "webp"]:
            st.image(uploaded_file, caption=f"📷 {uploaded_file.name}", use_column_width=True)
            uploaded_file.seek(0)
            btn_label = "⚡ Process Image"
            spinner_msg = "🔍 Reading image with Gemini Vision..."
        else:
            st.markdown(f"""
            <div style="background:#161d2e; border:1px solid #1e293b; border-radius:8px;
                 padding:10px 14px; font-size:0.82rem; color:#94a3b8; margin-top:6px;">
                📄 <strong style="color:#f97316;">{uploaded_file.name}</strong>
            </div>
            """, unsafe_allow_html=True)
            btn_label = "⚡ Process PDF"
            spinner_msg = "📖 Reading and indexing your PDF..."

    if uploaded_file and st.session_state.api_key_set:
        if st.button(btn_label, use_container_width=True):
            with st.spinner(spinner_msg):
                result = st.session_state.pipeline.ingest(uploaded_file)
                if result["success"]:
                    st.session_state.doc_info = result
                    st.session_state.chat_history = []
                    file_type = result.get("file_type", "pdf")
                    if file_type == "image":
                        st.success(f"✓ Image read! Extracted text indexed into {result['chunks']} chunks")
                    else:
                        st.success(f"✓ Indexed {result['chunks']} chunks from {result['pages']} pages")
                else:
                    st.error(result["error"])

    # Document info
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown("---")
        file_type = info.get("file_type", "pdf")
        stat_label = "Image" if file_type == "image" else "Pages"
        stat_val   = "📷" if file_type == "image" else info['pages']
        st.markdown('<div class="sidebar-title">📊 Document Stats</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{stat_val}</div>
                <div class="stat-label">{stat_label}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{info['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)

        if info.get("summary"):
            st.markdown('<div class="sidebar-title" style="margin-top:12px;">📝 Document Preview</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sidebar-doc-info">{info["summary"]}</div>', unsafe_allow_html=True)

        # Show extracted text preview for images
        if file_type == "image" and info.get("extracted_text"):
            with st.expander("🔍 Extracted text preview"):
                st.markdown(f"""
                <div style="font-size:0.78rem; color:#64748b; font-family:monospace;
                     line-height:1.6; padding:4px;">
                    {info['extracted_text']}...
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Tips
    st.markdown('<div class="sidebar-title">💡 Usage Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">
        Mark questions as <strong style="color:#f43f5e">important</strong> or 
        <strong style="color:#f59e0b">very important</strong> to get deeper explanations 
        and exam-focused answers.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box" style="margin-top:6px;">
        Ask for <strong>tricks</strong>, <strong>shortcuts</strong>, 
        or <strong>analogies</strong> to understand tough concepts faster.
    </div>
    """, unsafe_allow_html=True)

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─── Main Content ────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="studybot-header">
    <h1>⚡ Kakarot Study — AI Question Paper Assistant</h1>
    <p>Upload a question paper PDF or snap a photo of it → Ask anything → Get crystal-clear explanations with tricks & shortcuts</p>
</div>
""", unsafe_allow_html=True)

# ── Welcome state ──
if not st.session_state.doc_info:
    st.markdown("""
    <div class="welcome-card">
        <h3>Ready to make studying effortless 🚀</h3>
        <p>Upload a question paper PDF <strong>or a photo</strong> of your question paper<br>
        in the sidebar, then ask questions to get smart explanations.</p>
        <div class="feature-grid">
            <div class="feature-item">
                <span>📄</span>
                <strong>PDF Support</strong>
                Upload any question paper or textbook PDF
            </div>
            <div class="feature-item">
                <span>📷</span>
                <strong>Photo Support</strong>
                Snap a photo of your question paper — Gemini reads it!
            </div>
            <div class="feature-item">
                <span>🧠</span>
                <strong>Memory Tricks</strong>
                Mnemonics and shortcuts for every concept
            </div>
            <div class="feature-item">
                <span>🎯</span>
                <strong>Importance Mode</strong>
                Deeper answers for critical topics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat display ──
if st.session_state.chat_history:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            importance = turn.get("importance", "normal")
            badge = ""
            if importance == "high":
                badge = '<div class="badge-high">🔴 High Priority</div>'
            elif importance == "medium":
                badge = '<div class="badge-medium">🟡 Medium Priority</div>'
            st.markdown(f"""
            <div class="msg-user">
                {badge}
                {turn["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-bot">
                {turn["content"]}
            </div>
            """, unsafe_allow_html=True)
            # Show retrieved sources
            if turn.get("context_used"):
                with st.expander(f"📎 {len(turn['context_used'])} source excerpts used"):
                    for i, (chunk, score) in enumerate(turn["context_used"], 1):
                        relevance_color = "#10b981" if score > 0.3 else "#f59e0b" if score > 0.15 else "#475569"
                        st.markdown(f"""
                        <div style="background:#161d2e; border:1px solid #1e293b; border-radius:8px; 
                             padding:12px; margin-bottom:8px; font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                            <span style="color:{relevance_color}; font-weight:600; font-size:0.75rem;">
                                ◆ Excerpt {i} — relevance {score:.0%}
                            </span><br/>
                            {chunk[:350]}{"..." if len(chunk) > 350 else ""}
                        </div>
                        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Input area ──
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Check prerequisites
if not st.session_state.api_key_set:
    st.markdown("""
    <div style="background:#1e1b4b; border:1px solid rgba(168,85,247,0.3); border-radius:12px; 
         padding:16px 20px; text-align:center; color:#94a3b8; font-size:0.9rem;">
        👈 Enter your <strong style="color:#4ade80">Gemini API key</strong> in the sidebar to get started
    </div>
    """, unsafe_allow_html=True)
elif not st.session_state.doc_info:
    st.markdown("""
    <div style="background:#1e293b; border:1px solid rgba(249,115,22,0.2); border-radius:12px; 
         padding:16px 20px; text-align:center; color:#94a3b8; font-size:0.9rem;">
        👈 Upload and process a <strong style="color:#f97316">PDF document</strong> in the sidebar to begin
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Suggestion chips (fixed with session_state) ──────────────────────
    st.markdown("""
    <div style="margin-bottom:10px; font-size:0.78rem; color:#475569;">
        💬 Try a quick question:
    </div>
    """, unsafe_allow_html=True)

    chips_col1, chips_col2, chips_col3, chips_col4 = st.columns(4)

    with chips_col1:
        if st.button("📌 List all questions", use_container_width=True):
            st.session_state.pending_suggestion = "List all the questions present in this document"
            st.rerun()
    with chips_col2:
        if st.button("💡 Explain concepts", use_container_width=True):
            st.session_state.pending_suggestion = "Explain the key concepts in this document with simple analogies"
            st.rerun()
    with chips_col3:
        if st.button("🧠 Give me tricks", use_container_width=True):
            st.session_state.pending_suggestion = "What are memory tricks and shortcuts for the important topics in this paper?"
            st.rerun()
    with chips_col4:
        if st.button("⚠️ Important topics", use_container_width=True):
            st.session_state.pending_suggestion = "What are the most important and frequently asked topics? This is very important for my exam"
            st.rerun()

    # ── Chat input ───────────────────────────────────────────────────────
    user_input = st.chat_input("Ask me anything about your document... e.g. 'explain question 3' or 'give me a trick for this topic'")

    # Pick up either typed input or button suggestion
    final_input = user_input or st.session_state.get("pending_suggestion")

    if final_input and final_input.strip():
        # Clear the pending suggestion immediately
        st.session_state.pending_suggestion = None

        with st.spinner("⚡ Thinking..."):
            result = st.session_state.pipeline.query(
                user_query=final_input.strip(),
                chat_history=st.session_state.chat_history,
                top_k=5
            )

        st.session_state.chat_history.append({
            "role": "user",
            "content": final_input.strip(),
            "importance": result["importance"]
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "context_used": result["context_used"]
        })
        st.rerun()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:40px; padding-top:20px; 
     border-top:1px solid #1e293b; font-size:0.75rem; color:#334155;">
    ⚡ Kakarot Study · Powered by LangChain + Google Gemini · Made for smart learners
</div>
""", unsafe_allow_html=True)
