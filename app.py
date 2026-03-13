"""
Kakarot Study — RAG Chatbot
Pure conversational chat interface like Claude
"""

import streamlit as st
import os
from rag_engine import RAGPipeline

SERVER_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
PUBLIC_MODE = bool(SERVER_API_KEY)

st.set_page_config(
    page_title="Kakarot Study ⚡",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0e1a;
    --bg2: #111827;
    --card: #161d2e;
    --border: #1e293b;
    --orange: #f97316;
    --yellow: #facc15;
    --green: #4ade80;
    --blue: #4f9cf9;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --dim: #475569;
}

.stApp { background: var(--bg) !important; font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Header bar ── */
.topbar {
    background: linear-gradient(135deg, #0f172a, #1c1308, #0f172a);
    border-bottom: 1px solid rgba(249,115,22,0.2);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.topbar-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f97316, #facc15);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.topbar-sub { font-size: 0.8rem; color: var(--dim); margin-top: 2px; }

/* ── Chat area ── */
.chat-area {
    padding: 24px 32px 16px 32px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* User bubble */
.msg-user {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    align-items: flex-start;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #1e3a5f, #1a2d4a);
    border: 1px solid rgba(79,156,249,0.25);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    max-width: 70%;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.65;
}
.msg-user .avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #1e3a5f, #3b82f6);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0; margin-top: 2px;
}

/* Bot bubble */
.msg-bot {
    display: flex;
    justify-content: flex-start;
    gap: 10px;
    align-items: flex-start;
}
.msg-bot .bubble {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    max-width: 75%;
    color: var(--text);
    font-size: 0.93rem;
    line-height: 1.75;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.msg-bot .bubble strong { color: var(--orange) !important; }
.msg-bot .bubble code {
    background: rgba(249,115,22,0.1) !important;
    color: var(--green) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.88rem !important;
}
.msg-bot .avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #f97316, #facc15);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0; margin-top: 2px;
}

/* Priority badge */
.badge-high {
    display: inline-block;
    background: rgba(244,63,94,0.15);
    border: 1px solid rgba(244,63,94,0.3);
    color: #f43f5e !important;
    font-size: 0.7rem; font-weight: 700;
    padding: 2px 8px; border-radius: 100px;
    margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-medium {
    display: inline-block;
    background: rgba(245,158,11,0.15);
    border: 1px solid rgba(245,158,11,0.3);
    color: #f59e0b !important;
    font-size: 0.7rem; font-weight: 700;
    padding: 2px 8px; border-radius: 100px;
    margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;
}

/* Welcome screen */
.welcome {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 32px; text-align: center; gap: 16px;
}
.welcome-icon { font-size: 3rem; margin-bottom: 8px; }
.welcome h2 {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(90deg, #f97316, #facc15);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.welcome p { color: var(--muted); font-size: 0.95rem; line-height: 1.7; margin: 0; max-width: 480px; }
.sample-questions {
    display: flex; flex-direction: column; gap: 8px;
    margin-top: 8px; width: 100%; max-width: 500px;
}
.sample-q {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 16px;
    font-size: 0.87rem;
    color: var(--muted);
    text-align: left;
    cursor: default;
}
.sample-q span { color: var(--orange); margin-right: 8px; }

/* Input box */
[data-testid="stChatInput"] {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.1) !important;
}

/* Sidebar */
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem; font-weight: 700;
    color: var(--dim) !important;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 12px;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem; font-weight: 800;
    color: var(--orange);
}
.stat-label { font-size: 0.7rem; color: var(--dim); text-transform: uppercase; letter-spacing: 0.8px; }
.doc-info {
    background: rgba(249,115,22,0.05);
    border: 1px solid rgba(249,115,22,0.15);
    border-radius: 10px; padding: 12px;
    font-size: 0.83rem; color: var(--muted) !important; line-height: 1.6;
}
.tip-box {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.15);
    border-left: 3px solid #10b981;
    border-radius: 8px; padding: 12px 14px;
    font-size: 0.83rem; color: var(--muted) !important; line-height: 1.6;
    margin-bottom: 8px;
}

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed rgba(249,115,22,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(249,115,22,0.6) !important; }

.stButton > button {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    padding: 10px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

[data-testid="stAlert"] { background: var(--card) !important; border-radius: 10px !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "pipeline": None,
        "chat_history": [],
        "doc_info": None,
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if PUBLIC_MODE and not st.session_state.api_key_set:
        st.session_state.pipeline = RAGPipeline(api_key=SERVER_API_KEY)
        st.session_state.api_key_set = True

init_session()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 12px 0;'>
        <div style='font-size:2.2rem;'>⚡</div>
        <div style='font-family:Syne,sans-serif; font-size:1.2rem; font-weight:800;
             background:linear-gradient(90deg,#f97316,#facc15);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            Kakarot Study
        </div>
        <div style='font-size:0.7rem; color:#475569; margin-top:3px;'>
            AI Question Paper Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API key (only if not server-side)
    if not PUBLIC_MODE:
        st.markdown('<div class="sidebar-title">🔑 Gemini API Key</div>', unsafe_allow_html=True)
        api_key_input = st.text_input("API Key", type="password",
            placeholder="AIzaSy...", label_visibility="collapsed",
            help="Free key from aistudio.google.com")
        if api_key_input:
            if not st.session_state.api_key_set or api_key_input != getattr(st.session_state, '_last_key', ''):
                st.session_state._last_key = api_key_input
                st.session_state.pipeline = RAGPipeline(api_key=api_key_input)
                st.session_state.api_key_set = True
            st.success("✓ API key ready")
        else:
            st.warning("Enter your Gemini API key")
        st.markdown("---")
    else:
        st.markdown("""
        <div style="background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2);
             border-radius:10px; padding:10px 14px; font-size:0.82rem;
             color:#94a3b8; margin-bottom:12px;">
            ✅ <strong style="color:#4ade80;">Free to use</strong><br/>
            <span style="font-size:0.75rem; color:#475569;">Powered by Google Gemini</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    # Upload
    st.markdown('<div class="sidebar-title">📁 Upload File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        help="PDF or photo of your question paper")

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        is_image = ext in ["jpg", "jpeg", "png", "webp"]
        if is_image:
            st.image(uploaded_file, use_column_width=True)
            uploaded_file.seek(0)
        else:
            st.markdown(f"""
            <div style="background:#161d2e; border:1px solid #1e293b; border-radius:8px;
                 padding:10px 14px; font-size:0.82rem; color:#94a3b8; margin:6px 0;">
                📄 <strong style="color:#f97316;">{uploaded_file.name}</strong>
            </div>""", unsafe_allow_html=True)

    if uploaded_file and st.session_state.api_key_set:
        btn_label = "⚡ Read Image" if is_image else "⚡ Process PDF"
        spinner_msg = "🔍 Reading image with Gemini..." if is_image else "📖 Indexing your PDF..."
        if st.button(btn_label, use_container_width=True):
            with st.spinner(spinner_msg):
                result = st.session_state.pipeline.ingest(uploaded_file)
            if result["success"]:
                st.session_state.doc_info = result
                st.session_state.chat_history = []
                st.success(f"✓ Ready! {result['chunks']} chunks indexed")
                st.rerun()
            else:
                st.error(result["error"])

    # Doc stats
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown("---")
        st.markdown('<div class="sidebar-title">📊 Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pval = "🖼" if info.get("file_type") == "image" else info["pages"]
            st.markdown(f'<div class="stat-card"><div class="stat-val">{pval}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{info["chunks"]}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
        if info.get("summary"):
            st.markdown('<div class="sidebar-title" style="margin-top:12px;">📝 About</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="doc-info">{info["summary"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">Say <strong style="color:#f43f5e;">very important</strong> to get a deeper, exam-ready answer</div>
    <div class="tip-box">Ask for <strong style="color:#4ade80;">tricks</strong> or <strong style="color:#4ade80;">analogies</strong> to understand faster</div>
    <div class="tip-box">Ask <strong style="color:#4f9cf9;">anything</strong> — just chat naturally!</div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─── Main Chat Area ───────────────────────────────────────────────────────────

# Top bar
st.markdown("""
<div class="topbar">
    <div style="font-size:1.5rem;">⚡</div>
    <div>
        <div class="topbar-title">Kakarot Study</div>
        <div class="topbar-sub">Upload your question paper → Chat to understand anything</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat messages or welcome screen
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    if not st.session_state.doc_info:
        # No doc uploaded yet
        st.markdown("""
        <div class="welcome">
            <div class="welcome-icon">📚</div>
            <h2>Hey! Upload a file to start</h2>
            <p>Upload your question paper PDF or a photo of it from the sidebar,
            then just chat with me — ask anything, exactly like you're texting a friend.</p>
            <div class="sample-questions">
                <div class="sample-q"><span>💬</span>Explain question 3 in simple words</div>
                <div class="sample-q"><span>🧠</span>Give me a trick to remember this concept</div>
                <div class="sample-q"><span>🎯</span>What are the most important topics? (very important)</div>
                <div class="sample-q"><span>📝</span>List all questions in this paper</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Doc uploaded, no chat yet
        st.markdown("""
        <div class="welcome">
            <div class="welcome-icon">✅</div>
            <h2>Document ready! Ask me anything</h2>
            <p>Just type your question below — like you're chatting with a friend.<br/>
            I'll explain clearly with tricks and shortcuts.</p>
            <div class="sample-questions">
                <div class="sample-q"><span>💬</span>Explain question 3 in simple words</div>
                <div class="sample-q"><span>🧠</span>Give me a trick to remember the key concepts</div>
                <div class="sample-q"><span>🎯</span>What topics are most important? this is very important</div>
                <div class="sample-q"><span>📝</span>List all questions in this paper</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Render chat history
    for turn in st.session_state.chat_history:
        if turn["role"] == "user":
            importance = turn.get("importance", "normal")
            badge = ""
            if importance == "high":
                badge = '<div class="badge-high">🔴 High Priority</div><br/>'
            elif importance == "medium":
                badge = '<div class="badge-medium">🟡 Important</div><br/>'
            st.markdown(f"""
            <div class="msg-user">
                <div class="bubble">{badge}{turn["content"]}</div>
                <div class="avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Format bot response — convert markdown-ish to readable HTML
            answer = turn["content"] \
                .replace("\n\n", "<br/><br/>") \
                .replace("\n", "<br/>")
            st.markdown(f"""
            <div class="msg-bot">
                <div class="avatar">⚡</div>
                <div class="bubble">{answer}</div>
            </div>
            """, unsafe_allow_html=True)
            if turn.get("context_used"):
                with st.expander(f"📎 {len(turn['context_used'])} source excerpts"):
                    for i, (chunk, score) in enumerate(turn["context_used"], 1):
                        color = "#10b981" if score > 0.3 else "#f59e0b" if score > 0.15 else "#475569"
                        st.markdown(f"""
                        <div style="background:#161d2e; border:1px solid #1e293b; border-radius:8px;
                             padding:10px 14px; margin-bottom:6px; font-size:0.8rem;
                             color:#94a3b8; line-height:1.6;">
                            <span style="color:{color}; font-weight:700; font-size:0.72rem;">
                                ◆ Excerpt {i} — {score:.0%} match
                            </span><br/>{chunk[:350]}{"..." if len(chunk)>350 else ""}
                        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────
if not st.session_state.api_key_set:
    st.markdown("""
    <div style="margin:16px 32px; background:#1e1b4b; border:1px solid rgba(168,85,247,0.3);
         border-radius:12px; padding:14px 20px; text-align:center; color:#94a3b8; font-size:0.9rem;">
        👈 Enter your <strong style="color:#4ade80">Gemini API key</strong> in the sidebar
    </div>""", unsafe_allow_html=True)
elif not st.session_state.doc_info:
    st.markdown("""
    <div style="margin:16px 32px; background:#1e293b; border:1px solid rgba(249,115,22,0.2);
         border-radius:12px; padding:14px 20px; text-align:center; color:#94a3b8; font-size:0.9rem;">
        👈 Upload a <strong style="color:#f97316">PDF or photo</strong> in the sidebar first
    </div>""", unsafe_allow_html=True)
else:
    user_input = st.chat_input("Ask anything... e.g. 'explain question 2' or 'give me a trick for this topic'")

    if user_input and user_input.strip():
        with st.spinner("⚡ Thinking..."):
            result = st.session_state.pipeline.query(
                user_query=user_input.strip(),
                chat_history=st.session_state.chat_history,
                top_k=5
            )
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "importance": result["importance"]
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "context_used": result["context_used"]
        })
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px; font-size:0.72rem; color:#1e293b;">
    ⚡ Kakarot Study · Google Gemini · Made for smart learners
</div>
""", unsafe_allow_html=True)
