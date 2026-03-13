"""
Kakarot Study — All-in-one AI Study App
Navigation uses st.radio (stateful) — zero button-click issues
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
    --bg:#0a0e1a; --bg2:#111827; --card:#161d2e; --border:#1e293b;
    --orange:#f97316; --yellow:#facc15; --green:#4ade80;
    --blue:#4f9cf9; --purple:#a855f7; --red:#f43f5e;
    --text:#f1f5f9; --muted:#94a3b8; --dim:#475569;
}

/* ── Base ── */
.stApp { background:var(--bg) !important; font-family:'DM Sans',sans-serif !important; }
#MainMenu, footer, header { visibility:hidden !important; }
.stDeployButton { display:none !important; }
[data-testid="stSidebar"] { background:var(--bg2) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color:var(--text) !important; }
.main .block-container { padding:1.2rem 1.8rem !important; max-width:100% !important; }

/* ── Sidebar nav radio — this IS the page switcher ── */
div[data-testid="stSidebar"] [data-testid="stRadio"] {
    background: transparent !important;
}
div[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    cursor: pointer !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    display: flex !important;
    width: 100% !important;
}
div[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    border-color: rgba(249,115,22,0.4) !important;
    background: rgba(249,115,22,0.06) !important;
}
div[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"],
div[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div {
    border-color: rgba(249,115,22,0.5) !important;
    background: rgba(249,115,22,0.1) !important;
    color: var(--orange) !important;
}
/* hide the actual radio dot */
div[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── Header ── */
.topbar {
    background: linear-gradient(135deg,#0f172a,#1c1308,#0f172a);
    border-bottom: 1px solid rgba(249,115,22,0.2);
    padding: 16px 22px; display:flex; align-items:center; gap:12px;
    border-radius: 12px; margin-bottom: 16px;
}
.topbar-title {
    font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
    background:linear-gradient(90deg,#f97316,#facc15);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.topbar-sub { font-size:0.75rem; color:var(--dim); margin-top:2px; }

/* ── Chat bubbles ── */
.msg-user { display:flex; justify-content:flex-end; gap:10px; align-items:flex-start; margin-bottom:16px; }
.msg-user .bubble {
    background:linear-gradient(135deg,#1e3a5f,#1a2d4a);
    border:1px solid rgba(79,156,249,0.25); border-radius:18px 18px 4px 18px;
    padding:13px 17px; max-width:72%; color:var(--text); font-size:0.93rem; line-height:1.65;
}
.msg-user .av { width:32px;height:32px;background:linear-gradient(135deg,#1e3a5f,#3b82f6);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.8rem;flex-shrink:0;margin-top:2px; }
.msg-bot { display:flex; justify-content:flex-start; gap:10px; align-items:flex-start; margin-bottom:16px; }
.msg-bot .bubble {
    background:var(--card); border:1px solid var(--border); border-radius:18px 18px 18px 4px;
    padding:15px 19px; max-width:78%; color:var(--text); font-size:0.92rem; line-height:1.75;
    box-shadow:0 4px 20px rgba(0,0,0,0.2);
}
.msg-bot .bubble strong { color:var(--orange) !important; }
.msg-bot .av { width:32px;height:32px;background:linear-gradient(135deg,#f97316,#facc15);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0;margin-top:2px; }
.badge-high { display:inline-block;background:rgba(244,63,94,0.15);border:1px solid rgba(244,63,94,0.3);color:#f43f5e!important;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:100px;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px; }
.badge-medium { display:inline-block;background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);color:#f59e0b!important;font-size:0.68rem;font-weight:700;padding:2px 8px;border-radius:100px;margin-bottom:5px;text-transform:uppercase;letter-spacing:.5px; }

/* ── Welcome screen ── */
.welcome { text-align:center; padding:40px 20px; }
.welcome h2 { font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;background:linear-gradient(90deg,#f97316,#facc15);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px; }
.welcome p { color:var(--muted);font-size:0.92rem;line-height:1.7;max-width:460px;margin:0 auto 18px; }
.sample-q { background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 15px;font-size:0.85rem;color:var(--muted);text-align:left;margin-bottom:8px; }
.sample-q span { color:var(--orange);margin-right:8px; }

/* ── Quiz ── */
.quiz-header { font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--text);margin-bottom:4px; }
.quiz-sub { font-size:0.85rem;color:var(--muted);margin-bottom:20px; }
.q-card { background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin-bottom:6px; }
.q-num { font-size:0.7rem;font-weight:700;color:var(--orange);text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px; }
.q-text { font-size:0.96rem;color:var(--text);font-weight:500;line-height:1.5; }
.explanation { background:rgba(79,156,249,0.07);border-left:3px solid var(--blue);border-radius:6px;padding:10px 14px;font-size:0.83rem;color:var(--muted);margin-top:10px;line-height:1.6; }
.score-card { background:linear-gradient(135deg,rgba(249,115,22,0.1),rgba(250,204,21,0.08));border:1px solid rgba(249,115,22,0.25);border-radius:16px;padding:28px;text-align:center;margin-bottom:20px; }
.score-big { font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(90deg,#f97316,#facc15);-webkit-background-clip:text;-webkit-text-fill-color:transparent; }

/* ── Coming soon ── */
.coming-soon { text-align:center;padding:70px 20px; }
.coming-soon h2 { font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--text);margin:12px 0 8px; }
.coming-soon p { color:var(--muted);font-size:0.9rem; }

/* ── Sidebar helpers ── */
.sidebar-title { font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;color:var(--dim)!important;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px; }
.stat-card { background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center; }
.stat-val { font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:var(--orange); }
.stat-label { font-size:0.68rem;color:var(--dim);text-transform:uppercase;letter-spacing:.8px; }
.doc-info { background:rgba(249,115,22,0.05);border:1px solid rgba(249,115,22,0.15);border-radius:10px;padding:12px;font-size:0.82rem;color:var(--muted)!important;line-height:1.6; }
.tip-box { background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);border-left:3px solid #10b981;border-radius:8px;padding:11px 13px;font-size:0.82rem;color:var(--muted)!important;line-height:1.6;margin-bottom:8px; }

/* ── Inputs ── */
[data-testid="stFileUploader"] { background:var(--card)!important;border:1.5px dashed rgba(249,115,22,0.3)!important;border-radius:12px!important; }
.stButton > button { background:linear-gradient(135deg,#f97316,#ea580c)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:0.87rem!important;padding:10px!important; }
[data-testid="stChatInput"] { background:var(--card)!important;border:1.5px solid var(--border)!important;border-radius:14px!important; }
[data-testid="stChatInput"] textarea { background:transparent!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;font-size:0.94rem!important; }
[data-testid="stChatInput"]:focus-within { border-color:var(--orange)!important;box-shadow:0 0 0 3px rgba(249,115,22,0.1)!important; }
[data-testid="stAlert"] { background:var(--card)!important;border-radius:10px!important; }
[data-testid="stNumberInput"] input { background:var(--card)!important;color:var(--text)!important;border-color:var(--border)!important; }
div[data-baseweb="select"] { background:var(--card)!important; }
div[data-baseweb="select"] * { color:var(--text)!important;background:var(--card)!important; }
hr { border-color:var(--border)!important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg2); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
def init():
    defs = {
        "pipeline": None, "chat_history": [], "doc_info": None,
        "api_key_set": False,
        "quiz_questions": [], "quiz_answers": {}, "quiz_submitted": False,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if PUBLIC_MODE and not st.session_state.api_key_set:
        st.session_state.pipeline = RAGPipeline(api_key=SERVER_API_KEY)
        st.session_state.api_key_set = True

init()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 10px;'>
        <div style='font-size:2rem;'>⚡</div>
        <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:800;
             background:linear-gradient(90deg,#f97316,#facc15);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            Kakarot Study
        </div>
        <div style='font-size:0.68rem;color:#475569;margin-top:2px;'>AI Study Assistant</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── PAGE NAVIGATION — using radio (always works, never breaks) ──
    st.markdown('<div class="sidebar-title">📌 Navigate</div>', unsafe_allow_html=True)
    page = st.radio(
        "page",
        ["💬  Chat", "📝  Quiz Maker", "🎙️  Voice  (soon)", "🖼️  Image Gen  (soon)", "▶️  YouTube  (soon)"],
        label_visibility="collapsed",
        key="page_nav"
    )

    st.markdown("---")

    # ── API Key ──
    if not PUBLIC_MODE:
        st.markdown('<div class="sidebar-title">🔑 Gemini API Key</div>', unsafe_allow_html=True)
        api_key_input = st.text_input("key", type="password",
            placeholder="AIzaSy...", label_visibility="collapsed")
        if api_key_input:
            if not st.session_state.api_key_set or api_key_input != getattr(st.session_state, '_last_key', ''):
                st.session_state._last_key = api_key_input
                st.session_state.pipeline = RAGPipeline(api_key=api_key_input)
                st.session_state.api_key_set = True
            st.success("✓ API key ready")
        else:
            st.warning("Enter Gemini API key")
        st.markdown("---")
    else:
        st.markdown("""
        <div style="background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);
             border-radius:10px;padding:10px 14px;font-size:0.82rem;color:#94a3b8;margin-bottom:12px;">
            ✅ <strong style="color:#4ade80;">Free to use</strong><br/>
            <span style="font-size:0.73rem;color:#475569;">Powered by Google Gemini</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

    # ── Upload ──
    st.markdown('<div class="sidebar-title">📁 Upload File</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("f", type=["pdf","jpg","jpeg","png","webp"],
        label_visibility="collapsed")

    is_image = False
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        is_image = ext in ["jpg","jpeg","png","webp"]
        if is_image:
            st.image(uploaded_file, use_column_width=True)
            uploaded_file.seek(0)
        else:
            st.markdown(f'<div style="background:#161d2e;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#94a3b8;margin:6px 0;">📄 <strong style="color:#f97316;">{uploaded_file.name}</strong></div>', unsafe_allow_html=True)

        if st.session_state.api_key_set:
            btn = "⚡ Read Image" if is_image else "⚡ Process PDF"
            spin = "🔍 Reading image..." if is_image else "📖 Indexing PDF..."
            if st.button(btn, use_container_width=True, key="process_btn"):
                with st.spinner(spin):
                    result = st.session_state.pipeline.ingest(uploaded_file)
                if result["success"]:
                    st.session_state.doc_info = result
                    st.session_state.chat_history = []
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.success(f"✓ Ready! {result['chunks']} chunks indexed")
                    st.rerun()
                else:
                    st.error(result["error"])

    # ── Doc stats ──
    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.markdown("---")
        st.markdown('<div class="sidebar-title">📊 Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pv = "🖼" if info.get("file_type")=="image" else info["pages"]
            st.markdown(f'<div class="stat-card"><div class="stat-val">{pv}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{info["chunks"]}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
        if info.get("summary"):
            st.markdown('<div class="sidebar-title" style="margin-top:10px;">📝 About</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="doc-info">{info["summary"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tip-box">Say <strong style="color:#f43f5e;">very important</strong> for exam-ready depth</div>
    <div class="tip-box">Ask for <strong style="color:#4ade80;">tricks</strong> or <strong style="color:#4ade80;">analogies</strong></div>
    <div class="tip-box">Use <strong style="color:#facc15;">Quiz Maker</strong> to test yourself!</div>
    """, unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div style="font-size:1.4rem;">⚡</div>
    <div>
        <div class="topbar-title">Kakarot Study</div>
        <div class="topbar-sub">Your AI-powered study companion</div>
    </div>
</div>""", unsafe_allow_html=True)

# ─── Route to page ────────────────────────────────────────────────────────────
active = page.strip().split()[0]   # "💬", "📝", "🎙️", "🖼️", "▶️"

# ══════════════════════════════════════════════════════════════════
# 💬 CHAT PAGE
# ══════════════════════════════════════════════════════════════════
if active == "💬":
    if not st.session_state.chat_history:
        if not st.session_state.doc_info:
            st.markdown("""
            <div class="welcome">
                <div style="font-size:2.5rem;">📚</div>
                <h2>Upload a file to start chatting</h2>
                <p>Upload your question paper PDF or a photo from the sidebar, then just chat — like texting a friend.</p>
                <div class="sample-q"><span>💬</span>Explain question 3 in simple words</div>
                <div class="sample-q"><span>🧠</span>Give me a trick to remember this concept</div>
                <div class="sample-q"><span>🎯</span>What are the most important topics? very important</div>
                <div class="sample-q"><span>📝</span>List all questions in this paper</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="welcome">
                <div style="font-size:2.5rem;">✅</div>
                <h2>Document ready! Ask me anything</h2>
                <p>Just type below — I'll explain clearly with tricks and shortcuts.</p>
                <div class="sample-q"><span>💬</span>Explain question 3 in simple words</div>
                <div class="sample-q"><span>🧠</span>Give me a trick to remember the key concepts</div>
                <div class="sample-q"><span>🎯</span>Most important topics? this is very important</div>
                <div class="sample-q"><span>📝</span>List all questions in this paper</div>
            </div>""", unsafe_allow_html=True)
    else:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                imp = turn.get("importance","normal")
                badge = '<div class="badge-high">🔴 High Priority</div><br/>' if imp=="high" else \
                        '<div class="badge-medium">🟡 Important</div><br/>' if imp=="medium" else ""
                st.markdown(f'<div class="msg-user"><div class="bubble">{badge}{turn["content"]}</div><div class="av">👤</div></div>', unsafe_allow_html=True)
            else:
                ans = turn["content"].replace("\n\n","<br/><br/>").replace("\n","<br/>")
                st.markdown(f'<div class="msg-bot"><div class="av">⚡</div><div class="bubble">{ans}</div></div>', unsafe_allow_html=True)
                if turn.get("context_used"):
                    with st.expander(f"📎 {len(turn['context_used'])} source excerpts"):
                        for i,(chunk,score) in enumerate(turn["context_used"],1):
                            c="#10b981" if score>0.3 else "#f59e0b" if score>0.15 else "#475569"
                            st.markdown(f'<div style="background:#161d2e;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;margin-bottom:6px;font-size:0.8rem;color:#94a3b8;line-height:1.6;"><span style="color:{c};font-weight:700;font-size:0.7rem;">◆ Excerpt {i} — {score:.0%} match</span><br/>{chunk[:350]}{"..." if len(chunk)>350 else ""}</div>', unsafe_allow_html=True)

    # Input
    if not st.session_state.api_key_set:
        st.info("👈 Enter your Gemini API key in the sidebar")
    elif not st.session_state.doc_info:
        st.info("👈 Upload a PDF or photo in the sidebar first")
    else:
        user_input = st.chat_input("Ask anything... e.g. 'explain question 2' or 'give me a trick for this'")
        if user_input and user_input.strip():
            with st.spinner("⚡ Thinking..."):
                result = st.session_state.pipeline.query(
                    user_query=user_input.strip(),
                    chat_history=st.session_state.chat_history, top_k=5)
            st.session_state.chat_history.append({"role":"user","content":user_input.strip(),"importance":result["importance"]})
            st.session_state.chat_history.append({"role":"assistant","content":result["answer"],"context_used":result["context_used"]})
            st.rerun()

# ══════════════════════════════════════════════════════════════════
# 📝 QUIZ PAGE
# ══════════════════════════════════════════════════════════════════
elif active == "📝":
    from quiz_engine import generate_quiz

    st.markdown('<div class="quiz-header">📝 Quiz Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="quiz-sub">Auto-generates MCQs from your document. Test yourself instantly!</div>', unsafe_allow_html=True)

    if not st.session_state.api_key_set:
        st.warning("👈 Enter your Gemini API key in the sidebar first")
    elif not st.session_state.doc_info:
        st.info("👈 Upload a PDF or photo from the sidebar first, then come back here!")
    elif not st.session_state.quiz_questions:
        # Settings
        with st.container():
            st.markdown('<div style="background:var(--card);border:1px solid #1e293b;border-radius:14px;padding:22px;margin-bottom:18px;">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                num_q = st.number_input("Number of Questions", min_value=3, max_value=20, value=5)
            with c2:
                difficulty = st.selectbox("Difficulty", ["easy","medium","hard"], index=1)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("⚡ Generate Quiz Now!", use_container_width=True, key="gen_quiz"):
                with st.spinner("🧠 Gemini is crafting your quiz..."):
                    try:
                        chunks = st.session_state.pipeline.retriever.chunks
                        questions = generate_quiz(chunks, st.session_state.pipeline.llm, num_q, difficulty)
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Quiz generation failed: {str(e)}")

    elif not st.session_state.quiz_submitted:
        total = len(st.session_state.quiz_questions)
        answered = len(st.session_state.quiz_answers)
        st.markdown(f'<div style="background:var(--card);border:1px solid #1e293b;border-radius:10px;padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:var(--muted);">Progress: <strong style="color:#f97316">{answered}/{total}</strong> answered</div>', unsafe_allow_html=True)

        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f'<div class="q-card"><div class="q-num">Question {i+1} of {total}</div><div class="q-text">{q["question"]}</div></div>', unsafe_allow_html=True)
            opts = [f"{k}) {v}" for k, v in q["options"].items()]
            # Use selectbox — always reliable, no click issues
            prev = st.session_state.quiz_answers.get(i)
            prev_idx = next((j for j, o in enumerate(opts) if o.startswith(str(prev))), 0) if prev else 0
            choice = st.selectbox(f"Your answer for Q{i+1}", ["-- Select answer --"] + opts,
                index=0 if not prev else prev_idx+1, key=f"quiz_q_{i}", label_visibility="visible")
            if choice != "-- Select answer --":
                st.session_state.quiz_answers[i] = choice[0]
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Submit Quiz", use_container_width=True, key="submit_quiz"):
                st.session_state.quiz_submitted = True
                st.rerun()
        with col2:
            if st.button("🔄 New Quiz", use_container_width=True, key="new_quiz_1"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

    else:
        # Results
        questions = st.session_state.quiz_questions
        answers = st.session_state.quiz_answers
        total = len(questions)
        correct = sum(1 for i,q in enumerate(questions) if answers.get(i)==q["answer"])
        pct = int((correct/total)*100)
        grade = "🏆 Excellent!" if pct>=80 else "👍 Good job!" if pct>=60 else "📚 Keep practicing!"
        gc = "#4ade80" if pct>=80 else "#f59e0b" if pct>=60 else "#f43f5e"

        st.markdown(f'<div class="score-card"><div class="score-big">{correct}/{total}</div><div style="font-size:0.85rem;color:#94a3b8;">{pct}% Score</div><div style="font-size:1.1rem;margin-top:10px;color:{gc};font-weight:600;">{grade}</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-bottom:12px;">📋 Answer Review</div>', unsafe_allow_html=True)

        for i, q in enumerate(questions):
            ua = answers.get(i)
            ca = q["answer"]
            ok = ua == ca
            bc = "rgba(74,222,128,0.3)" if ok else "rgba(244,63,94,0.3)"
            bg = "rgba(74,222,128,0.05)" if ok else "rgba(244,63,94,0.05)"
            icon = "✅" if ok else "❌"
            wrong_line = "" if ok else f'<div style="font-size:0.84rem;color:#94a3b8;margin-bottom:6px;">Correct: <strong style="color:#4ade80">{ca}) {q["options"][ca]}</strong></div>'
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bc};border-radius:12px;padding:18px 20px;margin-bottom:12px;">
                <div style="font-size:0.7rem;font-weight:700;color:{'#4ade80' if ok else '#f43f5e'};text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px;">{icon} Q{i+1} — {'Correct' if ok else 'Incorrect'}</div>
                <div style="font-size:0.93rem;color:#f1f5f9;font-weight:500;margin-bottom:10px;">{q["question"]}</div>
                <div style="font-size:0.84rem;color:#94a3b8;margin-bottom:4px;">Your answer: <strong style="color:{'#4ade80' if ok else '#f43f5e'}">{ua}) {q['options'].get(ua,'Not answered') if ua else 'Not answered'}</strong></div>
                {wrong_line}
                <div class="explanation">💡 {q.get('explanation','')}</div>
            </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Again", use_container_width=True, key="retry_quiz"):
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
        with col2:
            if st.button("📝 New Quiz", use_container_width=True, key="new_quiz_2"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

# ══════════════════════════════════════════════════════════════════
# COMING SOON PAGES
# ══════════════════════════════════════════════════════════════════
else:
    info = {"🎙️":("🎙️","Voice Chat","Talk to your AI tutor instead of typing"),
            "🖼️":("🖼️","Image Generator","Describe a concept and get a visual"),
            "▶️":("▶️","YouTube Summarizer","Paste a video link and get instant study notes")}
    ic, title, desc = info.get(active, ("⚡","Coming Soon",""))
    st.markdown(f"""
    <div class="coming-soon">
        <div style="font-size:3rem;">{ic}</div>
        <div style="background:rgba(168,85,247,0.12);border:1px solid rgba(168,85,247,0.25);color:#a855f7;font-size:0.75rem;font-weight:700;padding:4px 14px;border-radius:100px;display:inline-block;letter-spacing:.5px;">COMING SOON</div>
        <h2>{title}</h2>
        <p>{desc}</p>
        <p style="color:#334155;font-size:0.82rem;">Being built right now. Check back soon! ⚡</p>
    </div>""", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;padding:16px;font-size:0.7rem;color:#1e293b;">⚡ Kakarot Study · Google Gemini · Made for smart learners</div>', unsafe_allow_html=True)
