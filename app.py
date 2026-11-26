"""
LearnMate AI — Your Personal AI Teaching Assistant.
Streamlit Application for LearnMate AI.
Features:
- Ask AI: Lecture-grounded doubt resolution with timestamp verification.
- Clickable lecture citations with instantaneous video seeking.
- Multi-language support: English, Hindi (हिन्दी), and Hinglish.
- Scope-aware retrieval: Current lecture or all curriculum lectures.
- Safe offline demonstration and Google Gemini Live Free Tier modes.
- Student feedback collection for continuous learning.
- Smart Notes and Practice Quiz modules (scaffolded for Phase 8).
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st

import config
from rag_pipeline import RAGPipeline
from llm import LLMClient
from feedback import FeedbackManager
from summarizer import LectureSummarizer
from quiz_generator import QuizGenerator

# -----------------------------------------------------------------------------
# 1. Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LearnMate AI — Your Personal AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom Modern EdTech Styling (Clean Blue, Slate, Neutral Palette)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.15rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Brand Header */
    .brand-header-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 1rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.25;
    }
    .brand-tagline {
        font-size: 0.95rem;
        opacity: 0.92;
        margin: 0.25rem 0 0 0;
        letter-spacing: -0.01em;
    }
    
    /* Lecture Information Card */
    .lecture-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.65rem 1.15rem;
        margin-bottom: 0.65rem;
    }
    .lecture-card h4 {
        margin: 0 0 0.4rem 0;
        color: #1E293B;
        font-size: 1.1rem;
    }
    .lecture-meta {
        display: flex;
        gap: 1rem;
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Video Player Responsive Container & Sizing (Max 900px, 16:9 ratio, centered) */
    .stVideo {
        max-width: 900px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 0.4rem !important;
    }
    .stVideo video {
        border-radius: 8px;
        aspect-ratio: 16 / 9;
        width: 100% !important;
        height: auto !important;
    }
    
    /* Citation Cards */
    .citation-box {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }
    .citation-title {
        font-weight: 600;
        color: #1E3A8A;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .citation-timestamp {
        background: #EFF6FF;
        color: #1D4ED8;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 700;
        font-family: monospace;
        font-size: 0.82rem;
    }
    
    /* Answerability Status Banners */
    .status-banner {
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        font-size: 0.88rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-answerable {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
    }
    .status-partial {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }
    .status-unanswerable {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .status-api-error {
        background-color: #FFF7ED;
        border: 1px solid #FFEDD5;
        color: #C2410C;
    }

    /* Mode Pill Indicator */
    .mode-pill-live {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .mode-pill-offline {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Suggested Question Buttons */
    .suggested-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Data & Resource Loaders (Cached for 8GB RAM Protection)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lecture_manifest() -> List[Dict[str, Any]]:
    """Load lecture metadata from manifest, sorted by tutorial number."""
    manifest_path = config.DATA_DIR / "lecture_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                lectures = data.get("lectures", [])
                # Order naturally by tutorial number
                return sorted(lectures, key=lambda x: x.get("tutorial_number", 0))
        except Exception as e:
            st.error(f"Failed to load lecture manifest: {e}")

    # Fallback manifest definition if file missing
    return [
        {
            "lecture_id": "DEMO_01",
            "lecture_title": "Your First HTML Website",
            "tutorial_number": 1,
            "duration_formatted": "00:04:12",
            "duration_seconds": 252.0,
            "video_filename": "lecture_01_html_fundamentals.mp4"
        },
        {
            "lecture_id": "DEMO_02",
            "lecture_title": "Video, Audio & Media in HTML",
            "tutorial_number": 2,
            "duration_formatted": "00:04:12",
            "duration_seconds": 252.0,
            "video_filename": "lecture_02_html_multimedia.mp4"
        },
        {
            "lecture_id": "DEMO_03",
            "lecture_title": "SEO & Core Web Vitals",
            "tutorial_number": 3,
            "duration_formatted": "00:04:12",
            "duration_seconds": 252.0,
            "video_filename": "lecture_03_seo_core_web_vitals.mp4"
        }
    ]


@st.cache_resource(show_spinner="Initializing semantic vector store and embedding engine...")
def get_rag_pipeline(mock_mode: bool) -> RAGPipeline:
    """
    Cache shared RAGPipeline instance globally.
    Prevents reloading sentence-transformer embeddings or rebuilding FAISS index on reruns.
    """
    llm = LLMClient(mock_mode=mock_mode)
    return RAGPipeline(llm_client=llm)


# -----------------------------------------------------------------------------
# 4. Session State Management
# -----------------------------------------------------------------------------
def init_session_state(lectures: List[Dict[str, Any]]):
    """Initialize Streamlit session state variables with clean defaults."""
    if "selected_lecture_id" not in st.session_state:
        st.session_state["selected_lecture_id"] = lectures[0]["lecture_id"] if lectures else "LEC_03"

    if "seek_time" not in st.session_state:
        st.session_state["seek_time"] = 0

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "feedback_recorded" not in st.session_state:
        st.session_state["feedback_recorded"] = set()

    if "suggested_prompt" not in st.session_state:
        st.session_state["suggested_prompt"] = None

    if "last_query_hash" not in st.session_state:
        st.session_state["last_query_hash"] = None

    if "smart_notes" not in st.session_state:
        st.session_state["smart_notes"] = {}

    if "active_quizzes" not in st.session_state:
        st.session_state["active_quizzes"] = {}

    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}

    if "quiz_results" not in st.session_state:
        st.session_state["quiz_results"] = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = {}

    if "backend_exec_mode" not in st.session_state:
        st.session_state["backend_exec_mode"] = "Offline Demonstration (Zero Quota)"


# -----------------------------------------------------------------------------
# 5. Suggested Questions Registry (Verified Against Transcripts)
# -----------------------------------------------------------------------------
SUGGESTED_QUESTIONS = {
    "DEMO_01": [
        "What is HTML and what does <!DOCTYPE html> mean?",
        "What are the main sections inside the <html> tag?",
        "How do I create a paragraph and heading in HTML?"
    ],
    "DEMO_02": [
        "How do I embed a video using the <video> tag in HTML?",
        "Why is muted required for video autoplay in modern browsers?",
        "What is the purpose of the <iframe> tag for embedding media?"
    ],
    "DEMO_03": [
        "What is SEO and why is the description meta tag important?",
        "What are Google Core Web Vitals and what does LCP measure?",
        "How does Google Lighthouse help audit website performance?"
    ],
    "LEC_03": [
        "How do I create my first HTML website?",
        "What is an HTML tag and element in HTML?",
        "How do I install and use VS Code for HTML?"
    ],
    "LEC_01": [
        "What is SEO and why is it important?",
        "सर्च इंजन ऑप्टिमाइजेशन और मेटा टैग क्या होते हैं?",
        "What are Core Web Vitals and what does LCP measure?"
    ],
    "LEC_02": [
        "How do I play a video using the video tag in HTML?",
        "HTML me video play karne ke liye kaun sa tag use karte hai aur controls kaise lagate hai?",
        "What is the purpose of the iframe tag and how to embed YouTube?"
    ]
}


# -----------------------------------------------------------------------------
# 6. Main Application
# -----------------------------------------------------------------------------
def main():
    lectures = load_lecture_manifest()
    init_session_state(lectures)

    # Fast lookup for lectures
    lec_map = {lec["lecture_id"]: lec for lec in lectures}

    # -------------------------------------------------------------------------
    # A. Top Brand Header
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="brand-header-container">
        <h1 class="brand-title">🎓 LearnMate AI</h1>
        <p class="brand-tagline">Your Personal AI Teaching Assistant — Lecture-Grounded Doubt Resolution & Revision</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # B. Sidebar Configuration
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📚 Course Navigation")
        st.caption("**Web Development Fundamentals** (3-Lecture Course & Interactive Video Demonstration)")

        # Lecture Selection Dropdown
        lec_options = [lec["lecture_id"] for lec in lectures]
        current_idx = lec_options.index(st.session_state["selected_lecture_id"]) if st.session_state["selected_lecture_id"] in lec_options else 0

        selected_lec_id = st.selectbox(
            "Select Lecture:",
            options=lec_options,
            index=current_idx,
            format_func=lambda lid: f"Tutorial #{lec_map[lid].get('tutorial_number')}: {lec_map[lid].get('lecture_title', '').split('|')[0].strip()}"
        )

        # Update session state if user changed dropdown
        if selected_lec_id != st.session_state["selected_lecture_id"]:
            st.session_state["selected_lecture_id"] = selected_lec_id
            st.session_state["seek_time"] = 0
            st.rerun()

        st.markdown("---")
        st.markdown("### 🎯 Retrieval & Language Settings")

        # Search Scope
        scope_choice = st.radio(
            "Search Scope:",
            options=["Current Lecture Only", "All Available Lectures"],
            index=0,
            help="Limit retrieval context to the selected video lecture or search across all ingested lectures."
        )
        lecture_filter = selected_lec_id if scope_choice == "Current Lecture Only" else None

        # Language Preference
        lang_choice = st.selectbox(
            "Response Language:",
            options=["Auto Detect", "English", "Hindi (हिन्दी)", "Hinglish"],
            index=0,
            help="LearnMate AI responds in your preferred language while strictly adhering to lecture transcripts."
        )

        st.markdown("---")
        st.markdown("### ⚡ Execution & Model Mode")

        # Live Gemini vs Offline Demonstration Toggle
        # In public cloud deployment, default to Offline Demonstration and disable live mode
        # unless explicitly enabled by administrator via ALLOW_LIVE_MODE=true with valid key.
        allow_live = os.getenv("ALLOW_LIVE_MODE", "false").lower() in ("true", "1", "yes") and bool(config.GEMINI_API_KEY)

        if allow_live:
            exec_mode = st.radio(
                "Backend Mode:",
                options=["Offline Demonstration (Zero Quota)", "Live AI (Google Gemini Free Tier)"],
                key="backend_exec_mode",
                help="Offline Demonstration uses deterministic mock synthesis. Live AI calls Google Gemini."
            )
            is_live_mode = (exec_mode == "Live AI (Google Gemini Free Tier)")

            if is_live_mode:
                st.markdown("""
                <div style="font-size:0.8rem; background:#ECFDF5; border:1px solid #A7F3D0; padding:0.5rem; border-radius:6px; color:#065F46;">
                    <strong>⚡ Live AI Active:</strong> Google Gemini (Primary: gemini-3.5-flash-lite &middot; Fallback: gemini-3.8-flash &middot; ₹0.00 Cost).<br>
                    <em>Rate-limit policy: 15 RPM with single bounded fallback.</em>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="font-size:0.8rem; background:#F1F5F9; border:1px solid #CBD5E1; padding:0.5rem; border-radius:6px; color:#475569;">
                    <strong>🧪 Offline Demo Mode:</strong> Zero API quota consumption. Instant local responses.
                </div>
                """, unsafe_allow_html=True)
        else:
            is_live_mode = False
            st.session_state["backend_exec_mode"] = "Offline Demonstration (Zero Quota)"
            st.markdown("""
            <div style="font-size:0.82rem; background:#F1F5F9; border:1px solid #CBD5E1; padding:0.6rem 0.75rem; border-radius:6px; color:#334155;">
                <div style="font-weight:600; color:#1E293B; margin-bottom:2px;">🧪 Public Demonstration Mode</div>
                <div>All features operate in deterministic <strong>Offline Demonstration mode</strong> with verified lecture content. Live generation is disabled in public deployment to protect API quotas.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Clear Conversation Button
        if st.button("🗑️ Clear Chat History", key="btn_clear_chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["feedback_recorded"] = set()
            st.session_state["suggested_prompt"] = None
            st.rerun()

        # Instructions & About
        with st.expander("ℹ️ About LearnMate AI", expanded=False):
            st.markdown("""
            **LearnMate AI** converts educational videos into interactive learning companions:
            - **Ask AI:** Grounded doubts with timestamps.
            - **Zero Hallucination:** Answers strictly from lecture transcripts.
            - **Clickable Sources:** Jump straight to video segments.
            - **Languages:** Natural Hindi, English, and Hinglish.
            """)

    # -------------------------------------------------------------------------
    # C. Lecture Learning Area (Video Player & Metadata)
    # -------------------------------------------------------------------------
    current_lec = lec_map.get(st.session_state["selected_lecture_id"], lectures[0])
    video_rel_path = current_lec.get("video_filename", "")
    full_video_path = config.VIDEOS_DIR / video_rel_path

    st.markdown(f"""
    <div class="lecture-card">
        <h4>🎬 Tutorial #{current_lec.get('tutorial_number')}: {current_lec.get('lecture_title', '').split('|')[0].strip()}</h4>
        <div class="lecture-meta">
            <span>⏱️ Duration: <strong>{current_lec.get('duration_formatted', 'N/A')}</strong></span>
            <span>📍 Active Seek: <strong>{int(st.session_state['seek_time'] // 60):02d}:{int(st.session_state['seek_time'] % 60):02d}</strong></span>
            <span>📂 Scope: <strong>{scope_choice}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Render Local WebM / MP4 Video Player with Native Seek Time
    # Centered, compact container (approx. 68% desktop width, max 900px, 16:9 ratio)
    v_col_left, v_col_center, v_col_right = st.columns([1.6, 6.8, 1.6])
    with v_col_center:
        sample_video_path = config.VIDEOS_DIR / "demo_sample_lecture.mp4"
        if full_video_path.exists():
            seek_sec = int(st.session_state.get("seek_time", 0))
            st.video(
                str(full_video_path),
                format="video/webm" if str(full_video_path).endswith(".webm") else "video/mp4",
                start_time=seek_sec
            )
        elif sample_video_path.exists():
            seek_sec = int(st.session_state.get("seek_time", 0))
            st.video(
                str(sample_video_path),
                format="video/mp4",
                start_time=seek_sec
            )
            st.caption("🎬 *Interactive Demonstration Video: Clickable timestamp seek buttons (`🎯 Seek to [MM:SS]`) dynamically cue this player.*")
        else:
            seek_sec = int(st.session_state.get("seek_time", 0))
            seek_m = int(seek_sec // 60)
            seek_s = int(seek_sec % 60)
            st.markdown(f"""
            <div style="background:#1E293B; border:1px solid #334155; border-radius:8px; padding:1.25rem; text-align:center; color:#F8FAFC; margin-bottom:0.75rem;">
                <div style="font-size:1.8rem; margin-bottom:0.25rem;">🎬</div>
                <strong style="font-size:1rem; color:#F8FAFC;">{current_lec.get('lecture_title', '').split('|')[0].strip()}</strong>
                <p style="font-size:0.8rem; color:#94A3B8; margin:0.35rem 0 0.75rem 0;">
                    Full lecture video excluded from repository for lightweight cloud deployment.
                </p>
                <div style="display:inline-block; background:#0F172A; border:1px solid #38BDF8; padding:0.35rem 0.85rem; border-radius:16px; font-size:0.8rem; color:#38BDF8;">
                    ⏱️ Active Timestamp Position: <strong>{seek_m:02d}:{seek_s:02d}</strong> ({seek_sec}s)
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # D. Feature Tabs: Ask AI, Smart Notes, Practice Quiz
    # -------------------------------------------------------------------------
    tab_ask, tab_notes, tab_quiz = st.tabs([
        "💬 Ask AI (Doubt Resolution)",
        "📝 Smart Notes",
        "🎯 Practice Quiz"
    ])

    # =========================================================================
    # TAB 1: ASK AI (Doubt Resolution) — FULL INTEGRATION
    # =========================================================================
    with tab_ask:
        st.markdown("#### 💡 Lecture-Grounded Doubt Assistant")
        st.caption("Ask questions in **English**, **हिन्दी**, or **Hinglish**. LearnMate AI answers strictly from lecture content with verified timestamps.")

        # Suggested Questions Pills
        current_suggestions = SUGGESTED_QUESTIONS.get(current_lec["lecture_id"], [])
        if current_suggestions:
            st.markdown('<div class="suggested-title">💡 Suggested Questions for this Lecture:</div>', unsafe_allow_html=True)
            cols = st.columns(len(current_suggestions))
            for i, sq in enumerate(current_suggestions):
                with cols[i]:
                    if st.button(f"👉 {sq}", key=f"sugg_{i}", use_container_width=True):
                        st.session_state["suggested_prompt"] = sq
                        st.rerun()

        st.markdown("---")

        # Render Conversation History
        for msg_idx, msg in enumerate(st.session_state["messages"]):
            role = msg["role"]
            with st.chat_message(role, avatar="🧑‍🎓" if role == "user" else "🎓"):
                if role == "user":
                    st.write(msg["content"])
                else:
                    # Assistant Response Rendering
                    status = msg.get("status", "ANSWERABLE")
                    answer_text = msg.get("content", "")
                    citations = msg.get("citations", [])
                    missing_info = msg.get("missing_information")
                    interaction_id = msg.get("interaction_id", f"msg_{msg_idx}")
                    is_mock_msg = msg.get("is_mock", False)
                    is_cached_msg = msg.get("is_cached", False)
                    gen_model_msg = msg.get("generating_model", "gemini-3.5-flash-lite")

                    if is_mock_msg:
                        badge_html = '<span style="background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #CBD5E1;">🧪 Offline Demonstration</span>'
                    elif is_cached_msg:
                        badge_html = f'<span style="background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #BFDBFE;">⚡ Cached ({gen_model_msg})</span>'
                    else:
                        badge_html = f'<span style="background:#ECFDF5; color:#047857; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #A7F3D0;">⚡ Live AI ({gen_model_msg})</span>'

                    # Render Answerability Banner
                    if status == "ANSWERABLE":
                        st.markdown(f'<div class="status-banner status-answerable">✅ <strong>Grounded Answer:</strong> Supported by lecture transcript evidence. &nbsp; {badge_html}</div>', unsafe_allow_html=True)
                    elif status == "PARTIALLY_ANSWERABLE":
                        st.markdown(f'<div class="status-banner status-partial">⚠️ <strong>Partially Covered in Lecture:</strong> Portions not found in the lecture are stated below. &nbsp; {badge_html}</div>', unsafe_allow_html=True)
                    elif status == "NOT_ANSWERABLE":
                        st.markdown(f'<div class="status-banner status-unanswerable">🛑 <strong>Topic Not Covered in Curriculum:</strong> This concept is not found in our available lectures. &nbsp; {badge_html}</div>', unsafe_allow_html=True)
                    elif status == "API_ERROR":
                        st.markdown(f'<div class="status-banner status-api-error">⚠️ <strong>Service Availability Notice:</strong> The AI service is temporarily experiencing high demand. Please try again later or switch to Offline Demonstration mode. &nbsp; {badge_html}</div>', unsafe_allow_html=True)
                        if st.button("👉 Switch to Offline Demonstration Mode", key=f"switch_offline_ask_{msg_idx}"):
                            st.session_state["backend_exec_mode"] = "Offline Demonstration (Zero Quota)"
                            st.rerun()

                    # Main Pedagogical Answer Body (Preserve Markdown and Code blocks)
                    if status != "API_ERROR":
                        st.markdown(answer_text)
                    else:
                        st.info("💡 You can switch to **Offline Demonstration mode** above for instant, deterministic doubt resolution with zero quota usage.")

                    # Missing Information Disclosure (for Partially Answerable doubts)
                    if status == "PARTIALLY_ANSWERABLE" and missing_info:
                        st.info(f"**Missing Lecture Evidence:** {missing_info}")

                    # Render Verified Clickable Citations ("Jump to lecture")
                    if citations:
                        st.markdown("**📌 Verified Lecture Sources:**")
                        for c_idx, cit in enumerate(citations):
                            c_id = cit.get("chunk_id", "CHUNK")
                            c_tut = cit.get("tutorial_number", current_lec.get("tutorial_number"))
                            c_title = cit.get("lecture_title", "").split("|")[0].strip()
                            s_ts = cit.get("start_timestamp", "00:00:00")
                            e_ts = cit.get("end_timestamp", "00:00:00")
                            start_seconds = cit.get("start_time", 0.0)

                            c_col1, c_col2 = st.columns([3, 1])
                            with c_col1:
                                st.markdown(f"""
                                <div class="citation-box">
                                    <div class="citation-title">
                                        <span>📖 Tutorial #{c_tut}: {c_title}</span>
                                        <span class="citation-timestamp">[{s_ts} – {e_ts}]</span>
                                    </div>
                                    <div style="color:#64748B; font-size:0.8rem; margin-top:0.25rem;">Source Chunk ID: <code>{c_id}</code></div>
                                </div>
                                """, unsafe_allow_html=True)
                            with c_col2:
                                st.write("")
                                if st.button(f"🎯 Seek to [{s_ts}]", key=f"jump_{msg_idx}_{c_idx}", use_container_width=True):
                                    st.session_state["selected_lecture_id"] = cit.get("lecture_id", current_lec["lecture_id"])
                                    st.session_state["seek_time"] = int(start_seconds)
                                    st.rerun()

                    # Student Feedback Widget
                    if status in ("ANSWERABLE", "PARTIALLY_ANSWERABLE", "NOT_ANSWERABLE"):
                        fb_key = f"fb_{interaction_id}"
                        if interaction_id not in st.session_state["feedback_recorded"]:
                            f_col1, f_col2, f_col3 = st.columns([1, 1, 4])
                            with f_col1:
                                if st.button("👍 Helpful", key=f"pos_{interaction_id}"):
                                    FeedbackManager().record_feedback(
                                        category="doubt",
                                        item_id=interaction_id,
                                        rating=5,
                                        comments="Helpful",
                                        metadata={"lecture_id": current_lec["lecture_id"], "status": status, "mode": exec_mode}
                                    )
                                    st.session_state["feedback_recorded"].add(interaction_id)
                                    st.toast("Thank you for your feedback!", icon="🎉")
                                    st.rerun()
                            with f_col2:
                                if st.button("👎 Confusing", key=f"neg_{interaction_id}"):
                                    FeedbackManager().record_feedback(
                                        category="doubt",
                                        item_id=interaction_id,
                                        rating=1,
                                        comments="Needs improvement",
                                        metadata={"lecture_id": current_lec["lecture_id"], "status": status, "mode": exec_mode}
                                    )
                                    st.session_state["feedback_recorded"].add(interaction_id)
                                    st.toast("Feedback recorded. We'll improve this explanation!", icon="📝")
                                    st.rerun()
                        else:
                            st.caption("✅ Feedback submitted. Thank you!")

        # ---------------------------------------------------------------------
        # Chat Input & Doubt Processing
        # ---------------------------------------------------------------------
        chat_prompt = st.chat_input("Ask a doubt about HTML, tags, SEO, Core Web Vitals, or media...")

        # Handle button click from suggested questions
        if st.session_state["suggested_prompt"]:
            active_prompt = st.session_state["suggested_prompt"]
            st.session_state["suggested_prompt"] = None
        else:
            active_prompt = chat_prompt

        if active_prompt:
            clean_q = active_prompt.strip()

            # Prevent empty inputs or excessively long payloads
            if not clean_q:
                st.warning("Please enter a non-empty question.")
                return

            if len(clean_q) > 600:
                st.warning("Your doubt exceeds the 600-character limit. Please summarize your query.")
                return

            # Append Student Query to Conversation History
            st.session_state["messages"].append({
                "role": "user",
                "content": clean_q,
                "timestamp": time.time()
            })

            # Prepare Conversation History for Contextual Coreference Resolution
            history_payload = []
            for m in st.session_state["messages"][:-1]:
                history_payload.append({
                    "role": m["role"],
                    "content": m["content"]
                })

            # Execute RAG Pipeline with Loading Indicator
            with st.spinner("Searching lecture transcripts & verifying evidence..."):
                pipeline = get_rag_pipeline(mock_mode=(not is_live_mode))

                # Inject language preference if explicitly selected
                processed_question = clean_q
                if lang_choice == "Hindi (हिन्दी)":
                    processed_question += " (Please answer in clear Hindi using Devanagari script)."
                elif lang_choice == "Hinglish":
                    processed_question += " (Please answer in natural conversational Hinglish)."
                elif lang_choice == "English":
                    processed_question += " (Please answer in English)."

                t0 = time.time()
                res = pipeline.answer_doubt(
                    question=processed_question,
                    lecture_id=lecture_filter,
                    conversation_history=history_payload
                )
                elapsed = time.time() - t0

            # Generate Unique Interaction ID
            interaction_id = f"doubt_{int(time.time())}_{hashlib.md5(clean_q.encode('utf-8')).hexdigest()[:6]}"

            # Detect API Error Condition
            status = res.get("answerability_status", "NOT_ANSWERABLE")
            if res.get("error"):
                status = "API_ERROR"

            # Store Assistant Response in Session State
            st.session_state["messages"].append({
                "role": "assistant",
                "content": res.get("answer", "No response generated."),
                "status": status,
                "citations": res.get("citations", []),
                "missing_information": res.get("missing_information"),
                "interaction_id": interaction_id,
                "timestamp": time.time(),
                "latency_sec": round(elapsed, 2),
                "generating_model": res.get("metrics", {}).get("model_name", "gemini-3.5-flash-lite"),
                "is_cached": res.get("is_cached", False),
                "is_mock": res.get("metrics", {}).get("is_mock", (not is_live_mode))
            })

            st.rerun()

    # =========================================================================
    # TAB 2: SMART NOTES (Full Integration)
    # =========================================================================
    with tab_notes:
        st.markdown(f"#### 📝 Smart Revision Notes: Tutorial #{current_lec.get('tutorial_number')}")
        st.caption(f"**Lecture:** {current_lec.get('lecture_title')} | High-yield AI-synthesized revision notes with verified timestamps.")

        # Notes Generation Controls
        n_col1, n_col2, n_col3 = st.columns([2, 2, 2])
        with n_col1:
            style_labels = {
                "Detailed Study": "detailed_study",
                "Quick Revision": "quick_revision",
                "Key Concepts": "key_concepts"
            }
            selected_style_label = st.selectbox(
                "Notes Style:",
                options=list(style_labels.keys()),
                index=0,
                key=f"notes_style_{current_lec['lecture_id']}"
            )
            selected_style = style_labels[selected_style_label]
        with n_col2:
            lang_opts = ["English", "Hindi", "Hinglish"]
            default_lang_idx = 0
            if "Hindi" in lang_choice:
                default_lang_idx = 1
            elif "Hinglish" in lang_choice:
                default_lang_idx = 2
            selected_notes_lang = st.selectbox(
                "Notes Language:",
                options=lang_opts,
                index=default_lang_idx,
                key=f"notes_lang_{current_lec['lecture_id']}"
            )
        with n_col3:
            st.write("")
            st.write("")
            gen_notes_btn = st.button("📄 Generate Smart Notes", key=f"btn_gen_notes_{current_lec['lecture_id']}", use_container_width=True)

        notes_cache_key = f"{current_lec['lecture_id']}_{selected_style}_{selected_notes_lang}"

        # Automatic generation if not present, or manual trigger
        if gen_notes_btn or (notes_cache_key not in st.session_state["smart_notes"]):
            with st.spinner("Synthesizing lecture transcript into Smart Notes..."):
                summarizer = LectureSummarizer(llm_client=LLMClient(mock_mode=(not is_live_mode)))
                try:
                    notes_result = summarizer.generate_smart_notes(
                        lecture_id=current_lec["lecture_id"],
                        notes_style=selected_style,
                        language=selected_notes_lang,
                        force_refresh=bool(gen_notes_btn)
                    )
                    if notes_result.get("error"):
                        st.session_state["notes_error"] = notes_result.get("error")
                    else:
                        st.session_state["smart_notes"][notes_cache_key] = notes_result
                        st.session_state.pop("notes_error", None)
                except Exception as e:
                    st.session_state["notes_error"] = str(e)

        if st.session_state.get("notes_error"):
            st.warning("⚠️ The AI service is temporarily experiencing high demand. Please try again later or switch to Offline Demonstration mode.")
            if st.button("👉 Switch to Offline Demonstration Mode", key=f"switch_offline_notes_{current_lec['lecture_id']}"):
                st.session_state["backend_exec_mode"] = "Offline Demonstration (Zero Quota)"
                st.session_state.pop("notes_error", None)
                st.rerun()

        # Render Active Notes
        if notes_cache_key in st.session_state["smart_notes"]:
            active_notes = st.session_state["smart_notes"][notes_cache_key]

            # Action Bar: Download Button + Metadata Pill + Source Badge
            d_col1, d_col2 = st.columns([4, 2])
            with d_col1:
                is_mock_n = active_notes.get("is_mock", (not is_live_mode))
                is_cached_n = active_notes.get("is_cached", False)
                gen_model_n = active_notes.get("generating_model", "gemini-3.5-flash-lite")
                if is_mock_n:
                    n_badge = '<span style="background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #CBD5E1;">🧪 Offline Demonstration</span>'
                elif is_cached_n:
                    n_badge = f'<span style="background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #BFDBFE;">⚡ Cached ({gen_model_n})</span>'
                else:
                    n_badge = f'<span style="background:#ECFDF5; color:#047857; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #A7F3D0;">⚡ Live AI ({gen_model_n})</span>'

                st.markdown(f"**Style:** `{selected_style_label}` | **Language:** `{selected_notes_lang}` | {n_badge}", unsafe_allow_html=True)
            with d_col2:
                from pdf_exporter import generate_notes_pdf
                pdf_payload = dict(active_notes)
                pdf_payload["course_title"] = "Web Development Fundamentals"
                pdf_payload["lecture_title"] = current_lec.get("lecture_title", f"Tutorial #{current_lec.get('tutorial_number')}").split("|")[0].strip()
                pdf_payload["tutorial_number"] = current_lec.get("tutorial_number", 1)

                # Generate publication-quality PDF directly in memory
                pdf_data = generate_notes_pdf(pdf_payload)

                tut_num = current_lec.get("tutorial_number", 1)
                clean_style = selected_style.replace("-", "_").title()
                clean_lang = selected_notes_lang.replace(" ", "_")
                pdf_filename = f"LearnMate_AI_Lecture_{tut_num:02d}_{clean_style}_{clean_lang}.pdf"

                st.download_button(
                    label="📄 Download Notes (PDF)",
                    data=pdf_data,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_pdf_notes_{notes_cache_key}"
                )

            st.markdown("---")

            # 1. Executive Summary
            st.markdown("##### 📌 Executive Summary")
            st.info(active_notes.get("executive_summary", ""))

            # 2. Learning Objectives
            st.markdown("##### 🎯 Learning Objectives")
            for obj in active_notes.get("learning_objectives", []):
                st.markdown(f"- ✅ {obj}")

            st.markdown("---")

            # 3. Core Topics & Lecture Breakdown
            st.markdown("##### 📖 Core Topics & Lecture Breakdown")
            for s_idx, sec in enumerate(active_notes.get("sections", [])):
                with st.expander(f"📌 {sec.get('heading', f'Topic {s_idx+1}')}", expanded=True):
                    st.markdown(sec.get("content", ""))

                    snippets = sec.get("code_snippets", [])
                    if snippets:
                        st.markdown("**💻 Code Example:**")
                        for snip in snippets:
                            st.code(snip, language="html")

                    citations = sec.get("citations", [])
                    if citations:
                        st.markdown("**📍 Lecture Source Timestamps:**")
                        c_cols = st.columns(min(len(citations), 4))
                        for c_i, cite in enumerate(citations[:4]):
                            ts_str = cite.get("timestamp", "00:00")
                            with c_cols[c_i]:
                                if st.button(f"🎯 Seek to [{ts_str}]", key=f"notes_seek_{notes_cache_key}_{s_idx}_{c_i}", use_container_width=True):
                                    parts = ts_str.split(":")
                                    sec_val = 0
                                    if len(parts) == 2:
                                        sec_val = int(parts[0]) * 60 + int(parts[1])
                                    elif len(parts) == 3:
                                        sec_val = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                                    st.session_state["seek_time"] = sec_val
                                    st.rerun()

            st.markdown("---")

            # 4. Definitions & Terminology
            st.markdown("##### 📚 Key Definitions & Terminology")
            terms = active_notes.get("definitions_and_terms", [])
            if terms:
                for t_idx, term_obj in enumerate(terms):
                    t_col1, t_col2, t_col3 = st.columns([2, 5, 2])
                    with t_col1:
                        st.markdown(f"**`{term_obj.get('term', '')}`**")
                    with t_col2:
                        st.markdown(term_obj.get("definition", ""))
                    with t_col3:
                        ts_val = term_obj.get("timestamp_ref", "00:00")
                        if st.button(f"🎯 [{ts_val}]", key=f"term_seek_{notes_cache_key}_{t_idx}"):
                            parts = ts_val.split(":")
                            sec_val = 0
                            if len(parts) == 2:
                                sec_val = int(parts[0]) * 60 + int(parts[1])
                            st.session_state["seek_time"] = sec_val
                            st.rerun()

            st.markdown("---")

            # 5. Common Pitfalls & Exam Traps
            st.markdown("##### ⚠️ Common Pitfalls & Exam Traps")
            for pit in active_notes.get("common_pitfalls", []):
                st.warning(f"🛑 {pit}")

            # 6. Revision Checklist
            st.markdown("##### ✅ Quick Revision Checklist")
            for chk_idx, chk in enumerate(active_notes.get("revision_checklist", [])):
                st.checkbox(chk, key=f"chk_{notes_cache_key}_{chk_idx}")
        else:
            st.info("👆 Click **'📄 Generate Smart Notes'** above to generate comprehensive revision notes for this lecture.")

    # =========================================================================
    # TAB 3: PRACTICE QUIZ (Full Integration)
    # =========================================================================
    with tab_quiz:
        st.markdown(f"#### 🎯 Interactive Practice Quiz: Tutorial #{current_lec.get('tutorial_number')}")
        st.caption(f"**Lecture:** {current_lec.get('lecture_title')} | Conceptual multiple-choice questions with instant scoring and revision bookmarks.")

        # Quiz Generation Controls
        q_col1, q_col2, q_col3, q_col4 = st.columns([2, 2, 2, 2])
        with q_col1:
            quiz_q_count = st.selectbox(
                "Questions:",
                options=[5, 10],
                index=0,
                key=f"quiz_count_{current_lec['lecture_id']}"
            )
        with q_col2:
            quiz_diff = st.selectbox(
                "Difficulty:",
                options=["Easy", "Medium", "Hard"],
                index=1,
                key=f"quiz_diff_{current_lec['lecture_id']}"
            )
        with q_col3:
            quiz_lang_opts = ["English", "Hindi", "Hinglish"]
            default_q_lang_idx = 0
            if "Hindi" in lang_choice:
                default_q_lang_idx = 1
            elif "Hinglish" in lang_choice:
                default_q_lang_idx = 2
            quiz_lang = st.selectbox(
                "Language:",
                options=quiz_lang_opts,
                index=default_q_lang_idx,
                key=f"quiz_lang_{current_lec['lecture_id']}"
            )
        with q_col4:
            st.write("")
            st.write("")
            gen_quiz_btn = st.button("🚀 Generate Quiz", key=f"btn_gen_quiz_{current_lec['lecture_id']}", use_container_width=True)

        quiz_cache_key = f"{current_lec['lecture_id']}_{quiz_diff}_{quiz_q_count}_{quiz_lang}"

        # Automatic generation if not present, or manual trigger
        if gen_quiz_btn or (quiz_cache_key not in st.session_state["active_quizzes"]):
            with st.spinner("Generating lecture practice questions..."):
                q_gen = QuizGenerator(llm_client=LLMClient(mock_mode=(not is_live_mode)))
                try:
                    quiz_res = q_gen.generate_quiz(
                        lecture_id=current_lec["lecture_id"],
                        difficulty=quiz_diff,
                        num_questions=quiz_q_count,
                        language=quiz_lang,
                        force_refresh=bool(gen_quiz_btn)
                    )
                    if quiz_res.get("error"):
                        st.session_state["quiz_error"] = quiz_res.get("error")
                    else:
                        st.session_state["active_quizzes"][quiz_cache_key] = quiz_res
                        st.session_state.pop("quiz_error", None)
                        if bool(gen_quiz_btn) or (quiz_cache_key not in st.session_state["quiz_submitted"]):
                            st.session_state["quiz_submitted"][quiz_cache_key] = False
                            st.session_state["quiz_results"][quiz_cache_key] = None
                            st.session_state["quiz_answers"][quiz_cache_key] = {}
                except Exception as e:
                    st.session_state["quiz_error"] = str(e)

        if st.session_state.get("quiz_error"):
            st.warning("⚠️ The AI service is temporarily experiencing high demand. Please try again later or switch to Offline Demonstration mode.")
            if st.button("👉 Switch to Offline Demonstration Mode", key=f"switch_offline_quiz_{current_lec['lecture_id']}"):
                st.session_state["backend_exec_mode"] = "Offline Demonstration (Zero Quota)"
                st.session_state.pop("quiz_error", None)
                st.rerun()

        # Render Active Quiz
        if quiz_cache_key in st.session_state["active_quizzes"]:
            current_quiz = st.session_state["active_quizzes"][quiz_cache_key]
            questions = current_quiz.get("questions", [])
            is_submitted = st.session_state["quiz_submitted"].get(quiz_cache_key, False)
            eval_result = st.session_state["quiz_results"].get(quiz_cache_key)

            is_mock_q = current_quiz.get("is_mock", (not is_live_mode))
            is_cached_q = current_quiz.get("is_cached", False)
            gen_model_q = current_quiz.get("generating_model", "gemini-3.5-flash-lite")
            if is_mock_q:
                q_badge = '<span style="background:#F1F5F9; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #CBD5E1;">🧪 Offline Demonstration</span>'
            elif is_cached_q:
                q_badge = f'<span style="background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #BFDBFE;">⚡ Cached ({gen_model_q})</span>'
            else:
                q_badge = f'<span style="background:#ECFDF5; color:#047857; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600; border:1px solid #A7F3D0;">⚡ Live AI ({gen_model_q})</span>'

            st.markdown(f"**Difficulty:** `{quiz_diff}` | **Questions:** `{len(questions)}` | {q_badge}", unsafe_allow_html=True)

            st.markdown("---")

            # Score Banner & Topic Analytics if submitted
            if is_submitted and eval_result:
                score = eval_result["score"]
                total = eval_result["total"]
                pct = eval_result["percentage"]
                grade = eval_result["grade"]

                banner_color = "#ECFDF5" if pct >= 80 else ("#FFFBEB" if pct >= 60 else "#FEF2F2")
                border_color = "#10B981" if pct >= 80 else ("#F59E0B" if pct >= 60 else "#EF4444")
                text_color = "#065F46" if pct >= 80 else ("#92400E" if pct >= 60 else "#991B1B")

                st.markdown(f"""
                <div style="background:{banner_color}; border:2px solid {border_color}; border-radius:10px; padding:1.25rem; margin-bottom:1.5rem; text-align:center;">
                    <h3 style="margin:0; color:{text_color}; font-size:1.6rem;">{grade}</h3>
                    <p style="font-size:1.2rem; font-weight:600; margin:0.4rem 0; color:#1E293B;">
                        Score: {score} / {total} ({pct}%)
                    </p>
                    <p style="color:#475569; font-size:0.95rem; margin:0;">
                        {eval_result.get('feedback_message', '')}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                analytics_col1, analytics_col2 = st.columns(2)
                with analytics_col1:
                    st.markdown("##### 🌟 Mastered Topics")
                    strong = eval_result.get("strong_topics", [])
                    if strong:
                        for stp in strong:
                            st.success(f"✅ {stp}")
                    else:
                        st.caption("No topics fully mastered yet. Review key timestamps below!")

                with analytics_col2:
                    st.markdown("##### 📚 Recommended Topics for Revision")
                    weak = eval_result.get("weak_topics", [])
                    if weak:
                        for w_i, wtp in enumerate(weak):
                            w_top = wtp.get("topic", "")
                            w_acc = wtp.get("accuracy", "")
                            w_ts = wtp.get("recommended_timestamp", "00:00")
                            w_subcol1, w_subcol2 = st.columns([3, 2])
                            with w_subcol1:
                                st.warning(f"🛑 **{w_top}** ({w_acc} correct)")
                            with w_subcol2:
                                if st.button(f"🎯 Review [{w_ts}]", key=f"rev_seek_{quiz_cache_key}_{w_i}", use_container_width=True):
                                    parts = w_ts.split(":")
                                    sec_val = 0
                                    if len(parts) == 2:
                                        sec_val = int(parts[0]) * 60 + int(parts[1])
                                    st.session_state["seek_time"] = sec_val
                                    st.rerun()
                    else:
                        st.balloons()
                        st.success("🎉 Outstanding! You have no weak topics in this quiz.")

                retake_col1, retake_col2 = st.columns([2, 4])
                with retake_col1:
                    if st.button("🔄 Retake Quiz", key=f"btn_retake_{quiz_cache_key}", use_container_width=True):
                        st.session_state["quiz_submitted"][quiz_cache_key] = False
                        st.session_state["quiz_results"][quiz_cache_key] = None
                        st.session_state["quiz_answers"][quiz_cache_key] = {}
                        st.rerun()

                st.markdown("---")
                st.markdown("#### 📋 Detailed Question Review:")

            # Render Questions
            user_ans_dict = st.session_state["quiz_answers"].setdefault(quiz_cache_key, {})

            for q_idx, q in enumerate(questions):
                qid = q.get("id", q_idx + 1)
                q_text = q.get("question", "")
                options = q.get("options", [])
                topic = q.get("topic", "")
                ts_ref = q.get("timestamp_ref", "00:00")

                st.markdown(f"**Question {q_idx + 1} of {len(questions)}** <span style='background:#E2E8F0; padding:2px 8px; border-radius:12px; font-size:0.75rem; color:#475569;'>{topic}</span>", unsafe_allow_html=True)
                st.markdown(f"##### {q_text}")

                if not is_submitted:
                    chosen_idx = user_ans_dict.get(qid, None)
                    current_idx = chosen_idx if chosen_idx is not None else None

                    selected_option = st.radio(
                        label=f"Options for Q{qid}:",
                        options=options,
                        index=current_idx if (current_idx is not None and 0 <= current_idx < len(options)) else None,
                        key=f"radio_{quiz_cache_key}_{qid}",
                        label_visibility="collapsed"
                    )

                    if selected_option is not None:
                        user_ans_dict[qid] = options.index(selected_option)

                    st.markdown("<br>", unsafe_allow_html=True)
                else:
                    q_res = eval_result["question_results"][q_idx] if eval_result else None
                    is_q_correct = q_res.get("is_correct", False) if q_res else False
                    user_chosen = q_res.get("user_answer", None) if q_res else None
                    corr_chosen = q.get("correct_index", 0)

                    for o_idx, opt in enumerate(options):
                        is_this_correct = (o_idx == corr_chosen)
                        is_this_user = (o_idx == user_chosen)

                        if is_this_correct:
                            st.markdown(f"<div style='background:#ECFDF5; border-left:4px solid #10B981; padding:6px 12px; margin:4px 0; border-radius:4px; color:#065F46; font-size:0.9rem;'>✅ <strong>{opt}</strong> <em>(Correct Answer)</em></div>", unsafe_allow_html=True)
                        elif is_this_user and not is_q_correct:
                            st.markdown(f"<div style='background:#FEF2F2; border-left:4px solid #EF4444; padding:6px 12px; margin:4px 0; border-radius:4px; color:#991B1B; font-size:0.9rem;'>❌ <strong>{opt}</strong> <em>(Your Answer)</em></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='padding:6px 12px; margin:4px 0; color:#64748B; font-size:0.9rem;'>⚪ {opt}</div>", unsafe_allow_html=True)

                    exp_col1, exp_col2 = st.columns([4, 1])
                    with exp_col1:
                        st.info(f"💡 **Explanation:** {q.get('explanation', '')}")
                    with exp_col2:
                        st.write("")
                        if st.button(f"🎯 Seek [{ts_ref}]", key=f"q_seek_{quiz_cache_key}_{qid}", use_container_width=True):
                            parts = ts_ref.split(":")
                            sec_val = 0
                            if len(parts) == 2:
                                sec_val = int(parts[0]) * 60 + int(parts[1])
                            st.session_state["seek_time"] = sec_val
                            st.rerun()

                    st.markdown("---")

            if not is_submitted:
                answered_count = len([k for k, v in user_ans_dict.items() if v is not None])
                s_col1, s_col2 = st.columns([2, 5])
                with s_col1:
                    submit_quiz_btn = st.button("📝 Submit Quiz", key=f"btn_sub_quiz_{quiz_cache_key}", use_container_width=True)
                with s_col2:
                    st.caption(f"Progress: {answered_count} of {len(questions)} answered.")

                if submit_quiz_btn:
                    if answered_count < len(questions):
                        st.warning(f"⚠️ You answered {answered_count} of {len(questions)} questions. Please answer all questions before submitting.")
                    else:
                        q_gen = QuizGenerator()
                        evaluation = q_gen.evaluate_quiz_submission(current_quiz, user_ans_dict)
                        st.session_state["quiz_submitted"][quiz_cache_key] = True
                        st.session_state["quiz_results"][quiz_cache_key] = evaluation
                        st.rerun()
        else:
            st.info("👆 Click **'🚀 Generate Quiz'** above to generate an interactive practice quiz for this lecture.")


if __name__ == "__main__":
    main()
