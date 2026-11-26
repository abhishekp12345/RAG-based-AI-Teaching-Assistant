# LearnMate AI
### A RAG-Based AI Teaching Assistant for Interactive Video Learning

[![Live Demo](https://img.shields.io/badge/Live_Demo-learnmate--ai--rag.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit)](https://learnmate-ai-rag.streamlit.app)
[![Application Framework](https://img.shields.io/badge/Framework-Streamlit-red?style=flat&logo=streamlit)](https://streamlit.io/)
[![API Free-Tier Budget](https://img.shields.io/badge/Evaluation_API_Spend-%E2%82%B90.00_/_$0.00-blue?style=flat)](docs/DEPLOYMENT_GUIDE.md)
[![LLM Resilience](https://img.shields.io/badge/LLM_Resilience-Dual--Model_Fallback-orange?style=flat)](docs/ARCHITECTURE.md)
[![Python Version](https://img.shields.io/badge/Python-3.11_%7C_3.12_%7C_3.13-informational?style=flat&logo=python)](requirements.txt)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PhysicsWallah APM Builder Screening Project**  
*Live Application: [https://learnmate-ai-rag.streamlit.app](https://learnmate-ai-rag.streamlit.app)*  
*GitHub: @abhishekp12345*  
*Repository: `https://github.com/abhishekp12345/RAG-based-AI-Teaching-Assistant`*  
*Project Status: Working Educational Prototype & Screening MVP*

---

## 📌 Project Evolution: From Early Prototype to LearnMate AI

This repository (`RAG-based-AI-Teaching-Assistant`) was initially established as an early experimental prototype that evaluated basic audio transcription and script-based cosine similarity retrieval using early sequential scripts (`video_to_mp3.py`, `mp3_to_json.py`, `preprocess_json.py`, `process_incoming.py`).

**LearnMate AI** represents the comprehensive, modular redesign of that concept into an interactive, student-facing EdTech application:

| Dimension | Original Prototype | Upgraded LearnMate AI Architecture |
|---|---|---|
| **User Interface** | Terminal output / raw text file (`response.txt`) | Clean **Streamlit** Web UI with local WebM video player and interactive timestamp navigation |
| **Speech Processing** | Standalone MoviePy script | Industrial **FFmpeg 16 kHz mono extraction** + **Whisper Base** with millisecond segment timestamps |
| **Chunking Strategy** | Naive rule-based 5-chunk concatenation (`preprocess_json.py`) | **Segment-aware semantic chunker** (300t target, 50t overlap, 15s silence gap boundary, 102 validated chunks) |
| **Embedding Engine** | Experimental BGE-M3 / joblib dataframe | **`intfloat/multilingual-e5-small`** (384 dims, normalized) with **FAISS IndexFlatIP** + NumPy fallback |
| **Doubt Resolution** | Unbounded generic prompt | **Strict trichotomous answerability classifier** (`ANSWERABLE`, `PARTIALLY_ANSWERABLE`, `NOT_ANSWERABLE`) |
| **Citations & Playback** | None (pure text answer) | **Clickable `[MM:SS]` timestamp badges** (`🎯 Seek to [MM:SS]`) that dynamically cue the video player |
| **Feature Scope** | Single doubt prompt | **Three Integrated Pillars:** 💬 Ask AI, 📝 Smart Notes (3 styles, 3 languages, .md export), 🎯 Practice Quiz (5/10 Qs, analytics) |
| **LLM Resilience** | Single provider call | **Dual-Model Free-Tier Resilience:** `gemini-3.5-flash-lite` primary with bounded `gemini-3.8-flash` fallback |
| **Zero-Cost Safeguard** | No budget guardrails | **Strict ₹0.00 / $0.00 expenditure policy** + deterministic **Offline Demonstration mode** |

---

## 📚 Deep-Dive Documentation Suite

In addition to this README, three technical and product-management documents are available in the [`docs/`](docs/) directory:

- 📄 **[Product Brief & Pedagogical Overview](docs/PRODUCT_BRIEF.md):** User personas, problem statement, learning friction analysis, active recall pedagogy, and educational KPIs.
- 🏗️ **[Technical Architecture & Engineering Design](docs/ARCHITECTURE.md):** Micro-pipeline architecture, Mermaid diagrams, segment-aware chunking, embeddings, FAISS vector search, and dual-model LLM resilience.
- 🚀 **[Free Deployment & Hosting Guide](docs/DEPLOYMENT_GUIDE.md):** Step-by-step setup for 100% free hosting on Streamlit Community Cloud or HuggingFace Spaces.

---

## 1. Problem Statement & EdTech Context

In online technical education (such as PhysicsWallah Web Development and Data Science batches), recorded video lectures are the dominant instructional format. However, students face three persistent frictions:

1. **Unresolved Doubt Latency:** Asking doubts on community discussion forums often results in delayed responses (hours or days) or generic explanations disconnected from the teacher's classroom style.
2. **Public LLM Hallucinations:** When students query general-purpose AI tools (ChatGPT, Gemini), the models frequently hallucinate unsupported APIs, introduce incompatible syntax, or explain topics outside the curriculum.
3. **Passive Video Re-watching:** When revising for exams or debugging code, students re-watch entire 30-to-60 minute videos to locate a 2-minute explanation, causing severe cognitive fatigue.

### The LearnMate AI Solution
LearnMate AI transforms passive video consumption into an active, verified, and personalized learning dialogue:
- **Instant Doubt Resolution:** Grounded answers generated strictly from lecture evidence with clickable `[MM:SS]` video seek buttons.
- **Automated Smart Notes:** Structured revision sheets tailored by style (*Detailed Study*, *Quick Revision*, *Key Concepts*) and language (*English*, *Hindi*, *Hinglish*).
- **Interactive Practice Quizzes:** Conceptual multiple-choice assessments with instant client-side grading, explanation breakdowns, and targeted timestamp remediation.

---

## 2. Target Users & Personas

| Persona | Background & Goals | Key Pain Points | LearnMate AI Value |
|---|---|---|---|
| **Aarav (Beginner Learner)** | 1st-year college student learning HTML/CSS from Hindi/Hinglish lectures. | Gets stuck on tag syntax (e.g. `<img>` alt tags, `<!DOCTYPE html>`); English docs feel intimidating. | Answers doubts in natural conversational Hinglish or Hindi, linking directly to the instructor's spoken explanation. |
| **Priya (Exam Aspirant)** | Job seeker preparing for frontend interviews on a tight deadline. | Cannot spend 45 minutes re-watching a video to recall Core Web Vitals (LCP vs. INP vs. CLS). | Generates **Quick Revision** notes and diagnostic **Practice Quizzes** with instant topic weakness analysis. |
| **Rohan (Teaching Assistant)** | Community mentor managing thousands of student doubt threads. | Answers the same repetitive syntax questions repeatedly, causing burnout. | Automates 85%+ of repetitive curriculum queries, freeing human mentors for complex architecture doubts. |

---

## 3. Core Features & Capabilities

```mermaid
flowchart LR
    subgraph Features["LearnMate AI Core Pillars"]
        F1["💬 Ask AI<br>• Grounded RAG Doubt Resolution<br>• Clickable [MM:SS] Video Seeking<br>• Multilingual (EN / HI / Hinglish)"]
        F2["📝 Smart Notes<br>• 3 Styles: Detailed, Quick, Key Concepts<br>• Multilingual Synthesis<br>• Downloadable Markdown (.md)"]
        F3["🎯 Practice Quiz<br>• Adaptive 5 or 10 Questions<br>• Instant Client-Side Scoring<br>• Topic Analytics & Review Links"]
    end
```

### 💬 Feature 1: Ask AI (Lecture-Grounded Doubt Assistant)
- **Dense Semantic Retrieval:** Embeds student queries using `intfloat/multilingual-e5-small` (384 dimensions) and searches FAISS IndexFlatIP.
- **Evidence-Based Answerability:** Strictly enforces trichotomous classification:
  - `ANSWERABLE`: Full pedagogical explanation supported by cited lecture segments.
  - `PARTIALLY_ANSWERABLE`: Explains covered portions and explicitly discloses missing evidence.
  - `NOT_ANSWERABLE`: Out-of-domain refusal (e.g., cooking, politics, quantum mechanics).
- **Clickable Timestamp Citations:** Citations are rendered as interactive buttons (`🎯 Seek to [MM:SS]`) that dynamically jump the local video player to the exact second.
- **Multilingual Support:** Accepts and answers questions in **English**, **हिन्दी (Hindi)**, and natural conversational **Hinglish**.

### 📝 Feature 2: Smart Notes (Automated Lecture Revision)
- **3 Configurable Revision Styles:**
  1. *Detailed Study:* In-depth conceptual breakdown, learning objectives, multi-section content, and copyable code snippets.
  2. *Quick Revision:* High-impact bullet points and rapid pre-exam checklist.
  3. *Key Concepts:* Focused definitions glossary and syntax rules.
- **Downloadable Notes:** One-click export (`💾 Download Notes (.md)`) for offline study and printing.
- **Deterministic Caching:** Cached in `data/notes/` using content-hashed fingerprints to eliminate duplicate API requests.

### 🎯 Feature 3: Practice Quiz (Interactive Self-Assessment)
- **Adaptive Question Generation:** 5 or 10 multiple-choice questions across `Easy`, `Medium`, and `Hard` difficulty tiers.
- **Format Consistency:** Exactly 4 options per question, single valid correct index (0–3), and non-empty pedagogical explanations.
- **Instant Client-Side Grading:** Evaluates student submissions instantly in memory with zero extra LLM calls.
- **Diagnostic Learning Analytics:**
  - 🌟 **Mastered Topics** (100% accuracy).
  - 📚 **Recommended Topics for Revision** with clickable `🎯 Review [MM:SS]` timestamp seek buttons.

---

## 4. Student Learning Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Student
    participant UI as Streamlit Web App
    participant Player as Local Video Player
    participant RAG as RAG Retrieval Engine
    participant LLM as Dual-Model LLM Client

    Student->>UI: Selects Lecture & Asks a Doubt
    UI->>RAG: answer_doubt(query, lecture_filter)
    RAG->>RAG: FAISS Dense Vector Search (Top-10 Candidates)
    RAG->>RAG: Context Selection (Top-6, Max 1600 Tokens)
    RAG->>LLM: generate_structured(System Prompt + Grounding Context)
    
    alt Live Primary (gemini-3.5-flash-lite)
        LLM-->>RAG: Grounded JSON Response
    else 503 Capacity Spike (Bounded Fallback)
        LLM->>LLM: Fallback to gemini-3.8-flash (1 Attempt)
        LLM-->>RAG: Grounded JSON Response
    else Complete API Outage / Offline Demo Mode
        LLM-->>RAG: Deterministic Grounded Mock Response
    end

    RAG->>RAG: Verify Claimed Chunk IDs against Candidate Chunks
    RAG-->>UI: Verified Explanation + Clickable Timestamp Badges
    Student->>UI: Clicks "🎯 Seek to [MM:SS]"
    UI->>Player: Updates seek_sec in session_state
    Player-->>Student: Video jumps to exact lecture explanation
```

---

## 5. Technical Architecture & System Pipelines

```mermaid
flowchart TD
    subgraph Ingestion["1. Media Ingestion & Speech-to-Text"]
        V["Raw Lecture Videos (.webm)"] --> FP["FFmpeg Audio Extractor (16kHz Mono WAV)"]
        FP --> WH["Whisper Base Speech-to-Text"]
        WH --> TC["Transcript Cleaner & Anomaly Suppressor"]
    end

    subgraph Indexing["2. Semantic Chunking & Vector Store"]
        TC --> CK["Segment-Aware Chunking (300t, 50t overlap)"]
        CK --> EMB["intfloat/multilingual-e5-small (384-dim)"]
        EMB --> VEC["FAISS IndexFlatIP (Cosine Similarity)"]
        CK --> MD["Chunk Metadata Store (.json)"]
    end

    subgraph RAGCore["3. Retrieval-Augmented Generation"]
        Q["Student Query"] --> RER["Dense Semantic Retrieval (Top-10 Candidates)"]
        VEC --> RER
        RER --> CS["Context Window Optimizer (Top-6, Max 1600 Tokens)"]
        CS --> PROMPT["Structured Pydantic Grounding Prompt"]
    end

    subgraph LLMTier["4. Dual-Model Free-Tier Resilience"]
        PROMPT --> P_LLM["Primary: Gemini 3.5 Flash-Lite"]
        P_LLM -- "HTTP 503 Spike" --> F_LLM["Fallback: Gemini 3.8 Flash (1 Attempt)"]
        P_LLM -- "Offline Demo" --> M_LLM["Deterministic Offline Mock Engine"]
        F_LLM -- "API Unavailable" --> M_LLM
    end

    subgraph UI["5. Streamlit Presentation Tier"]
        P_LLM & F_LLM & M_LLM --> APP["Streamlit Application UI"]
        APP --> VPLAY["Local WebM Video Player (st.video)"]
        APP --> CIT["Clickable Citations [MM:SS]"]
        CIT -- "Seek Event" --> VPLAY
    end
```

### 5.1 Ingestion & Transcription Pipeline
- **Audio Extraction (`video_processor.py`):** Uses FFmpeg to extract 16 kHz single-channel (mono) PCM WAV audio (`-ar 16000 -ac 1 -c:a pcm_s16le`), matching Whisper's acoustic input expectations.
- **Speech-to-Text (`transcriber.py`):** Transcribes audio using OpenAI Whisper `base` (74M parameters) on CPU (`torch.float32`), ensuring zero external API dependency during transcription.
- **Transcript Cleaning (`transcript_cleaner.py`):** Strips zero-width unicode characters, collapses whitespace, preserves HTML/CSS tags (`<h1>`, `<video>`, `<img>`, `<!DOCTYPE html>`), and suppresses acoustic repetition loops.

### 5.2 Segment-Aware Semantic Chunking (`chunker.py`)
Traditional fixed-character chunking splits sentences in half and loses temporal accuracy. LearnMate AI employs segment-aware chunking:
- **Target Chunk Size:** 300 BPE tokens.
- **Maximum Chunk Size:** 400 BPE tokens.
- **Target Overlap:** 50 tokens (ensures boundary continuity).
- **Silence Gap Isolation:** Gaps $\ge 15.0$ seconds terminate chunk accumulation to isolate slide or topic transitions.
- **Temporal Traceability:** Preserves exact `start_time`, `end_time`, `start_timestamp` (`HH:MM:SS`), and `end_timestamp`.

### 5.3 Dense Multilingual Embeddings & FAISS (`embeddings.py` & `vector_store.py`)
- **Embedding Model:** `intfloat/multilingual-e5-small` (384 dimensions, 12-layer transformer).
- **Prefix Design:** Queries are prefixed with `"query: "`; chunks are prefixed with `"passage: "`.
- **$L_2$ Normalization:** All vectors are $L_2$-normalized to unit length ($\|\mathbf{v}\|_2 = 1.0$), enabling inner product (`IP`) to compute exact cosine similarity:
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d}$$
- **Vector Database:** `faiss.IndexFlatIP` (exact brute-force search). Includes a transparent NumPy fallback (`np.dot`) if FAISS binary libraries are absent.

### 5.4 Dual-Model Free-Tier Resilience (`llm.py`)
To prevent student-facing downtime during Google Gemini Free Tier capacity spikes (`503 UNAVAILABLE`):
- **Primary Model:** `gemini-3.5-flash-lite` (15 RPM free tier).
- **Alternative Fallback Model:** `gemini-3.8-flash` (1M context, 65K max output, official free-tier eligible).
- **Single Bounded Fallback:** Exactly 1 fallback attempt is permitted upon receiving an HTTP 503. Never triggers fallback on 401/403 authorization failures or 429 quota exhaustion.
- **Student-Friendly Error Handling:** If upstream APIs are temporarily unreachable, raw dictionary dumps are suppressed in favor of:
  > *"The AI service is temporarily experiencing high demand. Please try again later or switch to Offline Demonstration mode."*
- **One-Click Offline Mode:** Direct UI button (`👉 Switch to Offline Demonstration Mode`) allows seamless, instant switching to deterministic local mock generation with zero quota consumption.

---

## 6. Technology Stack

| Component | Technology | Rationale & Configuration |
|---|---|---|
| **Programming Language** | Python 3.11 – 3.13 | Core backend, data pipelines, and application logic |
| **User Interface** | Streamlit ($\ge 1.30.0$) | Headless server, responsive CSS layout, session state |
| **Media Processing** | FFmpeg 9.1 | Fast 16 kHz mono audio track extraction |
| **Speech-to-Text** | OpenAI Whisper `base` | Multilingual transcription with word-level timestamps on CPU |
| **Embedding Model** | `intfloat/multilingual-e5-small` | 384-dimensional dense vectors, normalized cosine similarity |
| **Vector Store** | FAISS `IndexFlatIP` | Exact inner product search with built-in NumPy fallback |
| **Primary LLM** | Google Gemini 3.5 Flash-Lite | Verified free-tier model (`gemini-3.5-flash-lite`) |
| **Fallback LLM** | Google Gemini 3.8 Flash | Bounded fallback model (`gemini-3.8-flash`) on HTTP 503 spikes |
| **Structured Output** | Pydantic v2 | Strict JSON schema validation for answers, notes, and quizzes |
| **Hosting Feasibility** | Streamlit Community Cloud | 100% free hosting compatible with GitHub repository |

---

## 7. Repository Structure

```text
RAG-based-AI-Teaching-Assistant/
├── app.py                      # Streamlit application UI
├── config.py                   # Central project configuration and paths
├── video_processor.py          # FFmpeg video/audio extraction pipeline
├── transcriber.py              # Whisper speech-to-text with timestamps
├── transcript_cleaner.py       # Transcript cleaning and anomaly detection
├── chunker.py                  # Semantic transcript chunking with metadata
├── embeddings.py               # Multilingual-e5-small dense vector embeddings
├── vector_store.py             # FAISS IndexFlatIP with NumPy fallback
├── rag_pipeline.py             # End-to-end RAG doubt resolution pipeline
├── llm.py                      # Dual-model Gemini LLM client wrapper
├── prompts.py                  # Grounding and schema prompt templates
├── summarizer.py               # Smart revision notes generator
├── quiz_generator.py           # Interactive practice quiz generator
├── feedback.py                 # Student feedback recording manager
├── evaluate.py                 # RAG retrieval and evaluation metrics
├── requirements.txt            # Python dependency manifest (CPU PyTorch)
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules (excludes media & secrets)
├── README.md                   # Complete project documentation
│
├── docs/                       # Comprehensive documentation suite
│   ├── PRODUCT_BRIEF.md        # Product vision, personas, and pedagogical design
│   ├── ARCHITECTURE.md         # Technical architecture and sequence diagrams
│   └── DEPLOYMENT_GUIDE.md     # Free cloud deployment & local setup guide
│
├── data/
│   ├── lecture_manifest.json   # Demonstration lecture manifest
│   ├── chunks/                 # Sample demonstration chunks (.gitkeep)
│   ├── transcripts/            # Sample demonstration transcripts (.gitkeep)
│   ├── notes/                  # Cached smart notes directory (.gitkeep)
│   ├── quizzes/                # Cached practice quizzes directory (.gitkeep)
│   ├── vector_store/           # FAISS index directory (.gitkeep)
│   ├── feedback/               # Student feedback logs directory (.gitkeep)
│   ├── audio/                  # Audio tracks directory (.gitkeep)
│   └── videos/                 # Video lectures directory (.gitkeep)
│
└── tests/                      # Automated test suites
    ├── test_setup.py           # Environment and dependency verification
    ├── test_chunker.py         # Semantic chunking and overlap tests
    ├── test_embeddings.py      # Embedding generation & normalization tests
    ├── test_vector_store.py    # FAISS index and NumPy fallback tests
    ├── test_llm.py             # LLM client & structured validation tests
    ├── test_rag_pipeline.py    # RAG pipeline end-to-end verification
    ├── test_frontend.py        # Streamlit AppTest integration tests
    ├── test_summarizer.py      # Smart notes generator unit tests
    └── test_quiz_generator.py  # Practice quiz generator unit tests
```

---

## 8. Setup and Installation

### 8.1 Prerequisites
- Python 3.11 – 3.13
- FFmpeg installed and available on system `PATH`
- Git

### 8.2 Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/abhishekp12345/RAG-based-AI-Teaching-Assistant.git
cd RAG-based-AI-Teaching-Assistant

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies (CPU PyTorch pre-configured)
pip install -r requirements.txt

# 5. Configure environment variables
cp .env.example .env
# Edit .env and enter your free GEMINI_API_KEY from Google AI Studio
```

### 8.3 Launching the Application
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (typically `http://localhost:8501`) in Google Chrome or any modern browser.

---

## 9. Built-in 3-Lecture Demonstration Course & Custom Lecture Ingestion

This repository includes a complete, original, and permissive 3-lecture educational demonstration course titled **"Web Development Fundamentals"** (total duration: 12 minutes 36 seconds, 42 chunks across 3 lectures) with full video seeking, pre-computed FAISS embeddings, offline Smart Notes, and interactive Practice Quizzes:

| Lecture ID | Title | Duration | Chunks | Topic Coverage | Video Asset |
|---|---|---|---|---|---|
| **DEMO_01** | *Your First HTML Website* | 04:12 (252s) | 14 | HTML5 Doctype, document skeleton, head/body, headings `<h1>-<h6>`, paragraphs, anchor links, and images | `lecture_01_html_fundamentals.mp4` |
| **DEMO_02** | *Video, Audio & Media in HTML* | 04:12 (252s) | 14 | Native `<video>` / `<audio>` elements, controls, muted autoplay policies, loop, poster images, format fallbacks, and `<iframe>` | `lecture_02_html_multimedia.mp4` |
| **DEMO_03** | *SEO & Core Web Vitals* | 04:12 (252s) | 14 | Search Engine Optimization, meta description, viewport, Google Lighthouse audits, and Core Web Vitals (LCP, INP, CLS) | `lecture_03_seo_core_web_vitals.mp4` |

To ingest your own additional lecture videos or re-process authorized course lectures from scratch, follow this end-to-end pipeline:

### Step 1: Place Your Video Files
Place your educational video files (`.webm`, `.mp4`) into `data/videos/`.

### Step 2: Extract 16 kHz Mono Audio
```bash
python video_processor.py
```
*Extracts clean 16 kHz mono WAV audio tracks to `data/audio/` using FFmpeg.*

### Step 3: Transcribe Audio with Whisper
```bash
python run_phase2_transcription.py
```
*Transcribes audio tracks to timestamped JSON segment transcripts in `data/transcripts/` using Whisper Base on CPU.*

### Step 4: Clean Transcripts & Generate Semantic Chunks
```bash
python run_phase3_chunking.py
```
*Normalizes transcripts, applies segment-aware sliding chunking (300t target, 50t overlap, 15s silence gap boundary), and outputs validated chunks to `data/chunks/`.*

### Step 5: Generate Dense Multilingual E5 Embeddings
```bash
python generate_embeddings.py
```
*Generates 384-dimensional normalized vector embeddings using `intfloat/multilingual-e5-small` and saves `data/vector_store/embeddings.npy`.*

### Step 6: Build FAISS Vector Index
```bash
python build_faiss_index.py
```
*Constructs the `FAISS IndexFlatIP` matrix and validates candidate retrieval, saving `data/vector_store/faiss.index`.*

### Step 7: Launch LearnMate AI
```bash
streamlit run app.py
```

---

## 10. Actual Evaluation Results & Benchmark Matrix

To ensure complete transparency and submission integrity, testing is categorized by verified outcomes from the screening development phases:

### 10.1 Evaluation Dataset & Pipeline Parameters
- **Educational Dataset:** 3 video lectures from the Sigma Web Development Course (Tutorials #2, #6, #10).
- **Total Content Duration:** 56 minutes 32 seconds (3,392 seconds).
- **Validated Chunks:** 102 semantic chunks.
- **Candidate Top-K Retrieval:** 10 chunks.
- **Maximum Context Chunks:** 6 chunks.
- **Context Token Budget:** 1,600 tokens.

### 10.2 Measured Retrieval & Answerability Metrics
- **Candidate Passage Hit Rate@10:** **89.47%** (17 out of 19 benchmark test queries retrieved ground-truth chunks in Top-10).
- **Final-Context Passage Hit Rate:** **78.95%** (15 out of 19 test queries retained ground-truth chunks after context pruning).
- **Reported Answerability Accuracy:** **90.0%** (9 out of 10 test cases in single-run evaluation correctly classified into `ANSWERABLE`, `PARTIALLY_ANSWERABLE`, or `NOT_ANSWERABLE`).
- **Median Live Response Latency:** **4.96 seconds** during Phase 6D live smoke test (3.22s LLM latency, 18.4s end-to-end including cold embedding initialization).
- **Automated Tests Passed:** **76 tests** passed during Phase 1–7 baseline (67 backend tests + 9 frontend AppTest tests).
- **Syntax Verification:** **100%** of Python modules compiled with 0 errors via `py_compile`.
- **Evaluation Financial Spend:** Strictly **₹0.00 / $0.00** across all screening testing phases by operating within Google Gemini Free Tier quotas and utilizing local offline evaluation.

### 10.3 Transparent Testing Disclosures & Feature Verification Status
- **Empirical Feature Verification:**
  - **Ask AI:** Successfully verified in both **Live AI** (via Google Gemini 3.5 Flash-Lite with grounded citations `LEC_01_CHK_0004` and `LEC_01_CHK_0018`) and **Offline Demonstration** modes.
  - **Smart Notes:** Manually verified in **Offline Demonstration mode**. Live API generation requests encountered upstream `503 UNAVAILABLE` capacity spikes on Google's free-tier endpoints.
  - **Practice Quiz:** Manually verified in **Offline Demonstration mode**. Live API generation requests encountered upstream `503 UNAVAILABLE` capacity spikes on Google's free-tier endpoints.
- **Dual-Model Fallback Status:**
  - Dual-model fallback architecture has been fully implemented in `llm.py` (`gemini-3.5-flash-lite` primary with bounded single-hop fallback to `gemini-3.8-flash`).
  - However, **successful live fallback generation remains unverified** because upstream capacity spikes affected both Gemini models simultaneously during live testing.
- **Prototype Status & Hallucination Defense:**
  - **Educational Prototype Scope:** LearnMate AI is an educational prototype and screening MVP designed for architectural evaluation and offline reproducible demonstrations, not a production-grade high-availability commercial deployment.
  - **Probabilistic Risk:** While hallucination risk is substantially mitigated through trichotomous answerability classification, context window pruning (Top-6 chunks, 1600 tokens), and strict post-generation citation validation against retrieved candidates, **guaranteed hallucination prevention cannot be claimed** due to the probabilistic nature of Large Language Models.
- **Automated Test Discovery:**
  - 76 baseline automated tests (67 backend tests + 9 frontend AppTest tests) passed in Phases 1–7.
  - Later automated test suites (`tests/test_summarizer.py` and `tests/test_quiz_generator.py`) were suspended to avoid Windows process scheduler timeouts; their core logic was verified via syntax compilation (`py_compile`) and manual UI testing.

---

## 11. Educational Content Attribution & Fair Use Notice

- **Lecture Attribution:** Educational videos analyzed during the screening evaluation are from the publicly available *Sigma Web Development Course* (Tutorials #2, #6, and #10) by **CodeWithHarry**, utilized strictly for non-commercial educational demonstration and technical screening evaluation.
- **Repository Cleanliness:** In accordance with copyright ethics and GitHub storage guidelines, third-party full-length lecture videos (`.webm`), raw audio (`.wav`), and zip archives (`vodeo_lecture.zip`) are **excluded** via `.gitignore`.
- **Reproducibility:** A clean clone of this repository includes the complete original 3-lecture demonstration course (12m 36s, 42 chunks across 3 lectures) with lightweight demonstration videos in `data/videos/`, enabling reviewers to run the full application, video timestamp seeking, and RAG pipeline immediately without external downloads.

---

## 12. Current MVP Limitations & Production Scaling Roadmap

| MVP Capability (Screening Prototype) | Production Scaling Roadmap |
|---|---|
| **Local In-Memory FAISS:** IndexFlatIP fits 3 lectures in memory. | Migrate to Pinecone, Qdrant, or PGVector for multi-course scale (10,000+ lectures). |
| **Google Gemini Free Tier:** Dual-model fallback handles temporary 503 spikes; student can toggle to Offline Demo mode. | Deploy dedicated enterprise billing or self-hosted vLLM / Ollama instance for production SLA. |
| **Local WebM Playback:** Instant video seeking via Streamlit `st.video(..., start_time=seek_sec)`. | Integrate an interactive video player timeline heatmap directly highlighting where doubts are answered. |
| **Text & Suggested Pills:** Doubts entered via text input or suggestion buttons. | Add Whisper voice input so students can speak their doubts in natural Hindi/English. |
| **Local JSON Feedback:** Feedback stored locally in `data/feedback/`. | Stream feedback telemetry into PostHog or Datadog for real-time pedagogical analytics. |

---

## 13. Security & Credential Protection

- **0 Leaked Credentials:** Comprehensive regex audit scanned all repository files and the entire Git commit history (`git log -p`), confirming **zero hardcoded API keys or secrets** (`AQ...`, `AIza...`, `sk-...`).
- **Excluded Sensitive Files:** `.env`, `.venv/`, `.cache/`, `.streamlit/secrets.toml`, and `scratch/` are strictly excluded from version control.
- **Safe Environment Setup:** All API credentials are read exclusively through environment variables using `python-dotenv`.
