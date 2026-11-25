# LearnMate AI — Free Deployment Feasibility & Hosting Guide
**PhysicsWallah APM Builder Screening Project**  
*Step-by-Step Instructions for 100% Free Cloud Deployment and Local Reproducibility*

---

## 1. Free Deployment Feasibility Overview

LearnMate AI was designed from Day 1 to run on zero-cost infrastructure:
- **Zero API Cost:** Primary reasoning operates on Google Gemini Free Tier (`gemini-3.5-flash-lite` and `gemini-3.8-flash`), consuming ₹0.00 / $0.00.
- **Lightweight Inference:** The embedding model (`intfloat/multilingual-e5-small`, 384 dimensions) requires under 250 MB RAM on CPU.
- **No External Database Fees:** Vector storage utilizes local in-memory FAISS IndexFlatIP with serialized JSON chunk metadata.
- **No GPU Requirement:** Whisper speech-to-text and SentenceTransformers run efficiently on standard x86 CPU threads.

---

## 2. Option 1: Streamlit Community Cloud (Recommended Free Tier)

**Streamlit Community Cloud** provides 100% free hosting directly from a public or private GitHub repository.

### Prerequisites
- A GitHub account.
- Free Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### Deployment Steps
1. **Push Repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "feat: LearnMate AI submission release"
   git branch -M main
   git remote add origin https://github.com/<your-username>/LearnMate-AI.git
   git push -u origin main
   ```
   *(Note: Large media files like `.webm` videos and `.zip` archives are automatically excluded by `.gitignore` so your repo size stays under 10 MB).*

2. **Deploy on Streamlit Cloud:**
   - Log into [share.streamlit.io](https://share.streamlit.io/).
   - Click **"Create app"** > **"Deploy a public app from GitHub"**.
   - Select your repository: `<your-username>/LearnMate-AI`.
   - Branch: `main`.
   - Main file path: `app.py`.
   - App URL: Choose a custom subdomain (e.g. `learnmate-ai.streamlit.app`).

3. **Configure Environment Secrets:**
   - In the Streamlit Cloud app settings, open the **"Secrets"** tab.
   - Paste the following configuration:
     ```toml
     LLM_PROVIDER = "gemini"
     GEMINI_API_KEY = "AIzaSy..."
     GEMINI_MODEL = "gemini-3.5-flash-lite"
     GEMINI_FALLBACK_MODEL = "gemini-3.8-flash"
     ENABLE_MODEL_FALLBACK = true
     LLM_MODE = "live"
     ```
   - Click **Save**. The application will automatically rebuild and deploy.

4. **Resource Verification:**
   - RAM usage: ~450 MB (well below the 1.0 GB free-tier threshold).
   - CPU: 1 vCPU is fully sufficient for FAISS matrix dot products.

---

## 3. Option 2: Hugging Face Spaces (Alternative Free Tier)

Hugging Face Spaces offers a generous free tier featuring **2 vCPUs and 16 GB RAM** with built-in Streamlit SDK support.

### Deployment Steps
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Space SDK: Select **Streamlit**.
3. Space Hardware: Select **Free - 2 vCPU · 16 GB · 50 GB Disk**.
4. In **Settings > Variables and secrets**, add:
   - Secret Name: `GEMINI_API_KEY`, Secret Value: `your_gemini_api_key`.
   - Variable Name: `LLM_PROVIDER`, Value: `gemini`.
   - Variable Name: `GEMINI_MODEL`, Value: `gemini-3.5-flash-lite`.
5. Push your code to the Hugging Face Space Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/LearnMate-AI
   git push space main
   ```

---

## 4. Option 3: Local Reproducible Execution (Self-Hosted)

For zero-network evaluation with full local video playback:

```bash
# 1. Clone or unpack repository
git clone https://github.com/<your-username>/LearnMate-AI.git
cd LearnMate-AI

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
# Edit .env and enter your GEMINI_API_KEY (optional if running in Offline Demo mode)

# 6. Launch Streamlit Application
streamlit run app.py --server.port 8501
```
Open **`the local Streamlit port (typically http://localhost:8501)`** in your browser.

---

## 5. Media & Video Playback Architecture in Cloud vs. Local

| Component | Local Execution (`localhost:8501`) | Cloud Deployment (Streamlit Cloud) |
|---|---|---|
| **Video Source** | Local WebM files in `data/videos/` | Local lightweight clips or embedded iframe player |
| **Timestamp Seeking** | Dynamic `st.video(..., start_time=seek_sec)` | Dynamic seeking to exact seconds |
| **Doubt Resolution** | Transcript-grounded via FAISS & citation verification | Transcript-grounded via FAISS & citation verification |
| **Smart Notes** | Full multi-style generation & markdown download | Full multi-style generation & markdown download |
| **Practice Quiz** | Real-time interactive assessment & analytics | Real-time interactive assessment & analytics |
| **Repository Size** | ~480 MB (with local video lectures) | **< 10 MB** (pure code & transcript metadata) |

---

## 6. Live Feature Verification & Prototype Disclosures

> [!NOTE]
> **Empirical Feature Verification Status:**
> - **Ask AI:** Successfully verified in both Live AI (Gemini 3.5 Flash-Lite) and Offline Demonstration modes.
> - **Smart Notes:** Manually verified in Offline Demonstration mode. Live generation encountered HTTP 503 capacity limits on free-tier Gemini models.
> - **Practice Quiz:** Manually verified in Offline Demonstration mode. Live generation encountered HTTP 503 capacity limits on free-tier Gemini models.
> - **Dual-Model Fallback:** Fallback logic is implemented in `llm.py`, but successful live fallback generation remains unverified due to concurrent upstream 503 spikes.
> - **Scope:** LearnMate AI is an educational prototype and screening MVP, not a production-grade commercial system. Hallucination mitigation reduces ungrounded responses, but absolute hallucination prevention cannot be guaranteed.

