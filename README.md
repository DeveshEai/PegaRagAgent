# 🦅 Pega CLSA Knowledge Assistant & Autonomous RAG Agent

> **An Autonomous, Zero-Hallucination AI Architecture Agent powered by Groq, ChromaDB, and SentenceTransformers — trained on the complete Pega Lead System Architect (v7) Curriculum & Hands-on Challenges.**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-blue.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq%20Cloud-orange.svg)](https://groq.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 What is this Project? (Explained in Layman's Terms)

Imagine having a **certified Pega Senior Lead System Architect (CLSA)** sitting right next to you 24/7 who has memorized every single page of the official Pega Academy curriculum, architectural guides, and practical challenge walkthroughs.

Whenever you are designing an enterprise system, building a case hierarchy, configuring Constellation UI, or setting up security, you can ask this assistant any question in plain English.

### Why is this different from standard ChatGPT?
* **Normal ChatGPT Hallucinates:** When you ask generic AI about Pega, it often mixes up Pega 7 and Pega Infinity, invents non-existent rules, or guesses configurations that will fail in production.
* **This Assistant is 100% Grounded (Zero-Hallucination Policy):**
  1. It **never guesses**.
  2. It uses an **autonomous agent** with multi-step search tools to pull real excerpts directly from the official Pega PDFs.
  3. Every single statement, rule name, or pattern is accompanied by an exact citation:
     > `[01_fundamental-concepts-constellation.pdf, Page 5]`
  4. If the curriculum does not state something, it admits it upfront rather than making things up.

---

## 🧠 How Does it Work? (The Architecture)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Asks a Question                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               🤖 Autonomous CLSA Agent (Groq LLM Engine)               │
│                                                                        │
│   Plans multi-step research using 4 local autonomous tools:            │
│   ├── 1. `search_curriculum`: Searches 28 Master Architectural Modules  │
│   ├── 2. `search_challenges`: Searches 40 Practical Hands-on Exercises │
│   ├── 3. `read_full_page`: Reads complete original PDF pages           │
│   └── 4. `lookup_pega_terms`: Precision lookup for classes/rules/APIs   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Local Vector Database (ChromaDB)                       │
│     1,601 page-aware chunks embedded with all-MiniLM-L6-v2             │
│                 (100% offline, private, and free)                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Grounded, Cited Response                          │
│     Authoritative CLSA answer with clickable sources & page citations  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Document Ingestion:** PyMuPDF extracts text page-by-page from 68 official Pega PDFs, tracking file names, sections, categories, and page numbers.
2. **Context-Enriched Chunking:** Text is split using `RecursiveCharacterTextSplitter` (950 characters with 180 overlap) and tagged with rich metadata.
3. **Local Vector Storage:** ChromaDB stores the vectors locally (`pega_rag_db`) using SentenceTransformers (`all-MiniLM-L6-v2`) — no sensitive data is sent over the network for embeddings.
4. **Autonomous Agent Loop:** The agent reasons step-by-step, calls tools, inspects evidence, and cross-checks challenge steps with architectural theory.
5. **Universal Fallback:** If a model without tool-calling capability is selected (e.g. `deepseek-r1`, `qwen`, `gemma`), the engine automatically switches to **Direct Grounded Retrieval** so questions never fail.

---

## 📚 What Knowledge is Included?

The knowledge base comes pre-indexed with **68 official documents (447+ pages)** across 5 master architectural pillars:

| Pillar | Focus Area | Master Modules | Challenges |
| :--- | :--- | :---: | :---: |
| **01_LSA_Mindset** | Architect principles, system governance, delivery leadership | 1 | 0 |
| **02_Planning_and_Design** | Enterprise Class Structure (ECS), data modeling, API design, SAML/OIDC authentication, ABAC security, BIX data extraction, PDC monitoring | 8 | 15 |
| **03_Constellation_Adoption** | Constellation architecture, views authoring, DX API, partial views, hybrid UI | 5 | 11 |
| **04_Infrastructure_and_Architecture** | Enterprise architecture, cloud deployment, Pega GenAI architecture, security hardening | 7 | 8 |
| **05_Implementation_and_Deployment** | DevOps, pipeline automation, deployment strategies | 7 | 6 |
| **Course Summary** | Complete curriculum catalog with page counts & descriptions | 1 DOCX | — |
| **TOTAL** | **Entire Pega CLSA Curriculum** | **29 Documents** | **40 Challenges** |

> 📄 A complete, printable summary catalog is provided in [`README_Course_Catalog.docx`](file:///README_Course_Catalog.docx).

---

## 🚀 Quick Start Guide (Run in 2 Minutes)

### 1. Prerequisites
* **Python 3.10 or higher** installed on your computer.
* **Git** installed on your computer.
* A **Free Groq API Key** (Get one in 30 seconds at [console.groq.com/keys](https://console.groq.com/keys)).

### 2. Clone the Repository
```bash
git clone https://github.com/DeveshEai/PegaRagAgent.git
cd PegaRagAgent
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key
Create a `.env` file in the project folder (or simply copy `.env.example`):
```bash
cp .env.example .env
```
Open `.env` and add your Groq key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Note: You can also paste your API key directly into the web app sidebar if you prefer not to create a `.env` file!)*

### 5. Launch the Web Assistant
**On Windows:** Double-click the one-click launcher:
```bat
LAUNCH_AI_ASSISTANT.bat
```

**Or from any terminal (Windows / Mac / Linux):**
```bash
streamlit run app.py
```

The app will automatically open in your default browser at **`http://localhost:8501`**!

---

## 🖥️ Web App Features Walkthrough

### 💬 1. Ask Architect (Interactive Q&A)
* **Pre-loaded architectural prompts:** Click one-click suggested questions to see the agent in action.
* **Autonomous Reasoning Trace:** Expand the thought process accordion to inspect which tools the agent called and what queries it executed.
* **Evidence & Page Citations:** Expand the retrieved sources to see the exact document, section, page number, and similarity score (e.g. `94.2%`) for each citation.

### 📤 2. Upload & Add More Files
* Upload any new PDF, Word document (`.docx`), or text file (`.txt`, `.md`).
* The system automatically parses pages, splits them into semantic chunks, generates vector embeddings, and updates the local ChromaDB index in real time.

### 📑 3. Document & Chunk Explorer
* Search the vector database by keyword.
* Filter by curriculum section (e.g., `03_Constellation_Adoption`).
* Inspect how text was chunked and verify metadata tags.

---

## 📁 Repository Structure

```
PegaRagAgent/
│
├── app.py                      # Main Streamlit web application (UI & Chat Interface)
├── rag_engine.py               # Core RAG engine, ChromaDB integration, PyMuPDF parser & Agent loop
├── requirements.txt            # Python package dependencies
├── .env.example                # Sample environment file template
├── .gitignore                  # Prevents accidental commits of API keys and temporary files
│
├── LAUNCH_AI_ASSISTANT.bat     # Windows one-click desktop launcher for the web assistant
├── RUN_DOWNLOADER.bat          # Windows one-click launcher for the curriculum scraper
├── pega_downloader.py          # Playwright scraper for official Pega modules
├── download_challenges.py      # Playwright scraper for 40 hands-on challenges
│
├── README.md                   # This documentation guide
├── README_Course_Catalog.docx  # Detailed MS Word catalog of all 68 course files
│
├── Lead System Architect/      # 68 official curriculum PDFs (Modules & Challenges)
│   ├── 01_LSA_Mindset/
│   ├── 02_Planning_and_Design/
│   ├── 03_Constellation_Adoption/
│   ├── 04_Infrastructure_and_Architecture/
│   └── 05_Implementation_and_Deployment/
│
└── pega_rag_db/                # Persistent ChromaDB vector database (1,601 pre-indexed chunks)
    ├── chroma.sqlite3
    └── indexed_manifest.json
```

---

## ⚙️ Model Selection & Compatibility

The app detects which models are enabled on your Groq key and tags them:
* **`🛠️ (Agent Tools)`** — Models that support multi-step tool calling (e.g., `llama-3.3-70b-versatile`). The agent autonomously plans searches, reads pages, and gathers evidence.
* **`⚡ (Direct Grounded)`** — Models that use direct context injection (e.g., `deepseek-r1-distill`, `qwen`, `gemma`). The engine automatically fetches the best chunks and injects them into the prompt without tool calls.

---

## 🔒 Privacy & Local Vector Storage

* **100% Local Embeddings:** All vector calculations are performed on your local CPU/GPU using `sentence-transformers/all-MiniLM-L6-v2`. Your course materials and uploaded documents are never sent to third-party embedding APIs.
* **Secure API Communication:** Only the final synthesized prompt and retrieved context excerpts are sent via encrypted HTTPS to Groq for ultra-fast generation.
* **Credentials Protected:** The `.env` file is excluded from Git commits via `.gitignore`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open an issue or submit a pull request if you want to add more challenge walkthroughs, enhance chunking strategies, or improve UI components.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
All Pega educational course material references belong to Pegasystems Inc.

