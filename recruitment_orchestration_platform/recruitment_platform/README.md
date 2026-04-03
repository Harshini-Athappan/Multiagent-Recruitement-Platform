# 🤖 Recruitment Orchestration Platform

A fully functional **multi-agent AI recruitment pipeline** built with LangChain, Groq LLMs, and FastAPI. It automates the end-to-end hiring workflow through a conversational interface.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
│  /jd/upload  /chat  /pipelines/*  /dashboard  /approve      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Recruitment Supervisor (Orchestrator)           │
│                                                              │
│  • Intent classification (LLM-assisted)                      │
│  • Stage-machine routing                                      │
│  • Human escalation management                               │
│  • Pipeline state coordination                               │
└──────┬──────┬──────┬──────┬──────┬──────┬──────────────────┘
       │      │      │      │      │      │
       ▼      ▼      ▼      ▼      ▼      ▼
   ┌──────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌──────┐
   │  JD  │ │Src │ │Scr │ │Sch │ │Fdb │ │Ofr │ │Eval  │
   │Agent │ │Agt │ │Agt │ │Agt │ │Agt │ │Agt │ │Agent │
   └──────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └──────┘
       │      │      │      │      │      │         │
       └──────┴──────┴──────┴──────┴──────┴─────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Pipeline State Store │
              │   (In-memory / Redis)  │
              └───────────────────────┘
```

## 🔄 Pipeline Flow

```
JD_INTAKE → CANDIDATE_SEARCH → SCORING → SCHEDULING
    → FEEDBACK → OFFER → EVALUATION → COMPLETED
```

| Stage | Agent | Responsibility |
|-------|-------|----------------|
| 1 | **JD Agent** | Parse PDF/text/NL job descriptions into structured JSON |
| 2 | **Candidate Search Agent** | Source candidates with ≥40% skill match |
| 3 | **Scoring Agent** | Score candidates 0-100 using weighted rubric |
| 4 | **Scheduling Agent** | Schedule interviews with conflict detection |
| 5 | **Feedback Agent** | Collect feedback, sentiment analysis, hire/no-hire |
| 6 | **Offer Agent** | Draft personalised offer letters with salary validation |
| 7 | **Evaluation Agent** | Pipeline audit, metrics, closure report |

---

## ⚙️ Setup Instructions

**Prerequisites**: Python 3.11+

1. **Clone and navigate**:
   ```bash
   git clone https://github.com/your-org/recruitment-platform.git
   cd recruitment-platform
   ```
2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   .\venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment**:
   ```bash
   # Create .env file and add your GROQ_API_KEY
   # Ensure API_PORT=8016 is set
   ```
5. **Run the services (in separate terminals)**:
   
   *Terminal 1 (Backend)*:
   ```bash
   python run.py
   # API lives at http://localhost:8016
   ```
   
   *Terminal 2 (Frontend)*:
   ```bash
   streamlit run streamlit_app.py
   # UI lives at http://localhost:8501
   ```

---

## 📥 Required Inputs per Stage

To successfully navigate the autonomous pipeline, the following inputs are required at each step:

| Stage | Input Type | Description |
|-------|------------|-------------|
| **1. JD Intake** | `Text` or `PDF` | Paste the Job Description or upload a file to start. |
| **2. Search** | `Automatic` | AI searches the internal database for matches. |
| **3. Scoring** | `Automatic` | AI scores candidates. Review the rankings. |
| **4. Scheduling** | `Approval` | Click "Confirm & Send Invites" to book the proposed slots. |
| **5. Feedback** | `Form` | Enter Interviewer Name, Ratings (1-10), and raw review text for each candidate. |
| **6. Offer** | `Decision` | Review the Sentiment and Draft. Click "Send Final Offer" or provide text adjustments (e.g. "Add $10k bonus"). |
| **7. Eval** | `Review` | AI generates the closure report and finishes the pipeline. |

---

## 🚀 Interactive Usage & Testing

### Streamlit UI (Primary Interface)
The easiest way to use the platform is through the Streamlit dashboard:
1. Open **http://localhost:8501**
2. Use the **"➕ Start New Recruitment Role"** button in the sidebar to begin a pipeline.
3. Paste a job description in the text box and follow the autonomous conversational flow.
4. Interact with the LLM directly through the chat input at the bottom to progress/approve stages (e.g., type "approve" when prompted).
5. Switch between multiple active roles using the sidebar dashboard.

### FastAPI Swagger (API Testing)
You can test the headless endpoints directly via Swagger:
1. Open **http://localhost:8016/docs**
2. Use `POST /jd/text` to submit a Job Description. Copy the returned `pipeline_id`.
3. Use `POST /chat` with `{"pipeline_id": "...", "message": "proceed"}` to advance the multi-agent pipeline.
4. View pipeline state and metrics using the `GET /pipelines` and `GET /pipelines/{id}/status` endpoints.

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jd/upload` | Upload PDF/TXT job description |
| `POST` | `/jd/text` | Submit JD as text |
| `POST` | `/chat` | Conversational pipeline advancement |
| `GET`  | `/dashboard` | Aggregated metrics for all pipelines |
| `GET`  | `/pipelines` | List all pipelines |
| `GET`  | `/pipelines/{id}` | Get full pipeline state |

### Stage-Specific Endpoints

| Method | Endpoint | Stage |
|--------|----------|-------|
| `POST` | `/pipelines/{id}/advance` | Auto-advance to next stage |
| `POST` | `/pipelines/{id}/search-candidates` | Stage 2 |
| `POST` | `/pipelines/{id}/score-candidates` | Stage 3 |
| `POST` | `/pipelines/{id}/schedule-interviews` | Stage 4 |
| `POST` | `/pipelines/{id}/collect-feedback` | Stage 5 |
| `POST` | `/pipelines/{id}/generate-offers` | Stage 6 |
| `POST` | `/pipelines/{id}/evaluate` | Stage 7 |

### Data Endpoints

| Method | Endpoint | Returns |
|--------|----------|---------|
| `GET`  | `/pipelines/{id}/candidates` | Candidates + scores |
| `GET`  | `/pipelines/{id}/schedules` | Interview schedules |
| `GET`  | `/pipelines/{id}/feedback` | Feedback + sentiment |
| `GET`  | `/pipelines/{id}/offers` | Offer summaries |
| `GET`  | `/pipelines/{id}/offers/{offer_id}/letter` | Full offer text |
| `GET`  | `/pipelines/{id}/history` | Conversation history |
| `POST` | `/pipelines/{id}/approve?decision=approve` | Human approval |

---

## 🤖 Agent Prompts

All agent system prompts are stored in `prompts/agent_prompts.json`.
You can tune them without touching code.

---

## 📁 Project Structure

```
recruitment_platform/
├── agents/
│   ├── supervisor.py           # Orchestrator (LangChain routing)
│   ├── jd_agent.py             # Stage 1: JD Intake & Parsing
│   ├── candidate_search_agent.py  # Stage 2: Candidate Sourcing
│   ├── scoring_agent.py        # Stage 3: Candidate Scoring
│   ├── scheduling_agent.py     # Stage 4: Interview Scheduling
│   ├── feedback_agent.py       # Stage 5: Feedback & Sentiment
│   ├── offer_agent.py          # Stage 6: Offer Letter Drafting
│   └── evaluation_agent.py     # Stage 7: Pipeline Closure
├── api/
│   └── main.py                 # FastAPI app + all endpoints
├── core/
│   ├── config.py               # Settings (pydantic-settings)
│   ├── llm.py                  # Groq LLM factory
│   └── state_store.py          # Thread-safe pipeline store
├── models/
│   └── schemas.py              # All Pydantic models
├── prompts/
│   └── agent_prompts.json      # All agent system prompts
├── tests/
│   └── test_agents.py          # Pytest test suite (40+ tests)
├── utils/
│   ├── file_parser.py          # PDF/TXT extraction
│   ├── mock_data.py            # Candidate database
│   └── prompt_loader.py        # JSON prompt loader
├── .env.example                # Environment template
├── requirements.txt
├── run.py                      # Entry point
└── README.md
```

---

## 🔑 Key Design Decisions

### LangChain for Orchestration
- `ChatGroq` via `langchain-groq` for all LLM calls
- `SystemMessage` / `HumanMessage` for structured prompting
- Supervisor uses LLM for intent classification, then deterministic routing
- Each agent maintains its own conversation context

### Groq LLM (llama-3.3-70b-versatile)
- Fast inference (< 2s per agent call typical)
- Free tier sufficient for development
- Model configurable via `.env`

### State Management
- `PipelineState` Pydantic model as single source of truth
- Thread-safe in-memory store (`PipelineStore`)
- Full conversation history per pipeline
- Stage history tracking for audit

### Scoring Rubric (deterministic)
```
Overall Score = 0.40 × Skill Match
              + 0.30 × Experience Match
              + 0.15 × Education Match
              + 0.15 × Preferred Skills
```

### Conflict Detection
- 15-minute buffer between consecutive interviews
- Business hours: Mon–Fri, 09:00–18:00
- Auto-propose 3 alternatives on conflict

---

## 🔮 Production Roadmap

- [ ] PostgreSQL / Redis backend for state persistence
- [ ] OAuth2 authentication
- [ ] ATS integration (Greenhouse, Lever, Workday)
- [ ] Email notifications (SendGrid/SES)
- [ ] Calendar integration (Google Calendar API)
- [ ] React dashboard frontend
- [ ] LangGraph for complex multi-step workflows
- [ ] Async job queue (Celery + Redis)

---

## 📄 License
MIT License — see LICENSE file.
