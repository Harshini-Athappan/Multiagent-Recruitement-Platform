# Architecture Diagram — Recruitment Orchestration Platform

```text
╔══════════════════════════════════════════════════════════════════════╗
║                     RECRUITMENT ORCHESTRATION PLATFORM               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │                   FastAPI REST API Layer                      │    ║
║  │                                                               │    ║
║  │  POST /jd/upload   POST /jd/text   POST /chat                │    ║
║  │  GET  /pipelines   POST /pipelines/{id}/advance              │    ║
║  │  GET  /dashboard   POST /pipelines/{id}/approve              │    ║
║  └──────────────────────────┬────────────────────────────────── ┘    ║
║                             │ HTTP Request                            ║
║                             ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │           LANGGRAPH SUPERVISOR (Orchestrator)                 │    ║
║  │                                                               │    ║
║  │   Input ──► StateGraph Execution                             │    ║
║  │                        │                                      │    ║
║  │                        ▼                                      │    ║
║  │              Conditional Routing & Idempotency                │    ║
║  │                                                               │    ║
║  └──────┬──────┬──────┬──────┬──────┬──────┬──────────────────┘    ║
║         │      │      │      │      │      │                         ║
║         ▼      ▼      ▼      ▼      ▼      ▼                         ║
║  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌──────────────────┐   ║
║  │ JD │ │Src │ │Scr │ │Sch │ │Fdb │ │Ofr │ │ Eval Agent       │   ║
║  │Agt │ │Agt │ │Agt │ │Agt │ │Agt │ │Agt │ │                  │   ║
║  │    │ │    │ │    │ │    │ │    │ │    │ │ Final Report     │   ║
║  │NLP │ │LLM │ │Rub │ │Cnfl│ │Sent│ │Appv│ │ Funnel Metrics   │   ║
║  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └──────────────────┘   ║
║   Stg1   Stg2   Stg3   Stg4   Stg5   Stg6         Stg7             ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │                    Persistent Services                        │    ║
║  │                                                               │    ║
║  │  [ SQLite State Store ]      [ Notification Service ]         │    ║
║  │  State per Pipeline ID       Mock webhooks/emails per stage   │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │                 Groq LLM (llama-3.1-8b-instant)              │    ║
║  │                via Custom Rate Limit / Retry                 │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════╝

PIPELINE STAGE FLOW (LangGraph):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 START ──► JD_INTAKE ──► CANDIDATE_SEARCH ──► SCORING
                                                 │
                                                 ▼
                                        [ HUMAN APPROVAL ]
                                                 │
                                                 ▼
                                            SCHEDULING
                                                 │
                                                 ▼
                                        [ HUMAN APPROVAL ]
                                                 │
                                                 ▼
                                             FEEDBACK
                                                 │
                                                 ▼
   END ◄── EVALUATION ◄── OFFER ◄── [ HUMAN APPROVAL ]
                                                 │
                                                 ▼
                                        [ HUMAN APPROVAL ]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT RESPONSIBILITIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Stage 1 │ JD Agent          │ Parses natural language or PDFs into JSON JDs
 Stage 2 │ Search Agent      │ Synthesizes matching candidates (LLM → DDG fallback)
 Stage 3 │ Scoring Agent     │ Rates candidates 0-100 on a weighted rubric
 Stage 4 │ Scheduling Agent  │ Proposes slots & prevents interviewer conflict
 Stage 5 │ Feedback Agent    │ Analyzes interviewer sentiment & recommendations
 Stage 6 │ Offer Agent       │ Drafts dynamic offers based on final negotiations
 Stage 7 │ Evaluation Agent  │ Summarizes entire conversational chain output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY TECHNICAL INFRASTRUCTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 • Multiple Open Roles: Support for >N active pipelines in SQLite at once
 • Idempotency Rules: Re-executing a graph won't repeat passed stages
 • LLM Exponential Backoff: Built-in Groq 429 handler + Semaphore locks
 • Notification Hooks: Publishes transition messages to `pipeline.metadata`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
