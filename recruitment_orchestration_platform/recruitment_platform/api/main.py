"""
=============================================================================
Recruitment Orchestration Platform — FastAPI Application
=============================================================================

This file defines the RESTful endpoints that act as the interface between
the frontend UI (or API clients) and the underlying LangGraph Orchestrator.

Key Responsibilities:
- Receives job descriptions (Text or PDF)
- Exposes conversational endpoints for pipeline progression
- Provides status polling for dashboard metrics
- Manages cross-origin resource sharing (CORS)

For more information, see ARCHITECTURE.md.
"""
import os
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config import settings
from core.state_store import pipeline_store
from models.schemas import (
    ChatResponse, PipelineState, PipelineStage, InterviewFeedback,
)
from agents.supervisor import RecruitmentSupervisor
from agents.feedback_agent import FeedbackAgent
from utils.file_parser import extract_text_from_upload

# Standalone feedback agent (supervisor no longer exposes _feedback_agent)
_feedback_agent = FeedbackAgent()

# ── FastAPI Application Setup ───────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Multi-agent AI-powered recruitment orchestration system. "
        "Automates JD intake, candidate sourcing, scoring, interview "
        "scheduling, feedback collection, and offer generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)

logger.info(f"🚀 Recruitment Platform Backend is starting on {settings.api_host}:{settings.api_port}")
logger.info(f"LangGraph Agent Framework: ACTIVE")

# Singleton supervisor
supervisor = RecruitmentSupervisor()

def get_supervisor() -> RecruitmentSupervisor:
    return supervisor

# ── Endpoints: System Health ───────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": settings.app_name}

# ──────────────────────────────────────────────────────────────
# Dashboard — All Pipelines
# ──────────────────────────────────────────────────────────────

@app.get("/pipelines", tags=["Dashboard"])
async def list_all_pipelines():
    """
    Returns a summary list of ALL active pipelines across all open roles.
    Supports the multi-role dashboard requirement.
    """
    all_pipelines = pipeline_store.list_all()
    results = []
    for p in all_pipelines:
        jd = p.job_description
        results.append({
            "pipeline_id": p.pipeline_id,
            "job_title": jd.title if jd else "Unknown Role",
            "department": jd.department if jd else "—",
            "location": jd.location if jd else "—",
            "current_stage": p.current_stage.value,
            "total_candidates": len(p.candidates),
            "scored_candidates": len(p.scores),
            "scheduled_candidates": len(p.schedules),
            "feedbacks_received": len(p.feedbacks),
            "offers_drafted": len(p.offers),
            "requires_human_approval": p.requires_human_approval,
            "is_completed": p.is_completed,
            "created_at": p.created_at.isoformat(),
            "updated_at": p.updated_at.isoformat(),
            "metadata": p.metadata,
        })
    # Sort by updated_at descending (most recently active first)
    results.sort(key=lambda x: x["updated_at"], reverse=True)
    return {"total": len(results), "pipelines": results}

@app.get("/pipelines/{pipeline_id}/status", tags=["Dashboard"])
async def get_pipeline_status(pipeline_id: str):
    """Get full status details for a specific pipeline."""
    pipeline = pipeline_store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found.")
    jd = pipeline.job_description
    return {
        "pipeline_id": pipeline.pipeline_id,
        "job_title": jd.title if jd else "Unknown Role",
        "department": jd.department if jd else "—",
        "current_stage": pipeline.current_stage.value,
        "stage_history": pipeline.stage_history,
        "candidates": [
            {"id": c.candidate_id, "name": c.name, "status": c.status.value}
            for c in pipeline.candidates
        ],
        "feedbacks": [
            {
                "candidate_id": fb.candidate_id,
                "interviewer_name": fb.interviewer_name,
                "sentiment": fb.sentiment.value if fb.sentiment else None
            }
            for fb in pipeline.feedbacks
        ],
        "scores": [
            {"candidate_id": s.candidate_id, "overall_score": s.overall_score}
            for s in pipeline.scores
        ],
        "requires_human_approval": pipeline.requires_human_approval,
        "human_approval_reason": pipeline.human_approval_reason,
        "metadata": pipeline.metadata,
    }

# ──────────────────────────────────────────────────────────────
# Stage 1 — JD Upload (PDF / Natural Language)
# ──────────────────────────────────────────────────────────────

@app.post("/jd/upload", tags=["Stage 1 — JD Intake"], response_model=ChatResponse)
async def upload_job_description(
    file: Optional[UploadFile] = File(None),
    natural_language: Optional[str] = Form(None),
    pipeline_id: Optional[str] = Form(None),
    sv: RecruitmentSupervisor = Depends(get_supervisor),
):
    """
    Upload a job description via:
    - **file**: PDF or TXT file upload
    - **natural_language**: Free-text description of the role (optional if file provided)
    
    One of `file` or `natural_language` is required.
    Returns the pipeline_id and stage.
    """
    if not file and not natural_language:
        raise HTTPException(
            status_code=422,
            detail="Provide either a file upload or a natural_language description.",
        )

    # Extract text from file if provided
    jd_text = ""
    if file:
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        file_bytes = await file.read()
        if len(file_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {settings.max_file_size_mb}MB limit.",
            )
        try:
            # pdf-extract is done here
            jd_text = extract_text_from_upload(file_bytes, file.filename)
            logger.info(f"[API] JD text extracted from {file.filename}")
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
    
    # Prepend or substitute with natural language text if provided
    if natural_language:
        if jd_text:
            jd_text = f"{natural_language}\n\nAdditional Context from File:\n{jd_text}"
        else:
            jd_text = natural_language

    try:
        # Initial routing via supervisor
        # According to the user request, the supervisor should handle routing automatically
        response = sv.process(user_message=jd_text, pipeline_id=pipeline_id)
        return response
    except Exception as e:
        logger.error(f"[API] JD processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────
# Human Approval endpoint
# ──────────────────────────────────────────────────────────────

@app.post("/pipelines/{pipeline_id}/approve", tags=["Human Approval"])
async def approve_pipeline_action(
    pipeline_id: str,
    decision: str = Form(...),
    human_input: Optional[str] = Form(None),
    sv: RecruitmentSupervisor = Depends(get_supervisor),
):
    """
    Submit a human approval decision for an escalated action.
    """
    if decision not in ("approve", "reject"):
        raise HTTPException(status_code=422, detail="Decision must be 'approve' or 'reject'.")

    pipeline = pipeline_store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found.")

    if decision == "approve":
        try:
            return sv.auto_advance(pipeline_id, human_input=human_input)
        except Exception as e:
            logger.error(f"[API] Post-approval advance error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "pipeline_id": pipeline_id,
        "decision": decision,
        "current_stage": pipeline.current_stage.value,
        "message": "Pipeline rejected by human.",
    }

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# Stage 3 — Interview Feedback
# ──────────────────────────────────────────────────────────────

@app.post("/pipelines/{pipeline_id}/feedback", tags=["Stage 3 — Interview Feedback"])
async def submit_interview_feedback(
    pipeline_id: str,
    candidate_id: str = Form(...),
    interviewer_name: str = Form(...),
    technical_rating: int = Form(...),
    communication_rating: int = Form(...),
    culture_fit_rating: int = Form(...),
    problem_solving_rating: int = Form(...),
    raw_feedback: str = Form(...),
    advance_pipeline: bool = Form(True),
    sv: RecruitmentSupervisor = Depends(get_supervisor),
):
    """
    Submit feedback after an interview.
    Runs Sentiment Analysis and advances pipeline to OFFER stage.
    """
    pipeline = pipeline_store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found.")

    feedback = InterviewFeedback(
        candidate_id=candidate_id,
        job_id=pipeline.job_id or "JOB-001",
        interviewer_name=interviewer_name,
        technical_rating=technical_rating,
        communication_rating=communication_rating,
        culture_fit_rating=culture_fit_rating,
        problem_solving_rating=problem_solving_rating,
        raw_feedback=raw_feedback,
    )
    pipeline.feedbacks.append(feedback)

    try:
        # Instead of manual agent run, resume the graph.
        # The graph will pick up the new feedback in its next node.
        pipeline_store.update(pipeline)
        
        logger.info(f"[API] Feedback submitted for candidate {candidate_id}. Advance pipeline: {advance_pipeline}")
        if advance_pipeline:
            response = sv.auto_advance(pipeline_id, human_input="feedback_submitted")
            return response
        else:
            from models.schemas import ChatResponse
            return ChatResponse(
                pipeline_id=pipeline_id,
                stage=pipeline.current_stage.value,
                response=f"✅ Feedback temporarily saved for **{candidate_id}**. You can add feedback for another candidate or proceed to analysis.",
                requires_human_approval=False
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ──────────────────────────────────────────────────────────────
# Chat / Manual Progression
# ──────────────────────────────────────────────────────────────

@app.post("/chat", tags=["Orchestration"], response_model=ChatResponse)
async def process_chat_message(
    pipeline_id: str = Form(...),
    message: str = Form(...),
    sv: RecruitmentSupervisor = Depends(get_supervisor),
):
    """
    Manually advance the pipeline or send a custom message to the supervisor.
    """
    pipeline = pipeline_store.get(pipeline_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found.")
        
    try:
        response = sv.process(user_message=message, pipeline_id=pipeline_id)
        return response
    except Exception as e:
        logger.error(f"[API] Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
