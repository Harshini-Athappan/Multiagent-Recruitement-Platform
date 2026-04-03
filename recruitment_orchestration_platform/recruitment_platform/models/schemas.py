from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class PipelineStage(str, Enum):
    JD_INTAKE        = "jd_intake"
    CANDIDATE_SEARCH = "candidate_search"
    SCORING          = "scoring"
    SCHEDULING       = "scheduling"
    FEEDBACK         = "feedback"
    OFFER            = "offer"
    EVALUATION       = "evaluation"
    COMPLETED        = "completed"
    HUMAN_ESCALATION = "human_escalation"


class CandidateStatus(str, Enum):
    SOURCED      = "sourced"
    SCREENED     = "screened"
    SCHEDULED    = "scheduled"
    INTERVIEWED  = "interviewed"
    OFFER_SENT   = "offer_sent"
    ACCEPTED     = "accepted"
    REJECTED     = "rejected"


class FeedbackSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"


class InterviewStatus(str, Enum):
    SCHEDULED    = "scheduled"
    RESCHEDULED  = "rescheduled"
    COMPLETED    = "completed"
    CANCELLED    = "cancelled"
    CONFLICT     = "conflict"


# ─────────────────────────────────────────────
# Job Description Models
# ─────────────────────────────────────────────

class JobDescription(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    department: str
    location: str
    employment_type: str = "Full-time"
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    experience_years: int = 0
    education_requirement: str = ""
    responsibilities: List[str] = []
    salary_range: Optional[str] = None
    raw_text: str = ""
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class JDUploadRequest(BaseModel):
    natural_language_description: Optional[str] = None


# ─────────────────────────────────────────────
# Candidate Models
# ─────────────────────────────────────────────

class Candidate(BaseModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    phone: Optional[str] = None
    current_role: str = ""
    years_of_experience: int = 0
    skills: List[str] = []
    education: str = ""
    location: str = ""
    linkedin_url: Optional[str] = None
    resume_summary: str = ""
    source: str = "database"
    status: CandidateStatus = CandidateStatus.SOURCED


class CandidateScore(BaseModel):
    candidate_id: str
    job_id: str
    skill_match_score: float = Field(ge=0, le=100)
    experience_score: float = Field(ge=0, le=100)
    education_score: float = Field(ge=0, le=100)
    overall_score: float = Field(ge=0, le=100)
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendation: str = ""
    scored_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Interview / Scheduling Models
# ─────────────────────────────────────────────

class TimeSlot(BaseModel):
    start: datetime
    end: datetime
    interviewer_email: str
    interviewer_name: str


class InterviewSchedule(BaseModel):
    schedule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str
    job_id: str
    interviewer_name: str
    interviewer_email: str
    scheduled_at: datetime
    duration_minutes: int = 60
    meeting_link: Optional[str] = None
    status: InterviewStatus = InterviewStatus.SCHEDULED
    conflict_reason: Optional[str] = None
    rescheduled_from: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Feedback Models
# ─────────────────────────────────────────────

class InterviewFeedback(BaseModel):
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str
    job_id: str
    interviewer_name: str
    technical_rating: int = Field(ge=1, le=10)
    communication_rating: int = Field(ge=1, le=10)
    culture_fit_rating: int = Field(ge=1, le=10)
    problem_solving_rating: int = Field(ge=1, le=10)
    raw_feedback: str
    sentiment: Optional[FeedbackSentiment] = None
    sentiment_score: Optional[float] = None
    aggregated_score: Optional[float] = None
    hire_recommendation: Optional[bool] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Offer Models
# ─────────────────────────────────────────────

class OfferLetter(BaseModel):
    offer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_id: str
    job_id: str
    candidate_name: str
    job_title: str
    department: str
    salary_offered: Optional[str] = "Competitive / To be discussed"
    start_date: Optional[str] = "TBD"
    benefits: List[str] = []
    offer_letter_text: str = ""
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    requires_approval: bool = True
    approved: Optional[bool] = None


# ─────────────────────────────────────────────
# Pipeline State — the single source of truth
# ─────────────────────────────────────────────

class PipelineState(BaseModel):
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: Optional[str] = None
    current_stage: PipelineStage = PipelineStage.JD_INTAKE
    job_description: Optional[JobDescription] = None
    candidates: List[Candidate] = []
    scores: List[CandidateScore] = []
    schedules: List[InterviewSchedule] = []
    feedbacks: List[InterviewFeedback] = []
    offers: List[OfferLetter] = []
    messages: List[Dict[str, Any]] = []      # conversation history
    stage_history: List[str] = []
    requires_human_approval: bool = False
    human_approval_reason: Optional[str] = None
    is_completed: bool = False
    metadata: Dict[str, Any] = Field(default_factory=lambda: {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "llm_calls": 0,
        "notifications": [],
    })
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, role: str, content: str, agent: str = "system"):
        self.messages.append({
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.updated_at = datetime.utcnow()

    def advance_stage(self, new_stage: PipelineStage):
        self.stage_history.append(self.current_stage.value)
        self.current_stage = new_stage
        self.updated_at = datetime.utcnow()


# ─────────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    pipeline_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    pipeline_id: str
    stage: str
    response: str
    thought: Optional[str] = None
    requires_human_approval: bool = False
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineSummary(BaseModel):
    pipeline_id: str
    job_title: Optional[str]
    current_stage: str
    total_candidates: int
    shortlisted_candidates: int
    interviews_scheduled: int
    offers_drafted: int
    created_at: datetime
    updated_at: datetime
