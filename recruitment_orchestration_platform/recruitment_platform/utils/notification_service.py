"""
Notification Service
Fires mock stakeholder notifications at each pipeline stage transition.
In production this would integrate with email/Slack/Teams APIs.
Notifications are stored in the pipeline metadata for UI display.
"""
from datetime import datetime
from loguru import logger
from models.schemas import PipelineState


STAGE_STAKEHOLDERS = {
    "candidate_search": {"role": "Hiring Manager", "event": "Job Description parsed & candidate search started"},
    "scoring":          {"role": "Recruiter",       "event": "Candidates sourced — scoring in progress"},
    "scheduling":       {"role": "Recruiter",       "event": "Candidates scored — shortlist ready for review"},
    "feedback":         {"role": "Interviewer",     "event": "Interview slots confirmed — awaiting feedback"},
    "offer":            {"role": "HR Manager",      "event": "Interview feedback analysed — offer stage reached"},
    "evaluation":       {"role": "Hiring Manager",  "event": "Offer sent — pipeline closing with evaluation"},
    "completed":        {"role": "All Stakeholders","event": "Recruitment pipeline completed successfully ✅"},
}


def notify_stage_transition(pipeline: PipelineState, new_stage: str) -> str:
    """
    Fires a stakeholder notification when the pipeline advances to a new stage.
    Returns the notification message string.
    Stores notification log in pipeline.metadata["notifications"].
    """
    stage_info = STAGE_STAKEHOLDERS.get(new_stage)
    if not stage_info:
        return ""

    jd_title = pipeline.job_description.title if pipeline.job_description else "Unknown Role"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    notification = (
        f"📬 **[{timestamp}]** → **{stage_info['role']}** notified\n"
        f"   Role: **{jd_title}** | Pipeline `{pipeline.pipeline_id[:8]}…`\n"
        f"   Event: {stage_info['event']}"
    )

    logger.info(f"[NotificationService] {stage_info['role']} notified for pipeline {pipeline.pipeline_id[:8]} → stage: {new_stage}")

    # Store in pipeline metadata for UI display
    md = pipeline.metadata
    if "notifications" not in md:
        md["notifications"] = []
    md["notifications"].append({
        "timestamp": timestamp,
        "stage": new_stage,
        "stakeholder": stage_info["role"],
        "event": stage_info["event"],
        "role": jd_title,
    })

    return notification


def get_notifications(pipeline: PipelineState):
    """Return all notifications logged for this pipeline."""
    return pipeline.metadata.get("notifications", [])
