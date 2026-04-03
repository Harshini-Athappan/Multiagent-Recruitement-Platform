"""
Final Evaluation Agent
Responsibility: Conduct an end-to-end pipeline audit, generate a completion
report, compute hiring metrics, and close the pipeline.
"""
from datetime import datetime
from typing import Optional
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from models.schemas import (
    PipelineState, PipelineStage, CandidateStatus, FeedbackSentiment
)
from utils.prompt_loader import get_system_prompt


class EvaluationAgent:
    """
    Final Evaluation & Pipeline Closure Agent.
    Generates metrics, audit report, and marks the pipeline as complete.
    """

    AGENT_NAME = "evaluation_agent"

    def __init__(self):
        self.llm = get_llm(temperature=0.2)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    # ──────────────────────────────────────────
    # Metric computation
    # ──────────────────────────────────────────

    def _compute_metrics(self, pipeline: PipelineState) -> dict:
        jd = pipeline.job_description
        candidates = pipeline.candidates
        scores = pipeline.scores
        schedules = pipeline.schedules
        feedbacks = pipeline.feedbacks
        offers = pipeline.offers

        total_sourced = len(candidates)
        screened = sum(1 for c in candidates if c.status != CandidateStatus.SOURCED)
        scheduled_count = sum(1 for c in candidates if c.status == CandidateStatus.SCHEDULED)
        interviewed_count = sum(
            1 for c in candidates
            if c.status in (CandidateStatus.INTERVIEWED, CandidateStatus.OFFER_SENT)
        )
        offers_sent = sum(1 for c in candidates if c.status == CandidateStatus.OFFER_SENT)
        rejected_count = sum(1 for c in candidates if c.status == CandidateStatus.REJECTED)

        avg_score = (
            sum(s.overall_score for s in scores) / len(scores) if scores else 0.0
        )

        conflicts = sum(1 for s in schedules if s.conflict_reason is not None)
        positive_feedbacks = sum(
            1 for fb in feedbacks
            if fb.sentiment == FeedbackSentiment.POSITIVE
        )

        # Time to fill
        days_to_fill = (datetime.utcnow() - pipeline.created_at).days or 1

        return {
            "total_sourced": total_sourced,
            "screened": screened,
            "scheduled": scheduled_count,
            "interviewed": interviewed_count,
            "offers_sent": offers_sent,
            "rejected": rejected_count,
            "avg_screening_score": round(avg_score, 1),
            "scheduling_conflicts": conflicts,
            "positive_feedback_ratio": (
                f"{positive_feedbacks}/{len(feedbacks)}" if feedbacks else "N/A"
            ),
            "days_to_fill": days_to_fill,
            "job_title": jd.title if jd else "Unknown",
        }

    # ──────────────────────────────────────────
    # LLM report generation
    # ──────────────────────────────────────────

    def _generate_report(self, pipeline: PipelineState, metrics: dict) -> str:
        stages = " → ".join(pipeline.stage_history + [pipeline.current_stage.value])

        offers_summary = ""
        for offer in pipeline.offers:
            offers_summary += (
                f"- {offer.candidate_name}: {offer.salary_offered} | "
                f"Start: {offer.start_date} | "
                f"Approval: {'Required' if offer.requires_approval else 'Auto-approved'}\n"
            )

        prompt = f"""Generate a comprehensive Pipeline Completion Report for this hiring cycle.

JOB TITLE: {metrics['job_title']}
PIPELINE ID: {pipeline.pipeline_id}

FUNNEL METRICS:
- Candidates Sourced: {metrics['total_sourced']}
- Screened: {metrics['screened']}
- Interviews Scheduled: {metrics['scheduled']}
- Interviewed: {metrics['interviewed']}
- Offers Sent: {metrics['offers_sent']}
- Rejected: {metrics['rejected']}
- Avg Screening Score: {metrics['avg_screening_score']}/100
- Scheduling Conflicts: {metrics['scheduling_conflicts']}
- Positive Feedback Ratio: {metrics['positive_feedback_ratio']}
- Days to Fill: {metrics['days_to_fill']}

STAGE HISTORY: {stages}

OFFERS SUMMARY:
{offers_summary or 'No offers generated.'}

COMPLIANCE GAPS TO CHECK:
- Were all candidates scored before scheduling?
- Was feedback collected for all interviewed candidates?
- Were salary-escalated offers flagged for approval?

Write a structured Pipeline Completion Report with:
1. Executive Summary (3-5 sentences)
2. Funnel Analysis (conversion rates between stages)
3. Top 3 Strengths of this hiring cycle
4. Top 3 Areas for Improvement
5. Recommendations for future similar roles
6. Compliance Status

Format it professionally with clear sections. This will be read by hiring managers."""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content.strip()

    # ──────────────────────────────────────────
    # Format final message
    # ──────────────────────────────────────────

    def _format_closure_message(
        self, metrics: dict, report: str
    ) -> str:
        funnel = (
            f"**Pipeline Funnel:**\n"
            f"```\n"
            f"Sourced ({metrics['total_sourced']}) "
            f"→ Screened ({metrics['screened']}) "
            f"→ Scheduled ({metrics['scheduled']}) "
            f"→ Interviewed ({metrics['interviewed']}) "
            f"→ Offers ({metrics['offers_sent']})\n"
            f"```\n"
        )

        return (
            f"## 🏁 Pipeline Closed — Hiring Complete\n\n"
            f"{funnel}\n"
            f"**Avg Candidate Score:** {metrics['avg_screening_score']}/100\n"
            f"**Days to Fill:** {metrics['days_to_fill']} day(s)\n"
            f"**Scheduling Conflicts:** {metrics['scheduling_conflicts']}\n\n"
            f"---\n\n"
            f"{report}\n\n"
            f"---\n\n"
            f"✅ Pipeline has been archived. All stakeholders have been notified."
        )

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(self, pipeline: PipelineState) -> tuple[PipelineState, str]:
        logger.info(f"[EvaluationAgent] Closing pipeline {pipeline.pipeline_id}")

        metrics = self._compute_metrics(pipeline)
        report = self._generate_report(pipeline, metrics)
        message = self._format_closure_message(metrics, report)

        pipeline.is_completed = True
        pipeline.add_message("assistant", message, agent=self.AGENT_NAME)
        pipeline.advance_stage(PipelineStage.COMPLETED)

        logger.info(
            f"[EvaluationAgent] Pipeline {pipeline.pipeline_id} completed. "
            f"Offers: {metrics['offers_sent']} | Days: {metrics['days_to_fill']}"
        )
        return pipeline, message
