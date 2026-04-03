"""
Scheduling Agent
Responsibility: Propose available interview slots, get human confirmation,
and "send" emails to candidates with meeting details.
Supports custom preferred times provided by the human.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import re
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from models.schemas import (
    Candidate, InterviewSchedule, InterviewStatus,
    PipelineState, PipelineStage, CandidateStatus,
)
from utils.prompt_loader import get_system_prompt

# Default interviewers pool (mock)
DEFAULT_INTERVIEWERS = [
    {"name": "Ramesh Kumar", "email": "ramesh.kumar@company.com"},
    {"name": "Meera Pillai", "email": "meera.pillai@company.com"},
    {"name": "Suresh Nair",  "email": "suresh.nair@company.com"},
]

INTERVIEW_DURATION = 60

class SchedulingAgent:
    """
    Interview Scheduling Agent.
    Handles slot proposing, human confirmation, and email notifications.
    """

    AGENT_NAME = "scheduling_agent"

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    def _get_available_slots(self, count: int) -> List[datetime]:
        """Propose slots starting from next Monday."""
        base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        days_ahead = (7 - base_time.weekday()) % 7 or 7
        base_time += timedelta(days=days_ahead)
        
        slots = []
        for i in range(count):
            slots.append(base_time + timedelta(hours=i * 2))
        return slots

    def _parse_human_times(self, human_input: str, pipeline: PipelineState) -> List[datetime]:
        """Use LLM to extract dates from human natural language."""
        from utils.prompt_loader import get_agent_prompt_template
        template = get_agent_prompt_template(self.AGENT_NAME, "date_parser_prompt")
        prompt = template.format(human_input=human_input, current_year=datetime.now().year)
        
        try:
            messages = [SystemMessage(content="You are a date extractor."), HumanMessage(content=prompt)]
            
            from utils.cost_tracker import update_cost
            from utils.llm_retry import llm_invoke_with_retry
            res = llm_invoke_with_retry(self.llm, messages)
            update_cost(pipeline, res.response_metadata, getattr(self.llm, "model_name", "llama-3.1-8b-instant"))
            
            import json
            
            content = res.content.strip()
            
            # Remove <thought> tags before searching for JSON array
            content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()

            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            else:
                raw = content

            dates_str = json.loads(raw)
            return [datetime.fromisoformat(d) for d in dates_str]
        except Exception as e:
            logger.error(f"[SchedulingAgent] Failed to parse dates: {e}")
            return []

    def _detect_conflicts(self, schedules: list) -> dict:
        """
        Detect time-slot conflicts: same interviewer booked at overlapping times.
        Returns a dict of {schedule_id: conflict_reason}.
        """
        from models.schemas import InterviewStatus
        conflicts = {}
        for i, s1 in enumerate(schedules):
            for j, s2 in enumerate(schedules):
                if i >= j:
                    continue
                same_interviewer = s1.interviewer_email == s2.interviewer_email
                time_overlap = abs((s1.scheduled_at - s2.scheduled_at).total_seconds()) < 3600  # within 1 hour
                if same_interviewer and time_overlap:
                    reason = f"Conflict with {s2.candidate_id[:8]} at {s2.scheduled_at.strftime('%H:%M')}"
                    conflicts[s1.schedule_id] = reason
                    if s1.schedule_id not in conflicts:
                        conflicts[s2.schedule_id] = f"Conflict with {s1.candidate_id[:8]} at {s1.scheduled_at.strftime('%H:%M')}"
        return conflicts

    def _format_slot_proposal(self, slots: List[datetime], candidates: List[Candidate], conflicts: dict = None) -> str:
        lines = ["**📅 Proposed Interview Slots**\n"]
        lines.append("Please confirm if these slots work for the following candidates:\n")
        
        for i, (slot, cand) in enumerate(zip(slots, candidates)):
            lines.append(f"{i+1}. **{cand.name}**: {slot.strftime('%A, %B %d at %H:%M')}")
        
        if conflicts:
            lines.append("\n⚠️ **Scheduling Conflicts Detected** — the following slots overlap for the same interviewer:")
            for sid, reason in conflicts.items():
                lines.append(f"   - Schedule `{sid[:8]}…`: {reason}")
            lines.append("\n💡 To resolve: provide alternative times in the `human_input` field.")
        else:
            lines.append("\n✅ No scheduling conflicts detected.")
            lines.append("\n💡 **Pro Tip**: If you don't like these slots, provide your preferred times in the `human_input` field.")
        
        lines.append("\n⚠️ **Action Required**: Please submit with decision='approve' to confirm.")
        return "\n".join(lines)

    def _send_mock_email(self, schedule: InterviewSchedule, candidate_name: str):
        """Simulate sending an email."""
        logger.info(f"[Email] Sending invitation to {candidate_name} ({schedule.candidate_id})")
        logger.info(f"[Email] Subject: Interview Invitation - {schedule.job_id}")
        logger.info(f"[Email] Body: Hi {candidate_name}, your interview is scheduled at {schedule.scheduled_at}. Link: {schedule.meeting_link}")

    def run(self, pipeline: PipelineState, human_input: Optional[str] = None) -> tuple[PipelineState, str]:
        # If we are here and not approved, we need to propose slots and wait for human
        if pipeline.current_stage == PipelineStage.SCORING:
            pipeline.advance_stage(PipelineStage.SCHEDULING)

        eligible = [c for c in pipeline.candidates if c.status == CandidateStatus.SCREENED]
        if not eligible:
            return pipeline, "No candidates eligible for scheduling."

        # Case: Human provided custom times (Preferred times)
        if human_input and len(human_input.strip()) > 5:
            custom_slots = self._parse_human_times(human_input, pipeline)
            if custom_slots:
                pipeline.schedules = [] # Clear old ones
                for slot, cand in zip(custom_slots[:len(eligible)], eligible):
                    interviewer = DEFAULT_INTERVIEWERS[0]
                    schedule = InterviewSchedule(
                        candidate_id=cand.candidate_id,
                        job_id=pipeline.job_id or "JOB-001",
                        interviewer_name=interviewer["name"],
                        interviewer_email=interviewer["email"],
                        scheduled_at=slot,
                        meeting_link=f"https://meet.recruit.ai/int-{cand.candidate_id[:8]}",
                        status=InterviewStatus.SCHEDULED
                    )
                    pipeline.schedules.append(schedule)
                
                conflicts = self._detect_conflicts(pipeline.schedules)
                msg = "🔄 **Slots updated based on your preferred times.**\n\n" + self._format_slot_proposal(custom_slots, eligible, conflicts)
                pipeline.add_message("assistant", msg, agent=self.AGENT_NAME)
                pipeline.requires_human_approval = True
                return pipeline, msg

        # Case 1: Proposing initial slots
        if not pipeline.schedules:
            slots = self._get_available_slots(len(eligible))
            for slot, cand in zip(slots, eligible):
                interviewer = DEFAULT_INTERVIEWERS[0]
                schedule = InterviewSchedule(
                    candidate_id=cand.candidate_id,
                    job_id=pipeline.job_id or "JOB-001",
                    interviewer_name=interviewer["name"],
                    interviewer_email=interviewer["email"],
                    scheduled_at=slot,
                    meeting_link=f"https://meet.recruit.ai/int-{cand.candidate_id[:8]}",
                    status=InterviewStatus.SCHEDULED
                )
                pipeline.schedules.append(schedule)

            conflicts = self._detect_conflicts(pipeline.schedules)
            proposal_msg = self._format_slot_proposal(slots, eligible, conflicts)
            pipeline.requires_human_approval = True
            pipeline.human_approval_reason = "Confirm interview slots and trigger candidate emails."
            pipeline.add_message("assistant", proposal_msg, agent=self.AGENT_NAME)
            return pipeline, proposal_msg

        # Case 2: Slots confirmed (After human approval)
        confirm_msg = "✅ **Slots confirmed. Sending invitations to candidates...**\n\n"
        for schedule in pipeline.schedules:
            cand = next((c for c in pipeline.candidates if c.candidate_id == schedule.candidate_id), None)
            if cand:
                self._send_mock_email(schedule, cand.name)
                cand.status = CandidateStatus.SCHEDULED
                confirm_msg += f"- 📧 Invited **{cand.name}** (`{cand.candidate_id}`) for {schedule.scheduled_at.strftime('%m/%d %H:%M')} — **Interviewer**: {schedule.interviewer_name}\n"

        confirm_msg += "\n💡 **Next Step**: When you fill out the Feedback Form, use the **Candidate ID** and **Interviewer Name** from the list above."
        confirm_msg += "\n🔄 Moving to **Feedback Stage**."
        pipeline.advance_stage(PipelineStage.FEEDBACK)
        pipeline.add_message("assistant", confirm_msg, agent=self.AGENT_NAME)
        return pipeline, confirm_msg
