"""
Offer Agent
Responsibility: Draft offer letters, validate necessary details,
handle human confirmation and specific adjustments (preferred items).
"""
from datetime import datetime, timedelta
from typing import List, Optional
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from models.schemas import (
    Candidate, InterviewFeedback, OfferLetter,
    PipelineState, PipelineStage, CandidateStatus, FeedbackSentiment,
)
from utils.prompt_loader import get_system_prompt, get_agent_prompt_template
from utils.cost_tracker import update_cost
from utils.llm_retry import llm_invoke_with_retry

class OfferAgent:
    """
    Offer Letter Drafting Agent.
    Requires human confirmation of the draft and allows adding "preferred items" 
    before the final send.
    """

    AGENT_NAME = "offer_agent"

    def __init__(self):
        self.llm = get_llm(temperature=0.3)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    def _draft_offer_letter(self, candidate: Candidate, jd, human_input: Optional[str] = None, pipeline: PipelineState = None) -> str:
        template = get_agent_prompt_template(self.AGENT_NAME, "draft_prompt")
        
        prompt = template.format(
            candidate_name=candidate.name,
            job_title=jd.title,
            salary_offered=jd.salary_range or 'TBD',
            joining_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            work_mode='Remote' if 'remote' in jd.raw_text.lower() else 'On-site',
            human_input=human_input if human_input else 'None'
        )
        
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        
        response = llm_invoke_with_retry(self.llm, messages)
        update_cost(pipeline, response.response_metadata, getattr(self.llm, "model_name", "llama-3.1-8b-instant"))
        
        return response.content.strip()

    def run(self, pipeline: PipelineState, human_input: Optional[str] = None) -> tuple[PipelineState, str]:
        # Stage check: ensure we are in offer stage
        if pipeline.current_stage == PipelineStage.FEEDBACK:
            pipeline.advance_stage(PipelineStage.OFFER)

        # 1. Build map of all candidate IDs that have received feedback
        # In the unified workflow, we draft the offer regardless of sentiment so the
        # human reviewer has all options on the final screen.
        target_fb_map = {}
        for fb in pipeline.feedbacks:
            # Check if this feedback has actual sentiment processed (meaning it was just run)
            if fb.sentiment is not None:
                target_fb_map[fb.candidate_id] = True

        logger.info(f"[OfferAgent] Candidates targeted for offer draft: {list(target_fb_map.keys())}")

        # 2. Find matching candidates — any status is acceptable to handle ID-mismatch edge cases
        eligible = [c for c in pipeline.candidates if c.candidate_id in target_fb_map]

        # 3. Fallback: if no exact match in candidates list but there IS processed feedback,
        #    create a minimal pseudo-candidate so the offer can still be drafted
        if not eligible and target_fb_map and pipeline.job_description:
            for cand_id in target_fb_map:
                fb = next((f for f in pipeline.feedbacks if f.candidate_id == cand_id), None)
                pseudo = Candidate(
                    candidate_id=cand_id,
                    name=f"Candidate {cand_id[:8]}",
                    email=f"{cand_id[:8]}@candidate.example.com",
                    status=CandidateStatus.INTERVIEWED,
                )
                eligible.append(pseudo)
                logger.warning(f"[OfferAgent] Used pseudo-candidate for id={cand_id[:8]}")

        if not eligible and not pipeline.offers:
            logger.warning(f"[OfferAgent] No candidates eligible for offer drafting yet.")
            pipeline.requires_human_approval = True
            pipeline.human_approval_reason = "human_feedback_approval"
            return pipeline, "⚠️ **Offer Agent**: Waiting for candidates with positive feedback to generate drafts. Please submit interviewer feedback first."

        # Case 1: Final Send (After human approval of the DRAFT)
        # We assume if human_input is 'approve' or decision in main.py was 'approve'
        # we check if we already have drafts.
        is_approval = False
        if human_input and "approve" in human_input.lower():
            is_approval = True

        # If they already have drafts and this is an approval call, send them.
        if pipeline.offers and is_approval:
            send_report = "✅ **Confirmation received. Sending final offer letters via email...**\n\n"
            for offer in pipeline.offers:
                cand = next((c for c in pipeline.candidates if c.candidate_id == offer.candidate_id), None)
                if cand:
                    logger.info(f"[Email] Sending FINAL Offer to {cand.name} at {cand.email}")
                    cand.status = CandidateStatus.OFFER_SENT
                    send_report += f"- 📧 **Email sent to {cand.name} (`{cand.candidate_id}`) successfully!**\n"

            send_report += "\n🎉 **All offers sent. Pipeline complete.**"
            send_report += "\n🔄 Moving to **Final Evaluation**."
            pipeline.advance_stage(PipelineStage.EVALUATION)
            pipeline.requires_human_approval = False
            pipeline.add_message("assistant", send_report, agent=self.AGENT_NAME)
            return pipeline, send_report

        # Case 2: Initial Draft Generation (or update with preferred items)
        report_lines = ["**📄 Offer Letter Draft Generation**\n"]
        
        import concurrent.futures
        def _draft_for_cand(candidate):
            text = self._draft_offer_letter(candidate, pipeline.job_description, human_input, pipeline)
            return candidate, text

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_draft_for_cand, c) for c in eligible]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for candidate, offer_text in results:
            existing = next((o for o in pipeline.offers if o.candidate_id == candidate.candidate_id), None)
            if existing:
                existing.offer_letter_text = offer_text
            else:
                offer = OfferLetter(
                    candidate_id=candidate.candidate_id,
                    job_id=pipeline.job_description.job_id if pipeline.job_description else "JOB-001",
                    candidate_name=candidate.name,
                    job_title=pipeline.job_description.title if pipeline.job_description else "Role",
                    department=pipeline.job_description.department if pipeline.job_description else "Dept",
                    salary_offered=pipeline.job_description.salary_range if pipeline.job_description else "TBD",
                    start_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                    offer_letter_text=offer_text
                )
                pipeline.offers.append(offer)
            
            report_lines.append(f"✅ Drafted offer for **{candidate.name}**.\n")
            report_lines.append(f"---\n{offer_text}\n---\n")

        report_lines.append("\n⚠️ **Action Required**: Please review the draft above.\n")
        report_lines.append("- To add **preferred items**, provide them in `human_input` and click `Execute`.\n")
        report_lines.append("- To **send** the offer, select `approve` in the `decision` field or type 'approve' in `human_input`.")
        
        pipeline.requires_human_approval = True
        pipeline.human_approval_reason = "Confirm offer letter draft or add preferred items."
        
        full_report = "".join(report_lines)
        pipeline.add_message("assistant", full_report, agent=self.AGENT_NAME)
        return pipeline, full_report
