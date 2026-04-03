"""
Scoring Agent
Responsibility: Quantitatively score each candidate against the JD using
a weighted rubric (skills, experience, education, preferred skills).
"""
import json
import re
from typing import List, Tuple
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from models.schemas import (
    Candidate, CandidateScore, JobDescription,
    PipelineState, PipelineStage, CandidateStatus,
)
from utils.prompt_loader import get_system_prompt


class ScoringAgent:
    """
    Candidate Scoring & Ranking Agent.
    Produces a CandidateScore for each candidate using a deterministic
    weighted rubric, enhanced with LLM-generated recommendation text.
    """

    AGENT_NAME = "scoring_agent"

    # Weights (must sum to 1.0)
    W_SKILL      = 0.40
    W_EXPERIENCE = 0.30
    W_EDUCATION  = 0.15
    W_PREFERRED  = 0.15

    # Thresholds for hire recommendation
    STRONG_FIT   = 80
    GOOD_FIT     = 60
    MODERATE_FIT = 40

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    # ──────────────────────────────────────────
    # Deterministic scoring functions
    # ──────────────────────────────────────────

    def _normalize(self, s: str) -> str:
        return s.lower().strip()

    def _score_skills(
        self, candidate_skills: List[str], required_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        if not required_skills:
            return 100.0, [], []
        c = {self._normalize(s) for s in candidate_skills}
        r = [self._normalize(s) for s in required_skills]
        matched = [s for s in r if s in c]
        missing = [s for s in r if s not in c]
        score = (len(matched) / len(r)) * 100
        return round(score, 2), matched, missing

    def _score_experience(self, candidate_exp: int, required_exp: int) -> float:
        diff = abs(candidate_exp - required_exp)
        if diff == 0:
            return 100.0
        elif diff == 1:
            return 85.0
        elif diff == 2:
            return 65.0
        elif diff <= 4:
            return 45.0
        else:
            return 25.0

    def _score_education(self, candidate_edu: str, required_edu: str) -> float:
        """Simple heuristic based on degree keyword matching."""
        if not required_edu:
            return 80.0  # Not specified → partial credit

        edu_lower = candidate_edu.lower()
        req_lower = required_edu.lower()

        degree_hierarchy = ["phd", "doctorate", "m.tech", "m.sc", "mba", "master",
                            "b.tech", "b.e.", "b.sc", "bachelor", "diploma"]

        # Check if required field matches candidate field
        keywords = ["computer science", "software", "engineering", "data", "ai", "ml",
                    "information technology", "mathematics", "statistics"]

        field_match = any(kw in edu_lower for kw in keywords)

        # Find degree level
        cand_level = next((i for i, d in enumerate(degree_hierarchy) if d in edu_lower), 99)
        req_level = next((i for i, d in enumerate(degree_hierarchy) if d in req_lower), 99)

        if cand_level <= req_level and field_match:
            return 100.0
        elif cand_level <= req_level:
            return 75.0
        elif field_match:
            return 65.0
        else:
            return 40.0

    def _score_preferred(
        self, candidate_skills: List[str], preferred_skills: List[str]
    ) -> float:
        if not preferred_skills:
            return 80.0  # No preferred skills → baseline
        c = {self._normalize(s) for s in candidate_skills}
        p = [self._normalize(s) for s in preferred_skills]
        matched = [s for s in p if s in c]
        return round((len(matched) / len(p)) * 100, 2)

    def _compute_overall(
        self, skill: float, exp: float, edu: float, preferred: float
    ) -> float:
        overall = (
            self.W_SKILL * skill +
            self.W_EXPERIENCE * exp +
            self.W_EDUCATION * edu +
            self.W_PREFERRED * preferred
        )
        return round(overall, 2)

    def _get_label(self, score: float) -> str:
        if score >= self.STRONG_FIT:
            return "🟢 Strong Fit"
        elif score >= self.GOOD_FIT:
            return "🟡 Good Fit"
        elif score >= self.MODERATE_FIT:
            return "🟠 Moderate Fit — Review Required"
        else:
            return "🔴 Weak Fit — Do Not Proceed"

    # ──────────────────────────────────────────
    # LLM recommendation text
    # ──────────────────────────────────────────

    def _generate_recommendation(
        self, candidate: Candidate, score: CandidateScore, jd: JobDescription, pipeline: PipelineState
    ) -> str:
        from utils.prompt_loader import get_agent_prompt_template
        template = get_agent_prompt_template(self.AGENT_NAME, "recommendation_prompt")
        
        prompt = template.format(
            candidate=candidate,
            score=score,
            matched_skills=", ".join(score.matched_skills) or 'None',
            missing_skills=", ".join(score.missing_skills) or 'None',
            jd=jd,
            required_skills=", ".join(jd.required_skills)
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        
        from utils.cost_tracker import update_cost
        from utils.llm_retry import llm_invoke_with_retry
        response = llm_invoke_with_retry(self.llm, messages)
        update_cost(pipeline, response.response_metadata, getattr(self.llm, "model_name", "llama-3.1-8b-instant"))
        
        return response.content.strip()

    # ──────────────────────────────────────────
    # Score a single candidate
    # ──────────────────────────────────────────

    def score_candidate(
        self, candidate: Candidate, jd: JobDescription, pipeline: PipelineState
    ) -> CandidateScore:
        skill_score, matched, missing = self._score_skills(
            candidate.skills, jd.required_skills
        )
        exp_score = self._score_experience(
            candidate.years_of_experience, jd.experience_years
        )
        edu_score = self._score_education(
            candidate.education, jd.education_requirement
        )
        pref_score = self._score_preferred(
            candidate.skills, jd.preferred_skills
        )
        overall = self._compute_overall(skill_score, exp_score, edu_score, pref_score)

        score = CandidateScore(
            candidate_id=candidate.candidate_id,
            job_id=jd.job_id,
            skill_match_score=skill_score,
            experience_score=exp_score,
            education_score=edu_score,
            overall_score=overall,
            matched_skills=matched,
            missing_skills=missing,
        )

        # Generate LLM recommendation
        score.recommendation = self._generate_recommendation(candidate, score, jd, pipeline)
        logger.info(
            f"[ScoringAgent] {candidate.name}: overall={overall:.1f} "
            f"skill={skill_score:.1f} exp={exp_score} edu={edu_score} pref={pref_score}"
        )
        return score

    # ──────────────────────────────────────────
    # Format output message
    # ──────────────────────────────────────────

    def _format_scorecard(
        self,
        scored: List[Tuple[Candidate, CandidateScore]],
        proceed_candidates: List[Candidate],
    ) -> str:
        lines = ["**📊 Candidate Scoring Results (Ranked)**\n"]
        for i, (cand, score) in enumerate(scored, 1):
            label = self._get_label(score.overall_score)
            lines.append(
                f"**{i}. {cand.name}** (ID: `{cand.candidate_id}`) — {label}\n"
                f"   Overall: **{score.overall_score:.1f}/100** | "
                f"Skills: {score.skill_match_score:.0f}% | "
                f"Experience: {score.experience_score:.0f} | "
                f"Education: {score.education_score:.0f}\n"
                f"   Matched: {', '.join(score.matched_skills) or 'None'}\n"
                f"   Missing: {', '.join(score.missing_skills) or 'None'}\n"
                f"   📝 {score.recommendation}\n"
            )

        if proceed_candidates:
            names = ", ".join(f"{c.name} (`{c.candidate_id}`)" for c in proceed_candidates)
            lines.append(f"\n✅ **Proceeding to interview scheduling:** {names}")
            lines.append("\n💡 **Quick Tip**: Copy the `candidate_id` above to use in the feedback form in Step 4.")
        else:
            lines.append("\n⚠️ No candidates scored above the threshold. Human review required.")

        lines.append("\n🔄 Routing to **Scheduling Agent**...")
        return "\n".join(lines)

    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(self, pipeline: PipelineState) -> tuple[PipelineState, str]:
        jd = pipeline.job_description
        if not jd:
            raise ValueError("Job description not found in pipeline state.")
        if not pipeline.candidates:
            raise ValueError("No candidates to score.")

        scored_pairs: List[Tuple[Candidate, CandidateScore]] = []
        import concurrent.futures
        import threading
        import time

        # Semaphore limits simultaneous Groq calls to avoid TPM rate limits
        rate_semaphore = threading.Semaphore(2)

        def _safe_score(cand, attempt=0):
            with rate_semaphore:
                try:
                    score = self.score_candidate(cand, jd, pipeline)
                    return (cand, score)
                except Exception as e:
                    err_str = str(e).lower()
                    if "rate_limit" in err_str or "429" in err_str or "too many" in err_str:
                        wait = (2 ** attempt) * 5  # 5s, 10s, 20s
                        logger.warning(f"[ScoringAgent] Rate limit hit for {cand.name}. Retrying in {wait}s...")
                        time.sleep(wait)
                        if attempt < 3:
                            return _safe_score(cand, attempt + 1)
                    logger.error(f"[ScoringAgent] Failed to score {cand.name}: {e}")
                    return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_safe_score, cand) for cand in pipeline.candidates]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    pipeline.scores.append(result[1])
                    scored_pairs.append(result)

        # Sort by overall score descending
        scored_pairs.sort(key=lambda x: x[1].overall_score, reverse=True)

        # Determine which candidates proceed
        proceed = [
            cand for cand, score in scored_pairs
            if score.overall_score >= self.MODERATE_FIT
        ]

        # Flag moderate fits for human review
        for cand, score in scored_pairs:
            if self.MODERATE_FIT <= score.overall_score < self.GOOD_FIT:
                cand.status = CandidateStatus.SCREENED
            elif score.overall_score >= self.GOOD_FIT:
                cand.status = CandidateStatus.SCREENED
            else:
                cand.status = CandidateStatus.REJECTED

        message = self._format_scorecard(scored_pairs, proceed)
        message += "\n\n⚠️ **Human Approval Required** to proceed to interview scheduling."
        
        # Setting human approval as per request
        pipeline.requires_human_approval = True
        pipeline.human_approval_reason = "Candidates have been scored. Human must approve results before scheduling interviews."

        pipeline.add_message("assistant", message, agent=self.AGENT_NAME)
        # Advance the stage but with human approval flag set
        return pipeline, message
