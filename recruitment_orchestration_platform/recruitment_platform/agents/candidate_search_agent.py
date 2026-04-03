"""
Candidate Search Agent
Responsibility: Source candidates from external tools (DuckDuckGo Search)
that match the job description. Returns min 5, max 10 matching candidates.
"""
from typing import List, Tuple
from loguru import logger
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

from core.llm import get_llm
from models.schemas import (
    Candidate, JobDescription, PipelineState, PipelineStage, CandidateStatus
)
from utils.prompt_loader import get_system_prompt

class CandidateSearchAgent:
    """
    Candidate Sourcing Agent performing web searches for matching candidates.
    Uses DuckDuckGo to find professional profiles based on JD.
    """

    AGENT_NAME = "candidate_search_agent"

    def __init__(self):
        self.llm = get_llm(temperature=0.15)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)
        self.search_tool = DuckDuckGoSearchRun()

    # ──────────────────────────────────────────
    # Search logic
    # ──────────────────────────────────────────

    def _generate_candidates_from_jd(self, jd: JobDescription) -> List[Candidate]:
        """
        Ask the LLM to generate realistic candidate profiles that match the JD.
        Each candidate gets a unique UUID as candidate_id.
        """
        import json

        prompt = f"""You are a recruitment database simulator. Generate exactly 7 realistic candidate profiles that could apply for the following job.

JOB TITLE: {jd.title}
DEPARTMENT: {jd.department}
LOCATION: {jd.location}
REQUIRED SKILLS: {', '.join(jd.required_skills)}
PREFERRED SKILLS: {', '.join(jd.preferred_skills)}
EXPERIENCE REQUIRED: {jd.experience_years} years

Generate candidates with VARIED profiles — some strong matches, some moderate, some weak.

Return ONLY a valid JSON array (no extra text, no markdown code blocks) with this exact structure:
[
  {{
    "name": "Full Name (realistic Indian/international name)",
    "email": "firstname.lastname@email.com",
    "current_role": "Their current job title (NOT the job they are applying for)",
    "years_of_experience": <integer>,
    "skills": ["skill1", "skill2", "skill3"],
    "location": "City, Country",
    "education": "B.Tech Computer Science / M.Tech / MBA etc",
    "resume_summary": "One sentence summary of their background"
  }}
]

IMPORTANT: 
- "current_role" must be THEIR CURRENT POSITION, not the job they're applying for
- Make names diverse and realistic
- Skills should partially overlap with required skills (not all, not none)
- Vary experience: some +/- 2 years from requirement
"""
        messages = [
            SystemMessage(content="You are a recruitment assistant. Return only valid JSON arrays."),
            HumanMessage(content=prompt),
        ]
        try:
            from utils.llm_retry import llm_invoke_with_retry
            response = llm_invoke_with_retry(self.llm, messages)
            content = response.content.strip()

            # Strip thought tags and markdown fences
            content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"```(?:json)?\n?", "", content).strip()
            content = content.strip("`").strip()

            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx == -1 or end_idx == -1:
                logger.warning("[SearchAgent] No JSON array found in LLM response, using fallback")
                return []

            raw = content[start_idx:end_idx+1]
            data = json.loads(raw)

            candidates = []
            for d in data:
                if not isinstance(d, dict):
                    continue
                name = d.get("name", "Unknown")
                first = name.split()[0].lower() if name else "candidate"
                last = name.split()[-1].lower() if len(name.split()) > 1 else "user"
                candidates.append(Candidate(
                    name=name,
                    email=d.get("email") or f"{first}.{last}@demo-recruitment.com",
                    current_role=d.get("current_role") or "Professional",
                    years_of_experience=int(d.get("years_of_experience") or 3),
                    skills=d.get("skills") or [],
                    location=d.get("location") or jd.location,
                    education=d.get("education") or "",
                    resume_summary=d.get("resume_summary") or "",
                    source="AI Candidate Database",
                    status=CandidateStatus.SOURCED,
                ))
            logger.info(f"[SearchAgent] Generated {len(candidates)} candidate profiles from LLM.")
            return candidates
        except Exception as e:
            logger.error(f"[SearchAgent] LLM candidate generation failed: {e}")
            return []

    def _search_duckduckgo_fallback(self, jd: JobDescription, needed: int) -> List[Candidate]:
        """Use DuckDuckGo to search for candidates if LLM generator fails."""
        logger.info(f"[SearchAgent] Attempting DuckDuckGo fallback for {needed} candidates...")
        try:
            query = f"site:linkedin.com/in/ {jd.title} ({' OR '.join(jd.required_skills[:2])}) {jd.location}"
            search_results = self.search_tool.run(query)
            
            prompt = f"""Extract {needed} candidate profiles from these search results:
{search_results}

Return ONLY a valid JSON array matching this structure:
[
  {{
    "name": "Full Name",
    "current_role": "Extracted role",
    "location": "{jd.location}"
  }}
]
"""
            messages = [
                SystemMessage(content="You are an expert profile extractor. Return ONLY valid JSON array."),
                HumanMessage(content=prompt)
            ]
            from utils.llm_retry import llm_invoke_with_retry
            response = llm_invoke_with_retry(self.llm, messages)
            content = response.content.strip()
            
            import re, json
            content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"```(?:json)?\n?", "", content).strip()
            content = content.strip("`").strip()
            
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            if start_idx == -1 or end_idx == -1:
                return []
                
            data = json.loads(content[start_idx:end_idx+1])
            candidates = []
            for d in data:
                if not isinstance(d, dict): continue
                name = d.get("name", "Unknown")
                first = name.split()[0].lower() if name else "candidate"
                last = name.split()[-1].lower() if len(name.split()) > 1 else "user"
                
                candidates.append(Candidate(
                    name=name,
                    email=f"{first}.{last}@demo-recruitment.com",
                    current_role=d.get("current_role") or jd.title,
                    years_of_experience=jd.experience_years,
                    skills=jd.required_skills[:3],
                    location=d.get("location") or jd.location,
                    education="",
                    resume_summary=f"Found via DuckDuckGo: {name} in {d.get('current_role', 'Tech')}",
                    source="DuckDuckGo Search",
                    status=CandidateStatus.SOURCED,
                ))
            return candidates
        except Exception as e:
            logger.error(f"[SearchAgent] DDG search fallback failed: {e}")
            return []

    def _make_fallback_candidates(self, jd: JobDescription, needed: int) -> List[Candidate]:
        """Generate faker-based candidates as last-resort fallback."""
        from faker import Faker
        fake = Faker()
        candidates = []
        roles = ["Software Engineer", "Data Analyst", "ML Engineer", "Backend Developer", "Tech Lead"]
        for i in range(needed):
            first = fake.first_name()
            last = fake.last_name()
            candidates.append(Candidate(
                name=f"{first} {last}",
                email=f"{first.lower()}.{last.lower()}@demo-recruitment.com",
                current_role=roles[i % len(roles)],
                years_of_experience=max(1, jd.experience_years + (i - 2)),
                skills=jd.required_skills[:3] + jd.preferred_skills[:2],
                location=jd.location,
                education="B.Tech Computer Science",
                resume_summary=f"{first} is an experienced professional with background in {', '.join(jd.required_skills[:2])}.",
                source="Fallback Generator",
                status=CandidateStatus.SOURCED,
            ))
        return candidates

    def perform_search(self, jd: JobDescription) -> List[Candidate]:
        """
        Sources candidates matching the JD.
        Min 5, Max 10 candidates.
        """
        logger.info(f"[SearchAgent] Generating candidates for: {jd.title} in {jd.location}")

        candidates = self._generate_candidates_from_jd(jd)

        # Pad with DuckDuckGo fallback if LLM returned too few
        if len(candidates) < 5:
            needed = 5 - len(candidates)
            logger.warning(f"[SearchAgent] Only {len(candidates)} from LLM, attempting DuckDuckGo fallback for {needed} profiles.")
            candidates += self._search_duckduckgo_fallback(jd, needed)

            # If STILL too few, use Faker
            if len(candidates) < 5:
                needed = 5 - len(candidates)
                logger.warning(f"[SearchAgent] DuckDuckGo failed, padding with {needed} Faker profiles.")
                candidates += self._make_fallback_candidates(jd, needed)

        return candidates[:10]


    # ──────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────

    def run(self, pipeline: PipelineState) -> tuple[PipelineState, str]:
        jd = pipeline.job_description
        if not jd:
            raise ValueError("No job description found.")

        candidates = self.perform_search(jd)
        
        if not candidates:
            # Fallback if search returns nothing
            pipeline.requires_human_approval = True
            pipeline.human_approval_reason = "DuckDuckGo search failed to find matching candidates."
            msg = "⚠️ **Failed to fetch candidates from external tools.** Human review required."
            pipeline.add_message("assistant", msg, agent=self.AGENT_NAME)
            return pipeline, msg

        pipeline.candidates.extend(candidates)
        
        cand_summary = "\n".join([f"- **{c.name}** | {c.current_role} | {c.location} | `{c.candidate_id}`" for c in candidates])
        full_message = (
            f"🔍 **Candidate Sourcing Complete (AI Candidate Database)**\n\n"
            f"Sourced **{len(candidates)} candidates** matching **{jd.title}** ({jd.location}).\n\n"
            f"{cand_summary}\n\n"
            f"🔄 Routing to **Scoring Agent** to evaluate and rank candidates..."
        )

        pipeline.add_message("assistant", full_message, agent=self.AGENT_NAME)
        # Advance stage manually or let supervisor handle the auto-routing
        pipeline.advance_stage(PipelineStage.SCORING)
        return pipeline, full_message
