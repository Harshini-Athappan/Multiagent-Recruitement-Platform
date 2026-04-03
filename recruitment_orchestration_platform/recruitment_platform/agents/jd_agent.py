"""
JD Intake Agent
Responsibility: Parse job descriptions from raw text, PDFs, or natural language
and produce a structured JobDescription object.
"""
import json
import re
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from loguru import logger

from core.llm import get_llm
from models.schemas import JobDescription, PipelineState, PipelineStage
from utils.prompt_loader import get_system_prompt


class JDAgent:
    """
    Job Description Intake Agent.
    Accepts raw text or natural language and returns a structured JobDescription.
    """

    AGENT_NAME = "jd_agent"

    def __init__(self):
        self.llm = get_llm(temperature=0.1)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    def _build_extraction_prompt(self, raw_input: str) -> str:
        from utils.prompt_loader import get_agent_prompt_template
        template = get_agent_prompt_template(self.AGENT_NAME, "extraction_prompt")
        return template.format(raw_input=raw_input)


    def parse(self, raw_input: str, pipeline: PipelineState) -> tuple[JobDescription, str]:
        """
        Parse raw JD text or natural language into a JobDescription.
        Returns (JobDescription, assistant_message).
        """
        logger.info(f"[JDAgent] Parsing JD for pipeline {pipeline.pipeline_id}")

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_extraction_prompt(raw_input)),
        ]

        from utils.cost_tracker import update_cost
        from utils.llm_retry import llm_invoke_with_retry
        
        response = llm_invoke_with_retry(self.llm, messages)
        update_cost(pipeline, response.response_metadata, getattr(self.llm, "model_name", "llama-3.1-8b-instant"))
        content = response.content.strip()

        # Robust JSON extraction
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(0)
        else:
            raw_content = content

        try:
            jd_data = json.loads(raw_content)
        except json.JSONDecodeError as e:
            logger.error(f"[JDAgent] JSON parse error: {e}\nRaw: {content}")
            raise ValueError(f"Failed to parse JD from agent response. Please check technical logs.")

        # Ensure correct types and handle values
        exp_val = jd_data.get("experience_years", 0)
        try:
            exp_int = int(exp_val)
        except (ValueError, TypeError):
            exp_int = 0

        jd = JobDescription(
            title=jd_data.get("title", "Unknown Role"),
            department=jd_data.get("department", "Engineering"),
            location=jd_data.get("location", "Remote"),
            employment_type=jd_data.get("employment_type", "Full-time"),
            required_skills=jd_data.get("required_skills") or [],
            preferred_skills=jd_data.get("preferred_skills") or [],
            experience_years=exp_int,
            education_requirement=jd_data.get("education_requirement") or "",
            responsibilities=jd_data.get("responsibilities") or [],
            salary_range=jd_data.get("salary_range"),
            raw_text=raw_input,
        )

        assistant_message = self._format_confirmation(jd)
        logger.info(f"[JDAgent] Parsed JD: {jd.title} | Skills: {jd.required_skills}")
        return jd, assistant_message

    def _format_confirmation(self, jd: JobDescription) -> str:
        skills_str = ", ".join(jd.required_skills) if jd.required_skills else "None specified"
        preferred_str = ", ".join(jd.preferred_skills) if jd.preferred_skills else "None"
        responsibilities = "\n    • ".join(jd.responsibilities[:5]) if jd.responsibilities else "Not specified"

        return f"""✅ **Job Description Successfully Parsed**

**Role:** {jd.title}
**Department:** {jd.department}
**Location:** {jd.location}
**Employment Type:** {jd.employment_type}
**Experience Required:** {jd.experience_years}+ years
**Education:** {jd.education_requirement or 'Not specified'}
**Salary Range:** {jd.salary_range or 'Not specified'}

**Required Skills:**
{', '.join(jd.required_skills) if jd.required_skills else 'None'}

**Preferred Skills:**
{', '.join(jd.preferred_skills) if jd.preferred_skills else 'None'}

**Key Responsibilities:**
    • {responsibilities}

🔄 Routing to **Candidate Search Agent** to source matching profiles..."""

    def run(self, user_input: str, pipeline: PipelineState) -> tuple[PipelineState, str]:
        """
        Main entry point. Parses JD and updates pipeline state.
        """
        jd, message = self.parse(user_input, pipeline)
        pipeline.job_description = jd
        pipeline.job_id = jd.job_id
        pipeline.add_message("assistant", message, agent=self.AGENT_NAME)
        pipeline.advance_stage(PipelineStage.CANDIDATE_SEARCH)
        return pipeline, message
