"""
Feedback Agent
Responsibility: Collect interviewer feedback, perform sentiment analysis,
compute aggregated scores, and prompt human for next steps (Offer/Reject).
"""
import json
import re
from typing import List, Optional
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage

from core.llm import get_llm
from models.schemas import (
    Candidate, InterviewFeedback, FeedbackSentiment,
    PipelineState, PipelineStage, CandidateStatus,
)
from utils.prompt_loader import get_system_prompt

class FeedbackAgent:
    """
    Interview Feedback & Evaluation Agent.
    Analyses free-text feedback for sentiment and produces hire recommendations.
    Always pauses for human confirmation before moving to the Offer stage.
    """

    AGENT_NAME = "feedback_agent"

    # Weights for aggregated score
    W_TECHNICAL     = 0.35
    W_COMMUNICATION = 0.25
    W_CULTURE       = 0.20
    W_PROBLEM_SOLVE = 0.20

    def __init__(self):
        self.llm = get_llm(temperature=0.15)
        self.system_prompt = get_system_prompt(self.AGENT_NAME)

    def _aggregate_score(self, fb: InterviewFeedback) -> float:
        weighted = (
            self.W_TECHNICAL     * fb.technical_rating +
            self.W_COMMUNICATION * fb.communication_rating +
            self.W_CULTURE       * fb.culture_fit_rating +
            self.W_PROBLEM_SOLVE * fb.problem_solving_rating
        )
        return round((weighted / 10) * 100, 2)

    def _analyse_sentiment(self, raw_feedback: str, pipeline: PipelineState) -> tuple[FeedbackSentiment, float]:
        from utils.prompt_loader import get_agent_prompt_template
        template = get_agent_prompt_template(self.AGENT_NAME, "sentiment_prompt")
        try:
            prompt = template.format(raw_feedback=raw_feedback)
        except (KeyError, ValueError) as e:
            logger.warning(f"[FeedbackAgent] Prompt formatting failed: {e}. Falling back to basic prompt.")
            # Fallback: manually replace the keyword if format fails
            prompt = template.replace("{raw_feedback}", raw_feedback)

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=prompt)]
        
        from utils.cost_tracker import update_cost
        from utils.llm_retry import llm_invoke_with_retry
        try:
            response = llm_invoke_with_retry(self.llm, messages)
            update_cost(pipeline, response.response_metadata, getattr(self.llm, "model_name", "llama-3.1-8b-instant"))
        except Exception as e:
            logger.error(f"[FeedbackAgent] LLM invocation failed: {e}")
            return FeedbackSentiment.NEUTRAL, 0.0
        
        # Remove <thought> to not break json parsing
        content = re.sub(r"<thought>.*?</thought>", "", response.content, flags=re.DOTALL).strip()
        
        # Robust JSON extraction
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            raw = content

        try:
            data = json.loads(raw)
            sentiment_val = data.get("sentiment", "neutral").lower()
            # Map back to enum to be safe
            if "positive" in sentiment_val: s = FeedbackSentiment.POSITIVE
            elif "negative" in sentiment_val: s = FeedbackSentiment.NEGATIVE
            else: s = FeedbackSentiment.NEUTRAL
            
            return s, float(data.get("score", 0.0))
        except Exception as e:
            logger.error(f"[FeedbackAgent] JSON parse error: {e}. Raw content: {content}")
            return FeedbackSentiment.NEUTRAL, 0.0

    def run(self, pipeline: PipelineState) -> tuple[PipelineState, str]:
        # Identify non-processed feedbacks
        new_feedbacks = [fb for fb in pipeline.feedbacks if fb.sentiment is None]
        
        if not new_feedbacks:
            return pipeline, "No new feedback to process."

        report = "**📝 Feedback Analysis Result**\n\n"
        processed_count = 0
        
        import concurrent.futures

        def _process_feedback(fb):
            try:
                cand = next((c for c in pipeline.candidates if c.candidate_id == fb.candidate_id), None)
                
                fb.aggregated_score = self._aggregate_score(fb)
                fb.sentiment, fb.sentiment_score = self._analyse_sentiment(fb.raw_feedback, pipeline)
                # ALWAYS set hire_recommendation from sentiment (even if candidate not found in list)
                fb.hire_recommendation = (fb.sentiment == FeedbackSentiment.POSITIVE)

                if fb.sentiment == FeedbackSentiment.POSITIVE:
                    sentiment_ui = "🟢 POSITIVE"
                elif fb.sentiment == FeedbackSentiment.NEGATIVE:
                    sentiment_ui = "🔴 NEGATIVE"
                else:
                    sentiment_ui = "🟡 NEUTRAL"

                cand_name = cand.name if cand else f"Candidate {fb.candidate_id[:8]}"
                rep = f"👤 **Candidate**: {cand_name}\n"
                rep += f"📊 **Score**: {fb.aggregated_score}/100\n"
                rep += f"🎭 **Sentiment**: {sentiment_ui}\n"
                rep += f"💬 **Feedback**: \"{fb.raw_feedback}\"\n\n"
                
                if fb.sentiment == FeedbackSentiment.POSITIVE:
                    rep += "✅ Since the feedback is **POSITIVE**, would you like to **Proceed to Offer** or Reject?\n\n"
                elif fb.sentiment == FeedbackSentiment.NEGATIVE:
                    rep += "⚠️ Since the feedback is **NEGATIVE**, please confirm if you want to **Reject** the candidate.\n\n"
                else:
                    rep += "🤔 Sentiment is **NEUTRAL**. Please review and decide next steps.\n\n"

                if cand:
                    cand.status = CandidateStatus.INTERVIEWED
                return fb, cand, rep
            except Exception as e:
                logger.error(f"[FeedbackAgent] Failed to process feedback for {fb.candidate_id}: {e}")
                return None, None, ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_process_feedback, fb) for fb in new_feedbacks]
            for future in concurrent.futures.as_completed(futures):
                fb_res, cand_res, rep_str = future.result()
                if fb_res:
                    report += rep_str
                    if cand_res:
                        processed_count += 1
                        cand_res.status = CandidateStatus.INTERVIEWED


        if processed_count > 0:
            pipeline.requires_human_approval = True
            pipeline.human_approval_reason = "Human must decide next steps after feedback analysis."
        
        pipeline.add_message("assistant", report, agent=self.AGENT_NAME)
        return pipeline, report
