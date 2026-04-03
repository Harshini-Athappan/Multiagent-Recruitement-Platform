"""
=============================================================================
Supervisor Orchestrator
=============================================================================

This module acts as the bridge between the FastAPI frontend routes and the 
LangGraph state machine (`graph_workflow.py`).

Key Responsibilities:
- Initializes LangGraph state using the SQLite PipelineState.
- Triggers graph execution (`recruitment_graph.invoke`).
- Manages memory thread IDs configured per pipeline for continuity.
- Handles resuming from Human-in-the-Loop checkpoints using `Command(resume=...)`.
"""
import traceback
from typing import Optional
from loguru import logger

from core.state_store import pipeline_store
from models.schemas import PipelineState, PipelineStage, ChatResponse, JobDescription
from agents.graph_workflow import recruitment_graph
from utils.response_parser import extract_thought_and_clean


class RecruitmentSupervisor:
    """
    Thin wrapper around the LangGraph recruitment workflow.
    Each pipeline maps to a unique thread_id for state persistence.
    """

    def __init__(self):
        self.graph = recruitment_graph

    def _get_config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _extract_pipeline_from_result(self, result, fallback: PipelineState, config: dict) -> PipelineState:
        """
        Safely extract PipelineState from graph result.
        When graph.invoke() is interrupted, it may return None.
        In that case, read the true state from the LangGraph checkpoint via get_state().
        """
        # Try the direct result first (graph completed without interrupt)
        if result is not None and isinstance(result, dict) and result.get("pipeline"):
            return result["pipeline"]

        # Fallback: read from the LangGraph MemorySaver checkpoint (most up-to-date)
        try:
            checkpoint_state = self.graph.get_state(config)
            if checkpoint_state and checkpoint_state.values.get("pipeline"):
                logger.info("[Supervisor] Pipeline read from LangGraph checkpoint.")
                return checkpoint_state.values["pipeline"]
        except Exception as e:
            logger.warning(f"[Supervisor] graph.get_state() failed: {e}")

        # Last resort: read from SQLite (may lag behind by one node)
        logger.warning("[Supervisor] Falling back to SQLite pipeline state.")
        fresh = pipeline_store.get(fallback.pipeline_id)
        return fresh if fresh else fallback

    def _create_chat_response(self, updated_pipeline: PipelineState) -> ChatResponse:
        """
        Consolidated helper to build a ChatResponse from a PipelineState.
        Handles thought extraction and UI text cleaning.
        """
        last_msg_obj = (
            updated_pipeline.messages[-1]
            if updated_pipeline.messages
            else {}
        )
        raw_content = last_msg_obj.get("content", "Processing... Awaiting next step.")

        # Extract thought if present
        thought, clean_response = extract_thought_and_clean(raw_content)

        return ChatResponse(
            pipeline_id=updated_pipeline.pipeline_id,
            stage=updated_pipeline.current_stage.value,
            response=clean_response,
            thought=thought,
            requires_human_approval=updated_pipeline.requires_human_approval,
            data={"stage_history": updated_pipeline.stage_history},
            metadata=updated_pipeline.metadata,
        )

    def auto_advance(self, pipeline_id: str, human_input: Optional[str] = None) -> ChatResponse:
        """
        Resume the LangGraph pipeline after human approval.
        """
        pipeline = pipeline_store.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found.")

        config = self._get_config(pipeline_id)

        # Clear approval flag so graph knows we've approved
        pipeline.requires_human_approval = False
        pipeline.human_approval_reason = None

        try:
            # Update graph's checkpointed state with fresh pipeline + human_input
            self.graph.update_state(
                config,
                {"pipeline": pipeline, "human_input": human_input or ""}
            )

            # Resume from the interrupt checkpoint
            result = self.graph.invoke(None, config=config)

            updated_pipeline = self._extract_pipeline_from_result(result, pipeline, config)
            pipeline_store.update(updated_pipeline)

            return self._create_chat_response(updated_pipeline)

        except Exception as e:
            logger.error(f"[Supervisor] auto_advance error for {pipeline_id}: {e}")
            logger.error(traceback.format_exc())
            raise

    def process(self, user_message: str, pipeline_id: Optional[str] = None) -> ChatResponse:
        """
        Entry point for new Job Descriptions.
        Creates a new pipeline and starts the LangGraph workflow.
        For existing pipelines, resumes from checkpoint.
        """
        try:
            if pipeline_id:
                pipeline = pipeline_store.get(pipeline_id)
                if not pipeline:
                    raise ValueError(f"Pipeline '{pipeline_id}' not found.")
                
                # If JD already parsed (pipeline exists and has real JD), just RESUME
                jd = pipeline.job_description
                if jd and jd.title != "Pending..." and jd.required_skills:
                    logger.info(f"[Supervisor] Pipeline {pipeline_id} already has JD '{jd.title}'. Resuming graph.")
                    return self.auto_advance(pipeline_id)
            else:
                pipeline = PipelineState()
                # Pre-populate raw_text so jd_intake_node can pick it up
                pipeline.job_description = JobDescription(
                    title="Pending...",
                    department="Pending...",
                    location="Pending...",
                    raw_text=user_message,
                )
                pipeline_store.create(pipeline)
                logger.info(f"[Supervisor] Created new pipeline {pipeline.pipeline_id}")

            thread_id = pipeline.pipeline_id
            config = self._get_config(thread_id)

            logger.info(f"[Supervisor] Starting graph for pipeline {thread_id}")
            result = self.graph.invoke(
                {"pipeline": pipeline, "human_input": None},
                config=config,
            )

            updated_pipeline = self._extract_pipeline_from_result(result, pipeline, config)
            pipeline_store.update(updated_pipeline)

            logger.info(f"[Supervisor] Graph paused/completed at stage: {updated_pipeline.current_stage.value}")

            return self._create_chat_response(updated_pipeline)

        except Exception as e:
            logger.error(f"[Supervisor] process() error: {e}")
            logger.error(traceback.format_exc())
            raise
