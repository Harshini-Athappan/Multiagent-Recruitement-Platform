"""
=============================================================================
LangGraph Workflow Orchestrator
=============================================================================

This module defines the directed graph (state machine) that manages the 
progression of the recruitment pipeline. It connects all autonomous agents 
together using LangGraph's StateGraph.

Features:
- **Rate-Limiting & Idempotency**: Nodes check state and exit early if their stage is already complete.
- **Conditional Routing**: Decisions are made based on the current PipelineState.
- **Human-in-the-Loop**: Execution pauses via `interrupt_before` at critical approval gates.
"""
from typing import TypedDict, Optional, List, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from models.schemas import PipelineState, PipelineStage, CandidateStatus, FeedbackSentiment, JobDescription, CandidateScore, Candidate
from core.state_store import pipeline_store
from agents.jd_agent import JDAgent
from agents.candidate_search_agent import CandidateSearchAgent
from agents.scoring_agent import ScoringAgent
from agents.scheduling_agent import SchedulingAgent
from agents.feedback_agent import FeedbackAgent
from agents.offer_agent import OfferAgent
from agents.evaluation_agent import EvaluationAgent
from utils.notification_service import notify_stage_transition

# ── 1. GRAPH STATE ──────────────────────────────────────────────
class RecruitmentState(TypedDict):
    pipeline: PipelineState
    human_input: Optional[str]
    last_agent: Optional[str]

# ── 2. AGENT INSTANCES ──────────────────────────────────────────
jd_agent       = JDAgent()
search_agent   = CandidateSearchAgent()
scoring_agent  = ScoringAgent()
scheduling_agent = SchedulingAgent()
feedback_agent = FeedbackAgent()
offer_agent    = OfferAgent()
evaluation_agent = EvaluationAgent()

# ── 3. HELPERS ──────────────────────────────────────────────────

def node_log(node_name: str, pipeline: PipelineState, msg: str = ""):
    """Standardized logging and state persistence for graph nodes."""
    prefix = f"[Graph:{node_name}]"
    if msg:
        logger.info(f"{prefix} {msg}")
    pipeline_store.update(pipeline)

def jd_intake_node(state: RecruitmentState):
    """Stage 1: Parse JD text and extract structured job description."""
    pipeline = state["pipeline"]
    
    if pipeline.job_description and pipeline.job_description.title != "Pending...":
        node_log("JD", pipeline, "✅ JD already parsed. Skipping.")
        return {"pipeline": pipeline, "last_agent": "jd_agent"}

    node_log("JD", pipeline, "🚦 Parsing raw JD text...")
    pipeline, _ = jd_agent.run(pipeline.job_description.raw_text, pipeline)
    node_log("JD", pipeline)
    return {"pipeline": pipeline, "last_agent": "jd_agent"}

def search_node(state: RecruitmentState):
    """Stage 2: Search and source candidates matching the JD."""
    pipeline = state["pipeline"]
    
    if pipeline.candidates:
        node_log("Search", pipeline, f"✅ {len(pipeline.candidates)} candidates already sourced. Skipping.")
        return {"pipeline": pipeline, "last_agent": "search_agent"}

    node_log("Search", pipeline, "🚦 Sourcing candidates...")
    pipeline, _ = search_agent.run(pipeline)
    notify_stage_transition(pipeline, "candidate_search")
    node_log("Search", pipeline)
    return {"pipeline": pipeline, "last_agent": "search_agent"}

def scoring_node(state: RecruitmentState):
    """Stage 3: Score and rank candidates."""
    pipeline = state["pipeline"]
    
    if pipeline.scores:
        node_log("Scoring", pipeline, "✅ Scores already computed. Skipping.")
        pipeline.requires_human_approval = True
        pipeline.human_approval_reason = "scoring_complete"
        return {"pipeline": pipeline, "last_agent": "scoring_agent"}

    node_log("Scoring", pipeline, "🚦 Ranking candidates...")
    pipeline, _ = scoring_agent.run(pipeline)
    pipeline.requires_human_approval = True
    pipeline.human_approval_reason = "scoring_complete"
    notify_stage_transition(pipeline, "scoring")
    node_log("Scoring", pipeline)
    return {"pipeline": pipeline, "last_agent": "scoring_agent"}

def human_scoring_approval(state: RecruitmentState):
    """Breakpoint Node: Human has approved scores."""
    pipeline = state["pipeline"]
    logger.info(f"[Graph Routing] 🚦 DECISION: Resuming from 'human_scoring_approval'. Advancing to 'scheduling'.")
    pipeline.requires_human_approval = False
    pipeline.current_stage = PipelineStage.SCHEDULING
    notify_stage_transition(pipeline, "scheduling")
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline}

def scheduling_node(state: RecruitmentState):
    """Stage 4: Propose interview time slots."""
    pipeline = state["pipeline"]
    human_input = state.get("human_input")
    node_log("Scheduling", pipeline, "🚦 Coordinating interview slots...")
    pipeline, _ = scheduling_agent.run(pipeline, human_input=human_input)
    pipeline.requires_human_approval = True
    pipeline.human_approval_reason = "scheduling_complete"
    node_log("Scheduling", pipeline)
    return {"pipeline": pipeline, "last_agent": "scheduling_agent", "human_input": None}

def human_scheduling_approval(state: RecruitmentState):
    """Breakpoint Node: Human has reviewed slots."""
    pipeline = state["pipeline"]
    human_input = state.get("human_input")
    
    # Check if human provided input (reschedule request)
    if human_input and "approve" not in human_input.lower() and len(human_input.strip()) > 1:
        logger.info(f"[Graph Routing] 🚦 Human requested rescheduling: '{human_input}'. Re-routing to scheduling_node.")
        return {"pipeline": pipeline, "human_input": human_input}

    logger.info(f"[Graph Routing] 🚦 Slots confirmed or no input provided. Advancing to 'feedback' node.")
    pipeline.requires_human_approval = False
    pipeline.current_stage = PipelineStage.FEEDBACK
    notify_stage_transition(pipeline, "feedback")
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline, "human_input": None}

def human_feedback_approval(state: RecruitmentState):
    """Breakpoint Node: Human (Interviewer) has submitted feedback or HR is ready to proceed."""
    pipeline = state["pipeline"]
    logger.info(f"[Graph Routing] 🚦 Resuming from feedback pause. Running sentiment analysis now.")
    pipeline.requires_human_approval = False
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline}

def feedback_node(state: RecruitmentState):
    """Stage 5: Process interview feedback & run sentiment analysis."""
    pipeline = state["pipeline"]
    
    # Check if all existing feedback is processed
    processed_count = len([fb for fb in pipeline.feedbacks if fb.sentiment and fb.hire_recommendation])
    
    if pipeline.feedbacks and processed_count == len(pipeline.feedbacks):
        node_log("Feedback", pipeline, "✅ All feedback already processed.")
        return {"pipeline": pipeline, "last_agent": "feedback_agent"}

    if not pipeline.feedbacks:
        node_log("Feedback", pipeline, "⚠️ Waiting for interviewer feedback.")
        pipeline.requires_human_approval = True
        pipeline.human_approval_reason = "human_feedback_approval"
        node_log("Feedback", pipeline)
        return {"pipeline": pipeline, "last_agent": "feedback_agent"}

    node_log("Feedback", pipeline, "🚦 Running sentiment analysis...")
    pipeline, _ = feedback_agent.run(pipeline)
    node_log("Feedback", pipeline)
    return {"pipeline": pipeline, "last_agent": "feedback_agent"}

def human_sentiment_review(state: RecruitmentState):
    """
    Breakpoint Node: Human reviews sentiment analysis results.
    - If positive candidate exists → proceed to offer drafting
    - If negative/neutral → human can approve (end pipeline) or reject (go to evaluation directly)
    """
    pipeline = state["pipeline"]
    human_input = state.get("human_input", "")
    
    logger.info(f"[Graph Routing] 🚦 Human sentiment review. Input: '{human_input}'")
    pipeline.requires_human_approval = False
    
    # Check if human wants to skip offer and end pipeline
    if human_input and "reject" in human_input.lower():
        logger.info("[Graph Routing] 🚦 Human chose to REJECT — routing to evaluation.")
        pipeline.advance_stage(PipelineStage.EVALUATION)
    elif human_input and "approve" in human_input.lower():
        logger.info("[Graph Routing] 🚦 Human chose to APPROVE. Overriding AI sentiment to positive.")
        for fb in pipeline.feedbacks:
            fb.hire_recommendation = True
    
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline, "human_input": human_input}

def offer_node(state: RecruitmentState):
    """Stage 6: Draft or send the offer letter."""
    pipeline = state["pipeline"]
    human_input = state.get("human_input")
    node_log("Offer", pipeline, f"🚦 Drafting offer (Human Input: {bool(human_input)})...")
    pipeline, _ = offer_agent.run(pipeline, human_input=human_input)
    node_log("Offer", pipeline)
    return {"pipeline": pipeline, "last_agent": "offer_agent", "human_input": None}

def human_offer_approval(state: RecruitmentState):
    """Breakpoint Node: Human has reviewed the offer draft."""
    pipeline = state["pipeline"]
    logger.info(f"[Graph Routing] 🚦 DECISION: Resuming from 'human_offer_approval'. Offer decision received.")
    pipeline.requires_human_approval = False
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline}

def evaluation_node(state: RecruitmentState):
    """Stage 7: Evaluate the pipeline and generate final report."""
    pipeline = state["pipeline"]
    logger.info(f"[Graph Routing] 🚦 DECISION: Moving to 'evaluation' to generate completion report.")
    pipeline, msg = evaluation_agent.run(pipeline)
    notify_stage_transition(pipeline, "completed")
    pipeline_store.update(pipeline)
    return {"pipeline": pipeline, "last_agent": "evaluation_agent"}

# ── 4. ROUTING ──────────────────────────────────────────────────

def route_after_search(state: RecruitmentState):
    """Only proceed to scoring if we actually found candidates."""
    candidates = state["pipeline"].candidates
    if not candidates:
        logger.warning("[Graph] No candidates found after search. Ending pipeline.")
        return END
    return "scoring_node"

def route_after_scheduling(state: RecruitmentState):
    """Route based on whether human provided a reschedule request in scheduling_node or just approved."""
    human_input = state.get("human_input")
    if human_input and "approve" not in human_input.lower() and len(human_input.strip()) > 1:
        return "scheduling_node"
    return "human_feedback_approval"

def route_after_sentiment_review(state: RecruitmentState):
    """
    In the unified workflow, we ALWAYS route to offer_node so the human can see BOTH the sentiment
    and the draft offer on the final review screen.
    """
    logger.info("[Graph Mapping] Unified review mode. Routing to offer_node directly.")
    return "offer_node"

def route_after_offer(state: RecruitmentState):
    """Finish the pipeline if evaluation stage reached, otherwise wait for approval."""
    pipeline = state["pipeline"]
    if pipeline.current_stage == PipelineStage.EVALUATION:
        logger.info("[Graph Mapping] 🏁 Offer process finalized. Moving to evaluation.")
        return "evaluation_node"
    return "human_offer_approval"

# ── 5. BUILD GRAPH ──────────────────────────────────────────────

builder = StateGraph(RecruitmentState)

# Add all nodes
builder.add_node("jd_intake_node",             jd_intake_node)
builder.add_node("search_node",                search_node)
builder.add_node("scoring_node",               scoring_node)
builder.add_node("human_scoring_approval",     human_scoring_approval)
builder.add_node("scheduling_node",            scheduling_node)
builder.add_node("human_scheduling_approval",  human_scheduling_approval)
builder.add_node("human_feedback_approval",    human_feedback_approval)
builder.add_node("feedback_node",              feedback_node)
builder.add_node("human_sentiment_review",     human_sentiment_review)  # NEW
builder.add_node("offer_node",                 offer_node)
builder.add_node("human_offer_approval",       human_offer_approval)
builder.add_node("evaluation_node",            evaluation_node)

# Define edges
builder.add_edge(START,              "jd_intake_node")
builder.add_edge("jd_intake_node",   "search_node")

builder.add_conditional_edges("search_node", route_after_search, {
    "scoring_node": "scoring_node",
    END: END,
})

builder.add_edge("scoring_node",               "human_scoring_approval")
builder.add_edge("human_scoring_approval",     "scheduling_node")

builder.add_conditional_edges("scheduling_node", route_after_scheduling, {
    "scheduling_node": "scheduling_node",
    "human_feedback_approval": "human_scheduling_approval"
})

builder.add_edge("human_scheduling_approval",  "human_feedback_approval")
builder.add_edge("human_feedback_approval",    "feedback_node")
# After sentiment analysis → pause for human to review → then route
builder.add_edge("feedback_node",              "human_sentiment_review")

builder.add_conditional_edges("human_sentiment_review", route_after_sentiment_review, {
    "offer_node":      "offer_node",
    "evaluation_node": "evaluation_node",
})

builder.add_conditional_edges("offer_node", route_after_offer, {
    "human_offer_approval": "human_offer_approval",
    "evaluation_node":      "evaluation_node",
})

builder.add_edge("human_offer_approval", "offer_node")
builder.add_edge("evaluation_node", END)

# ── 6. COMPILE WITH INTERRUPTS ──────────────────────────────────
# Interrupt BEFORE each human approval node.
memory = MemorySaver()
recruitment_graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[
        "human_scoring_approval",    # ⛔ Pause after scoring — HR reviews ranked candidates
        "human_scheduling_approval", # ⛔ Pause after scheduling — HR confirms slots
        "human_feedback_approval",   # ⛔ Pause to let interviewer submit feedback form
        "human_offer_approval",      # ⛔ Pause after offer draft — HR reviews before sending
    ]
)

logger.info("[Graph] LangGraph Recruitment Workflow compiled successfully.")
