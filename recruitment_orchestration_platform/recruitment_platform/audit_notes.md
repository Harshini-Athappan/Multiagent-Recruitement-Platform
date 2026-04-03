# Multiagent Flow Audit & Fix Plan

## Issues Identified

### 1. JD Re-intake After Candidates Found
**Root Cause**: `supervisor.process()` is called with a `pipeline_id` when the pipeline already exists (stage = scoring/scheduling). But the graph ALWAYS starts from `START → jd_intake_node`. The `auto_advance()` method correctly resumes mid-graph, but the API endpoint `sv.process()` re-runs from start. Looking at the flow:
- UI calls `call_api("POST", "/jd/upload", ...)` which calls `sv.process()`
- `sv.process()` always calls `self.graph.invoke({"pipeline": pipeline, ...})` from scratch
- LangGraph replays from START for the thread_id → jd_intake_node fires again

**Fix**: In graph_workflow.py, `jd_intake_node` must check if JD is already parsed (not `Pending...`). If candidates exist, skip. Also, the `supervisor.process()` should ONLY be called for fresh pipelines - never with existing pipeline_id after JD is done.

### 2. Rate Limit with Groq 
**Root Cause**: `concurrent.futures.ThreadPoolExecutor(max_workers=5)` fires 5–7 simultaneous LLM calls which blasts Groq's TPM limit. Each candidate score requires an LLM call.

**Fix**: Use semaphore-controlled concurrency (max 2 simultaneous) + exponential backoff retry.

### 3. Multiagent Flow - LangGraph Architecture Issues
- `feedback_node` goes directly to `offer_node` without interrupt — correct per design
- `human_scheduling_approval` → `human_feedback_approval` in one step (no scheduling email sending in between) — **WRONG**: scheduling_node should confirm, THEN go to feedback wait
- The `feedback_node` is triggered by auto_advance from the feedback API, but the graph interrupt is `human_feedback_approval` before `feedback_node` — this is correct
- `offer_node` loops back via `human_offer_approval` — correct

### 4. Graph Restart Issue (Core Bug)
When `supervisor.process()` is called with an existing `pipeline_id`, `graph.invoke()` called with full initial state on an existing thread_id tries to replay from the last checkpoint. But if the graph was interrupted at `human_scoring_approval`, calling `invoke()` again with new initial state creates a conflict. The correct call is `graph.invoke(None, config=config)` (resume) not `graph.invoke({...}, config=config)` (restart).
