import streamlit as st
import httpx
import json
import time
from datetime import datetime

API_URL = "http://127.0.0.1:8016"

st.set_page_config(
    page_title="HR AI Agentic Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4a90e2;
        color: white !important;
    }
    .stButton>button * {
        color: white !important;
    }
    /* Ensure agent-card does NOT bleed into buttons */
    .agent-card .stButton>button {
        color: white !important;
    }
    .stMetric label, .stMetric div[data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    .agent-card {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4a90e2;
        background-color: #ffffff !important;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #1a1a1a !important;
    }
    /* Only target text nodes inside agent-card, not input/form widgets */
    .agent-card p, .agent-card span, .agent-card div > p,
    .agent-card li, .agent-card h1, .agent-card h2, .agent-card h3,
    .agent-card h4, .agent-card strong, .agent-card b, .agent-card em {
        color: #1a1a1a !important;
    }
    /* Ensure Streamlit input widgets keep their own styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    /* Fix label text color for form inputs - Use a professional Blue instead of harsh black */
    .stTextInput label, .stTextArea label, .stSlider label,
    .stSelectbox label, .stForm label {
        color: #4a90e2 !important;
        font-weight: 600 !important;
    }
    /* Ensure metric delta also has color */
    [data-testid="stMetricDelta"] svg {
        fill: #1a1a1a !important;
    }
    /* Fix code blocks inside agent-card */
    .agent-card pre {
        background-color: #1e1e2e !important;
        border-radius: 6px;
        padding: 12px;
        overflow-x: auto;
    }
    .agent-card pre code, .agent-card code {
        color: #cdd6f4 !important;
        background-color: transparent !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.875rem;
    }
    /* Evaluation report specific styling */
    .eval-report {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #22c55e;
        background-color: #f0fdf4 !important;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .eval-report pre {
        background-color: #1e1e2e !important;
        border-radius: 8px;
        padding: 14px;
    }
    .eval-report pre code {
        color: #a6e3a1 !important;
        background-color: transparent !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'pipeline_id' not in st.session_state:
    st.session_state.pipeline_id = None
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = "jd_intake"
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'requires_approval' not in st.session_state:
    st.session_state.requires_approval = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'thought' not in st.session_state:
    st.session_state.thought = ""

# --- UTILS ---
def call_api(method, endpoint, params=None, data=None, files=None):
    try:
        # Increased timeout to 5 minutes for long-running autonomous chains
        with httpx.Client(timeout=300.0) as client:
            url = f"{API_URL}/{endpoint.lstrip('/')}"
            
            if method == "GET":
                resp = client.get(url, params=params)
            elif method == "POST":
                # Handle form data (for file uploads) or JSON
                if files or data:
                    resp = client.post(url, data=data, files=files)
                else:
                    resp = client.post(url, json=params)
            
            resp.raise_for_status()
            return resp.json()
            
    except httpx.TimeoutException:
        st.error("⏳ **Agent Timeout**: The autonomous agents are taking longer than expected (5 mins). "
                 "They might still be working in the background. Check your terminal logs for progress.")
        return None
    except Exception as e:
        st.error(f"❌ API Connection Error: {e}")
        return None

# --- SIDEBAR ---
STAGE_ICONS = {
    "jd_intake": "📋", "candidate_search": "🔍", "scoring": "📊",
    "scheduling": "📅", "feedback": "💬", "offer": "📄",
    "evaluation": "🏁", "completed": "✅", "human_escalation": "🔔",
}

with st.sidebar:
    st.title("🛡️ Recruitment Dashboard")

    # ── All Pipelines Dashboard ──────────────────────────────────
    st.subheader("📂 All Open Roles")
    all_pipelines_data = call_api("GET", "/pipelines") or {}
    all_pipelines = all_pipelines_data.get("pipelines", [])

    if all_pipelines:
        for p in all_pipelines:
            stage = p.get("current_stage", "")
            icon = STAGE_ICONS.get(stage, "⚙️")
            title = p.get("job_title", "Unknown Role")
            pid = p.get("pipeline_id", "")
            is_active = (pid == st.session_state.pipeline_id)
            needs_action = p.get("requires_human_approval", False)

            label = f"{icon} **{title}**"
            if needs_action:
                label += " 🔔"
            if is_active:
                label += " ← *active*"

            with st.expander(label, expanded=is_active):
                st.caption(f"`{pid[:12]}…`")
                cols = st.columns(3)
                cols[0].metric("Candidates", p.get("total_candidates", 0))
                cols[1].metric("Scored", p.get("scored_candidates", 0))
                cols[2].metric("Offers", p.get("offers_drafted", 0))
                st.caption(f"Stage: `{stage}`")
                if not is_active:
                    if st.button("🔀 Switch to this role", key=f"switch_{pid}"):
                        # Load and switch to this pipeline
                        with st.spinner("Switching context..."):
                            detail = call_api("GET", f"/pipelines/{pid}/status")
                            if detail:
                                st.session_state.pipeline_id = pid
                                st.session_state.current_stage = detail.get("current_stage", "jd_intake")
                                st.session_state.requires_approval = detail.get("requires_human_approval", False)
                                st.session_state.last_response = detail.get("response", "")
                                st.session_state.thought = detail.get("thought", "")
                                st.session_state.last_metadata = detail.get("metadata", {})
                        st.rerun()
    else:
        st.info("No active pipelines. Start by providing a Job Description.")

    st.divider()

    # ── Start New Role Button ────────────────────────────────────
    if st.button("➕ Start New Recruitment Role", use_container_width=True):
        # Clear only the active pipeline session — don't wipe the whole state
        st.session_state.pipeline_id = None
        st.session_state.current_stage = "jd_intake"
        st.session_state.last_response = ""
        st.session_state.requires_approval = False
        st.session_state.thought = ""
        st.session_state.last_metadata = {}
        st.rerun()

    st.divider()

    # ── Active Pipeline Details ──────────────────────────────────
    if st.session_state.pipeline_id:
        st.subheader("💰 Active Pipeline Costs")
        md = st.session_state.get("last_metadata", {
            "total_cost_usd": 0.0, "total_tokens": 0, "llm_calls": 0
        })
        col1, col2 = st.columns(2)
        col1.metric("USD Cost", f"${md.get('total_cost_usd', 0.0):.4f}")
        col2.metric("AI Calls", md.get("llm_calls", 0))
        st.write(f"**Tokens:** {md.get('total_tokens', 0):,}")

        if st.session_state.requires_approval:
            st.warning("🔔 **Action Required**: Human-in-the-loop pending approval.")

        # ── Notification Log ─────────────────────────────────────
        notifications = md.get("notifications", [])
        if notifications:
            st.divider()
            st.subheader("📬 Notification Log")
            for n in reversed(notifications[-5:]):  # show last 5
                st.markdown(
                    f"**{n['stage'].replace('_',' ').title()}** → {n['stakeholder']}  \n"
                    f"_{n['event']}_",
                )

        st.divider()
        if st.button("🗑️ Reset This Pipeline", use_container_width=True):
            for key in ["pipeline_id", "current_stage", "last_response",
                        "requires_approval", "thought", "last_metadata"]:
                st.session_state[key] = None if key == "pipeline_id" else (
                    "jd_intake" if key == "current_stage" else (
                    {} if key == "last_metadata" else ""))
            st.rerun()

    st.divider()
    st.caption("🤖 LangGraph Multi-Agent Orchestration  \n7 Specialized Agents | SQLite State Store")


# --- MAIN UI ---
st.title("🤖 Multi-Agent Recruitment Platform")
st.markdown("Automating the full recruitment lifecycle from Intake to Offer.")

# Stage 1: JD Intake
if not st.session_state.pipeline_id:
    st.header("📋 Stage 1: Job Description Intake")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload JD File")
        uploaded_file = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
        
    with col2:
        st.subheader("Provide Natural Language Intake")
        nl_input = st.text_area("Describe the role or paste JD text...", height=200)

    if st.button("🚀 Execute Pipeline Intake"):
        files = None
        data = {}
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        if nl_input:
            data["natural_language"] = nl_input
            
        with st.spinner("🤖 Autonomous Agents searching and scoring candidates..."):
            res = call_api("POST", "/jd/upload", data=data, files=files)
            if res:
                st.session_state.pipeline_id = res['pipeline_id']
                st.session_state.current_stage = res['stage']
                st.session_state.last_response = res['response']
                st.session_state.thought = res.get('thought', "")
                st.session_state.requires_approval = res['requires_human_approval']
                st.session_state.last_metadata = res.get('metadata', {})
                st.rerun()

# Stages Flow
else:
    # 1. Output Display Area
    st.markdown("### 📡 Agent Output")
    
    # New: Agent Thought / Reasoning Section
    if st.session_state.get("thought"):
        with st.expander("🤔 Agent's Internal Reasoning & Thinking Process", expanded=True):
            st.markdown(f"*{st.session_state.thought}*")

    st.markdown(f'<div class="agent-card">{st.session_state.last_response}</div>', unsafe_allow_html=True)
    
    # 2. Dynamic Input Area — shown based on stage
    st.divider()
    stage = st.session_state.current_stage

    # ── A. SCORING APPROVAL ───────────────────────────────────────
    if stage == "scoring" and st.session_state.requires_approval:
        st.header("🎯 Step 2: Approve Candidate Scores")
        st.info("The Scoring Agent has ranked all candidates. Review the results above and approve to proceed to Interview Scheduling.")
        btn_disabled = st.session_state.processing
        if st.button("✅ Approve Scores & Proceed to Interview Scheduling", key="approve_scoring", disabled=btn_disabled):
            st.session_state.processing = True
            with st.spinner("Scheduling interviews..."):
                res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve", data={"decision": "approve"})
            st.session_state.processing = False
            if res:
                st.session_state.current_stage = res['stage']
                st.session_state.last_response = res['response']
                st.session_state.thought = res.get('thought', "")
                st.session_state.requires_approval = res['requires_human_approval']
                st.rerun()

    # ── B. SCHEDULING APPROVAL ────────────────────────────────────
    elif stage == "scheduling" or (st.session_state.requires_approval and "schedul" in stage):
        st.header("📅 Step 3: Confirm Interview Schedule")
        st.info("The Scheduling Agent has proposed interview slots. You can accept them or provide your preferred times below.")
        
        custom_input = st.text_input(
            "📝 Custom preferred times (optional, e.g. 'Monday 10am, Wednesday 3pm'):",
            placeholder="Describe any changes or conflicts here...",
            key="schedule_input_box"
        )
        
        col1, col2 = st.columns(2)
        btn_disabled = st.session_state.processing
        
        with col1:
            if st.button("🔄 Update / Reschedule", key="reschedule_btn", disabled=btn_disabled, use_container_width=True):
                if not custom_input.strip():
                    st.warning("Please provide rescheduling instructions first.")
                else:
                    st.session_state.processing = True
                    with st.spinner("Updating schedule..."):
                        res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve",
                                       data={"decision": "update", "human_input": custom_input})
                    st.session_state.processing = False
                    if res:
                        st.session_state.current_stage = res['stage']
                        st.session_state.last_response = res['response']
                        st.session_state.thought = res.get('thought', "")
                        st.session_state.requires_approval = res['requires_human_approval']
                        st.rerun()

        with col2:
            if st.button("✅ Confirm & Send Invites", key="approve_schedule", disabled=btn_disabled, use_container_width=True, type="primary"):
                st.session_state.processing = True
                with st.spinner("Confirming schedule and sending invites..."):
                    res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve",
                                   data={"decision": "approve", "human_input": "approve"})
                st.session_state.processing = False
                if res:
                    st.session_state.current_stage = res['stage']
                    st.session_state.last_response = res['response']
                    st.session_state.thought = res.get('thought', "")
                    st.session_state.requires_approval = res['requires_human_approval']
                    st.rerun()



    # ── D. FEEDBACK SUBMISSION FORM ────────────────────────────────
    # Shown if we are in feedback stage but analysis hasn't run yet
    elif stage == "feedback":
        st.header("🗣️ Step 4: Interviewer Feedback Submission")
        st.info(
            "Interviews have been scheduled. Once the interview is completed, "
            "the interviewer should fill out this form. The AI will then run "
            "**Sentiment Analysis** to determine the hire recommendation."
        )

        # GET candidates cleanly to populate dropdown
        detail = call_api("GET", f"/pipelines/{st.session_state.pipeline_id}/status")
        cand_list = detail.get("candidates", []) if detail else []
        feedbacks = detail.get("feedbacks", []) if detail else []
        cand_options = {c["id"]: f"{c['name']} ({c['id'][:8]})" for c in cand_list}

        if feedbacks:
            st.success(f"📌 **{len(feedbacks)}** feedback entries have been saved so far.")
            with st.expander("View Saved Feedback Summaries"):
                for fb in feedbacks:
                    st.write(f"- **{fb['interviewer_name']}** for candidate `{fb['candidate_id'][:8]}`")
        else:
            st.warning("⚠️ No feedback entries saved yet for this pipeline.")

        with st.form("feedback_form", clear_on_submit=True):
            if cand_options:
                c_id = st.selectbox("👤 Choose Candidate", options=list(cand_options.keys()), format_func=lambda x: cand_options[x])
            else:
                c_id = st.text_input("🆔 Candidate ID", placeholder="Paste the candidate UUID...")

            i_name = st.text_input("👤 Interviewer Name", placeholder="e.g. John Smith")
            col1, col2, col3, col4 = st.columns(4)
            tr  = col1.slider("🖥️ Technical",        1, 10, 7)
            cr  = col2.slider("💬 Communication",    1, 10, 7)
            cfr = col3.slider("🤝 Culture Fit",      1, 10, 7)
            psr = col4.slider("🧩 Problem Solving",  1, 10, 7)
            fb_text = st.text_area("📝 Detailed Feedback", placeholder="Describe the candidate's performance...", height=150)

            submitted = st.form_submit_button("➕ Save Feedback for Candidate", use_container_width=True)
            if submitted:
                if not c_id.strip() or not fb_text.strip():
                    st.warning("⚠️ Please enter Candidate ID and Feedback text.")
                else:
                    data = {
                        "candidate_id": c_id.strip(),
                        "interviewer_name": i_name.strip() or "Anonymous Interviewer",
                        "technical_rating": tr,
                        "communication_rating": cr,
                        "culture_fit_rating": cfr,
                        "problem_solving_rating": psr,
                        "raw_feedback": fb_text.strip(),
                        "advance_pipeline": "false",
                    }
                    with st.spinner("Saving feedback..."):
                        res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/feedback", data=data)
                    if res:
                        st.session_state.last_response = res['response']
                        st.success(f"✅ Feedback saved for {c_id.strip()}!")
                        st.rerun()

        st.divider()
        st.subheader("🚀 Finalize & Analyze")
        st.write("Click below ONLY after you have saved feedback for all interviewed candidates.")
        
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🧠 Analyze Sentiment & Draft Offer", type="primary", use_container_width=True, disabled=not feedbacks):
                with st.spinner("🧠 AI is analyzing sentiment and drafting offer letters..."):
                    res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve", data={"decision": "approve"})
                    if res:
                        st.session_state.current_stage = res['stage']
                        st.session_state.last_response = res['response']
                        st.session_state.thought = res.get('thought', "")
                        st.session_state.requires_approval = res['requires_human_approval']
                        st.rerun()
        with btn_col2:
             if st.button("📡 Refresh Status", use_container_width=True):
                 st.rerun()

    # ── D. FINAL OFFER REVIEW ──────────────────────────
    elif stage == "offer" and st.session_state.requires_approval:
        st.header("📄 Step 5: Sentiment & Offer Letter Review")
        
        # 1. First, fetch and display the sentiment score gracefully via pipelines status
        detail = call_api("GET", f"/pipelines/{st.session_state.pipeline_id}/status")
        if detail and detail.get("feedbacks"):
            latest_fb = detail["feedbacks"][-1]
            s_val = latest_fb.get("sentiment", "unknown").upper()
            
            if "POSITIVE" in s_val:
                st.success("✅ The AI analyzed interviewer feedback and assigned **POSITIVE** sentiment.")
            elif "NEGATIVE" in s_val:
                st.error("❌ The AI analyzed interviewer feedback and assigned **NEGATIVE** sentiment. Are you sure you want to proceed?")
            elif "NEUTRAL" in s_val:
                st.warning("🟡 The AI analyzed interviewer feedback and assigned **NEUTRAL** sentiment.")
            else:
                st.info("ℹ️ Sentiment analysis complete. Please verify the drafted letter below.")
                
        st.info("The Draft Offer Letter is ready for your final review. Make required adjustments, or approve to send via simulated email.")

        adjust_input = st.text_input(
            "✏️ Offer adjustments (e.g., '+10k signing bonus', 'add remote work clause'):",
            placeholder="Describe any changes you want the AI to make...",
            key="offer_adjust"
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔄 Apply Adjustments", key="update_offer", disabled=st.session_state.processing):
                st.session_state.processing = True
                with st.spinner("Updating offer draft..."):
                    res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve",
                                   data={"decision": "approve", "human_input": adjust_input or "update"})
                st.session_state.processing = False
                if res:
                    st.session_state.current_stage = res['stage']
                    st.session_state.last_response = res['response']
                    st.session_state.thought = res.get('thought', "")
                    st.session_state.requires_approval = res['requires_human_approval']
                    st.session_state.last_metadata = res.get('metadata', {})
                    st.rerun()
        with col_b:
            if st.button("❌ Reject & End Flow", key="reject_sentiment", disabled=st.session_state.processing):
                st.session_state.processing = True
                with st.spinner("Ending pipeline and generating evaluation report..."):
                    res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve",
                                   data={"decision": "approve", "human_input": "reject"})
                st.session_state.processing = False
                if res:
                    st.session_state.current_stage = res['stage']
                    st.session_state.last_response = res['response']
                    st.session_state.thought = res.get('thought', "")
                    st.session_state.requires_approval = res['requires_human_approval']
                    st.session_state.last_metadata = res.get('metadata', {})
                    st.rerun()
        with col_c:
            if st.button("📫 Send Final Offer", key="send_offer", disabled=st.session_state.processing, type="primary"):
                st.session_state.processing = True
                with st.spinner("Sending final offer email..."):
                    res = call_api("POST", f"/pipelines/{st.session_state.pipeline_id}/approve",
                                   data={"decision": "approve", "human_input": "approve"})
                st.session_state.processing = False
                if res:
                    st.session_state.current_stage = res['stage']
                    st.session_state.last_response = res['response']
                    st.session_state.thought = res.get('thought', "")
                    st.session_state.requires_approval = res['requires_human_approval']
                    st.session_state.last_metadata = res.get('metadata', {})
                    st.success("🎉 Offer Sent Successfully!")
                    time.sleep(2)
                    st.rerun()

    # [Note: Removed old Stage 4 block as it's now integrated above]

    # ── E. FINAL EVALUATION & COMPLETION ──────────────────────────
    elif stage in ("evaluation", "completed") or (st.session_state.get("last_metadata", {}).get("is_completed")):
        st.success("🎉 Recruitment Process Completed Successfully!")
        st.balloons()
        
        if st.session_state.last_response:
            st.markdown("### 📊 Pipeline Evaluation Report")
            st.markdown(st.session_state.last_response)
        
        st.divider()
        if st.button("🔄 Start New Recruitment Cycle", key="new_cycle", type="primary", use_container_width=True):
            for key in ["pipeline_id", "current_stage", "last_response", "requires_approval", "thought", "last_metadata"]:
                st.session_state[key] = None if key == "pipeline_id" else ("jd_intake" if key == "current_stage" else ({} if key == "last_metadata" else ""))
            st.rerun()

    # ── F. CATCH-ALL / PROGRESSING STATES ─────────────────────────
    else:
        st.info(f"🚦 **Current Stage**: `{stage.replace('_',' ').title()}`")
        st.markdown("The agents are currently processing or awaiting the next automated trigger.")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("⏩ Proceed to Next Stage", key="manual_proceed", type="primary", use_container_width=True):
                with st.spinner("Orchestrating agents..."):
                    res = call_api("POST", "/chat", data={"pipeline_id": st.session_state.pipeline_id, "message": "proceed"})
                    if res:
                        st.session_state.current_stage = res['stage']
                        st.session_state.last_response = res['response']
                        st.session_state.requires_approval = res['requires_human_approval']
                        st.rerun()
        with col2:
            if st.button("🔄 Refresh Status", key="refresh_status", use_container_width=True):
                detail = call_api("GET", f"/pipelines/{st.session_state.pipeline_id}/status")
                if detail:
                    st.session_state.current_stage = detail.get("current_stage", "jd_intake")
                    st.session_state.requires_approval = detail.get("requires_human_approval", False)
                    st.session_state.last_response = detail.get("response", "")
                    st.session_state.last_metadata = detail.get("metadata", {})
                st.rerun()

    st.divider()
    st.caption(f"Pipeline ID: `{st.session_state.pipeline_id}` | Status: {'🔴 Needs Action' if st.session_state.requires_approval else '🟢 Autonomous'}")
