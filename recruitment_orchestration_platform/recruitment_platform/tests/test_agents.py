"""
Test suite for the Recruitment Orchestration Platform.
Tests agent logic, scoring, conflict detection, and pipeline state transitions.

Run with: pytest tests/test_agents.py -v
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from models.schemas import (
    Candidate, CandidateStatus, CandidateScore,
    JobDescription, InterviewFeedback, FeedbackSentiment,
    InterviewSchedule, InterviewStatus, PipelineState, PipelineStage,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_jd():
    return JobDescription(
        job_id="jd-test-001",
        title="Senior Python Engineer",
        department="Engineering",
        location="Bangalore, India",
        employment_type="Full-time",
        required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        preferred_skills=["Kubernetes", "Redis", "LangChain"],
        experience_years=5,
        education_requirement="B.Tech Computer Science",
        responsibilities=["Build backend APIs", "Design microservices"],
        salary_range="18-25 LPA",
    )


@pytest.fixture
def strong_candidate():
    return Candidate(
        candidate_id="cand-test-001",
        name="Arjun Sharma",
        email="arjun@test.com",
        current_role="Senior Software Engineer",
        years_of_experience=6,
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "Redis"],
        education="B.Tech Computer Science - IIT Bombay",
        location="Bangalore, India",
        status=CandidateStatus.SCREENED,
    )


@pytest.fixture
def weak_candidate():
    return Candidate(
        candidate_id="cand-test-002",
        name="John Doe",
        email="john@test.com",
        current_role="Junior Developer",
        years_of_experience=1,
        skills=["HTML", "CSS", "JavaScript"],
        education="Diploma IT",
        location="Remote",
        status=CandidateStatus.SOURCED,
    )


@pytest.fixture
def pipeline(sample_jd):
    p = PipelineState()
    p.job_description = sample_jd
    p.job_id = sample_jd.job_id
    p.current_stage = PipelineStage.SCORING
    return p


# ──────────────────────────────────────────────────────────────
# 1. Scoring Agent tests
# ──────────────────────────────────────────────────────────────

class TestScoringAgent:

    def setup_method(self):
        from agents.scoring_agent import ScoringAgent
        self.agent = ScoringAgent.__new__(ScoringAgent)
        # Mock LLM to avoid API calls
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(
            content="Strong candidate with excellent Python expertise. Recommend proceeding to interview."
        )
        self.agent.system_prompt = "mock prompt"

    def test_skill_score_perfect_match(self, strong_candidate, sample_jd):
        score, matched, missing = self.agent._score_skills(
            strong_candidate.skills, sample_jd.required_skills
        )
        assert score == 100.0
        assert len(missing) == 0
        assert "python" in matched

    def test_skill_score_partial_match(self, weak_candidate, sample_jd):
        score, matched, missing = self.agent._score_skills(
            weak_candidate.skills, sample_jd.required_skills
        )
        assert score == 0.0
        assert len(matched) == 0
        assert len(missing) == len(sample_jd.required_skills)

    def test_experience_score_exact(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        assert agent._score_experience(5, 5) == 100.0

    def test_experience_score_one_year_off(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        assert agent._score_experience(6, 5) == 85.0
        assert agent._score_experience(4, 5) == 85.0

    def test_experience_score_large_gap(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        # diff = 4 → hits the "<=4" bracket → 45.0
        assert agent._score_experience(1, 5) == 45.0
        # diff = 5 → truly large gap → 25.0
        assert agent._score_experience(0, 5) == 25.0

    def test_overall_score_strong_candidate(self, strong_candidate, sample_jd):
        score = self.agent.score_candidate(strong_candidate, sample_jd)
        assert score.overall_score >= 75.0
        assert score.skill_match_score == 100.0

    def test_overall_score_weak_candidate(self, weak_candidate, sample_jd):
        score = self.agent.score_candidate(weak_candidate, sample_jd)
        assert score.overall_score < 40.0

    def test_score_label_strong(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        assert "Strong" in agent._get_label(85.0)

    def test_score_label_moderate(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        assert "Moderate" in agent._get_label(50.0)

    def test_score_label_weak(self):
        from agents.scoring_agent import ScoringAgent
        agent = ScoringAgent.__new__(ScoringAgent)
        assert "Weak" in agent._get_label(30.0)


# ──────────────────────────────────────────────────────────────
# 2. Candidate Search Agent tests
# ──────────────────────────────────────────────────────────────

class TestCandidateSearchAgent:

    def setup_method(self):
        from agents.candidate_search_agent import CandidateSearchAgent
        self.agent = CandidateSearchAgent.__new__(CandidateSearchAgent)
        self.agent.llm = MagicMock()
        self.agent.llm.invoke.return_value = MagicMock(content="3 strong matches found.")
        self.agent.system_prompt = "mock prompt"
        self.agent.MIN_SKILL_MATCH_RATIO = 0.40

    def test_skill_overlap_full_match(self, strong_candidate, sample_jd):
        matched, missing, ratio = self.agent._skill_overlap(
            strong_candidate.skills, sample_jd.required_skills
        )
        assert ratio == 1.0
        assert len(missing) == 0

    def test_skill_overlap_no_match(self, weak_candidate, sample_jd):
        matched, missing, ratio = self.agent._skill_overlap(
            weak_candidate.skills, sample_jd.required_skills
        )
        assert ratio == 0.0
        assert len(matched) == 0

    def test_experience_match_within_range(self):
        assert self.agent._experience_match(5, 5) is True
        assert self.agent._experience_match(6, 5) is True
        assert self.agent._experience_match(3, 5) is True

    def test_experience_match_out_of_range(self):
        assert self.agent._experience_match(1, 8) is False

    def test_search_returns_matching_candidates(self, sample_jd):
        results = self.agent.search(sample_jd)
        assert len(results) > 0
        # All results should be above MIN_SKILL_MATCH_RATIO
        for _, _, _, ratio in results:
            assert ratio >= self.agent.MIN_SKILL_MATCH_RATIO

    def test_search_results_sorted_by_ratio(self, sample_jd):
        results = self.agent.search(sample_jd)
        ratios = [r[3] for r in results]
        assert ratios == sorted(ratios, reverse=True)

    def test_classify_strong_match(self):
        assert "Strong" in self.agent._classify_match(0.80)

    def test_classify_moderate_match(self):
        assert "Moderate" in self.agent._classify_match(0.60)

    def test_classify_stretch_match(self):
        assert "Stretch" in self.agent._classify_match(0.45)


# ──────────────────────────────────────────────────────────────
# 3. Scheduling Agent — conflict detection tests
# ──────────────────────────────────────────────────────────────

class TestSchedulingAgent:

    def setup_method(self):
        from agents.scheduling_agent import SchedulingAgent
        self.agent = SchedulingAgent.__new__(SchedulingAgent)
        self.agent.llm = MagicMock()
        self.agent.system_prompt = "mock prompt"
        self.agent._booked_slots = {}

    def test_business_hours_valid(self):
        # Monday 10:00 AM
        slot = datetime(2025, 7, 7, 10, 0)  # Monday
        assert self.agent._is_business_hours(slot) is True

    def test_business_hours_weekend(self):
        # Saturday
        slot = datetime(2025, 7, 5, 10, 0)
        assert self.agent._is_business_hours(slot) is False

    def test_business_hours_after_hours(self):
        slot = datetime(2025, 7, 7, 20, 0)
        assert self.agent._is_business_hours(slot) is False

    def test_no_conflict_empty_schedule(self):
        start = datetime(2025, 7, 7, 10, 0)
        end = start + timedelta(hours=1)
        conflict = self.agent._has_conflict("interviewer@test.com", start, end)
        assert conflict is None

    def test_conflict_overlapping_slots(self):
        email = "interviewer@test.com"
        start1 = datetime(2025, 7, 7, 10, 0)
        end1 = start1 + timedelta(hours=1)
        self.agent._book_slot(email, start1, end1)

        # Try to book overlapping slot
        start2 = datetime(2025, 7, 7, 10, 30)
        end2 = start2 + timedelta(hours=1)
        conflict = self.agent._has_conflict(email, start2, end2)
        assert conflict is not None
        assert len(conflict) > 0

    def test_conflict_buffer_respected(self):
        email = "interviewer@test.com"
        start1 = datetime(2025, 7, 7, 10, 0)
        end1 = start1 + timedelta(hours=1)
        self.agent._book_slot(email, start1, end1)

        # Slot immediately after — within buffer
        start2 = datetime(2025, 7, 7, 11, 5)
        end2 = start2 + timedelta(hours=1)
        conflict = self.agent._has_conflict(email, start2, end2)
        assert conflict is not None  # within 15-min buffer

    def test_no_conflict_after_buffer(self):
        email = "interviewer@test.com"
        start1 = datetime(2025, 7, 7, 10, 0)
        end1 = start1 + timedelta(hours=1)
        self.agent._book_slot(email, start1, end1)

        # Slot well after buffer
        start2 = datetime(2025, 7, 7, 11, 30)
        end2 = start2 + timedelta(hours=1)
        conflict = self.agent._has_conflict(email, start2, end2)
        assert conflict is None

    def test_propose_alternatives_returns_3(self):
        base = datetime(2025, 7, 7, 9, 0)
        alternatives = self.agent._propose_alternatives("test@test.com", base, count=3)
        assert len(alternatives) == 3

    def test_propose_alternatives_are_business_hours(self):
        base = datetime(2025, 7, 7, 9, 0)
        alternatives = self.agent._propose_alternatives("test@test.com", base, count=3)
        for alt in alternatives:
            assert self.agent._is_business_hours(alt)


# ──────────────────────────────────────────────────────────────
# 4. Feedback Agent tests
# ──────────────────────────────────────────────────────────────

class TestFeedbackAgent:

    def setup_method(self):
        from agents.feedback_agent import FeedbackAgent
        self.agent = FeedbackAgent.__new__(FeedbackAgent)
        self.agent.llm = MagicMock()
        self.agent.system_prompt = "mock prompt"

    def _make_feedback(self, tech, comm, culture, problem):
        return InterviewFeedback(
            candidate_id="cand-001",
            job_id="jd-001",
            interviewer_name="Test Interviewer",
            technical_rating=tech,
            communication_rating=comm,
            culture_fit_rating=culture,
            problem_solving_rating=problem,
            raw_feedback="Test feedback",
        )

    def test_aggregate_score_perfect(self):
        fb = self._make_feedback(10, 10, 10, 10)
        score = self.agent._aggregate_score(fb)
        assert score == 100.0

    def test_aggregate_score_average(self):
        fb = self._make_feedback(7, 7, 7, 7)
        score = self.agent._aggregate_score(fb)
        assert score == 70.0

    def test_aggregate_score_below_threshold(self):
        fb = self._make_feedback(3, 3, 3, 3)
        score = self.agent._aggregate_score(fb)
        assert score == 30.0

    def test_hire_recommendation_strong(self):
        hire, label = self.agent._hire_recommendation(80.0, FeedbackSentiment.POSITIVE)
        assert hire is True
        assert "Strong" in label

    def test_hire_recommendation_conditions(self):
        hire, label = self.agent._hire_recommendation(60.0, FeedbackSentiment.NEUTRAL)
        assert hire is True
        assert "Conditions" in label

    def test_hire_recommendation_do_not_hire(self):
        hire, label = self.agent._hire_recommendation(30.0, FeedbackSentiment.NEGATIVE)
        assert hire is False
        assert "Do Not Hire" in label

    def test_hire_recommendation_on_the_fence(self):
        hire, label = self.agent._hire_recommendation(50.0, FeedbackSentiment.NEUTRAL)
        assert hire is None
        assert "Fence" in label


# ──────────────────────────────────────────────────────────────
# 5. Pipeline State tests
# ──────────────────────────────────────────────────────────────

class TestPipelineState:

    def test_pipeline_creation(self):
        p = PipelineState()
        assert p.pipeline_id is not None
        assert p.current_stage == PipelineStage.JD_INTAKE
        assert p.candidates == []

    def test_add_message(self):
        p = PipelineState()
        p.add_message("user", "Hello")
        assert len(p.messages) == 1
        assert p.messages[0]["role"] == "user"

    def test_advance_stage(self):
        p = PipelineState()
        assert p.current_stage == PipelineStage.JD_INTAKE
        p.advance_stage(PipelineStage.CANDIDATE_SEARCH)
        assert p.current_stage == PipelineStage.CANDIDATE_SEARCH
        assert PipelineStage.JD_INTAKE.value in p.stage_history

    def test_stage_history_tracks_transitions(self):
        p = PipelineState()
        p.advance_stage(PipelineStage.CANDIDATE_SEARCH)
        p.advance_stage(PipelineStage.SCORING)
        assert len(p.stage_history) == 2

    def test_offer_agent_salary_parsing(self):
        from agents.offer_agent import OfferAgent
        agent = OfferAgent.__new__(OfferAgent)
        agent.llm = MagicMock()
        agent.system_prompt = "mock"
        min_s, max_s = agent._parse_salary_range("18-25 LPA")
        assert min_s == 18.0
        assert max_s == 25.0


# ──────────────────────────────────────────────────────────────
# 6. State Store tests
# ──────────────────────────────────────────────────────────────

class TestPipelineStore:

    def setup_method(self):
        from core.state_store import PipelineStore
        self.store = PipelineStore()

    def test_create_and_get(self):
        p = PipelineState()
        self.store.create(p)
        retrieved = self.store.get(p.pipeline_id)
        assert retrieved is not None
        assert retrieved.pipeline_id == p.pipeline_id

    def test_get_nonexistent(self):
        result = self.store.get("nonexistent-id")
        assert result is None

    def test_update(self):
        p = PipelineState()
        self.store.create(p)
        p.advance_stage(PipelineStage.CANDIDATE_SEARCH)
        self.store.update(p)
        retrieved = self.store.get(p.pipeline_id)
        assert retrieved.current_stage == PipelineStage.CANDIDATE_SEARCH

    def test_delete(self):
        p = PipelineState()
        self.store.create(p)
        deleted = self.store.delete(p.pipeline_id)
        assert deleted is True
        assert self.store.get(p.pipeline_id) is None

    def test_list_all(self):
        p1, p2 = PipelineState(), PipelineState()
        self.store.create(p1)
        self.store.create(p2)
        all_pipelines = self.store.list_all()
        ids = [p.pipeline_id for p in all_pipelines]
        assert p1.pipeline_id in ids
        assert p2.pipeline_id in ids
