from inference import SYSTEM_PROMPT
from budget_router.models import Observation
from inference import LLMRouter


def test_system_prompt_has_required_structural_sections():
    upper_prompt = SYSTEM_PROMPT.upper()
    assert "GOLDEN RULE" in upper_prompt or "DEFAULT STRATEGY" in upper_prompt
    assert "BUDGET RUNWAY" in upper_prompt
    assert "TASK PROFILE" in upper_prompt
    assert "NOISE CALIBRATION" in upper_prompt


def test_system_prompt_communicates_bankruptcy_consequence():
    assert "-10" in SYSTEM_PROMPT or "bankruptcy" in SYSTEM_PROMPT.lower()
    assert "0.500" in SYSTEM_PROMPT or "unobserved" in SYSTEM_PROMPT.lower()


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class _FakeClient:
    def with_options(self, **kwargs):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return _FakeResponse("route_to_a")


def test_llm_router_preserves_task_name_on_first_step():
    router = LLMRouter(api_base_url="https://example.com/v1", model_name="test-model", api_key="test-key")
    router._client = _FakeClient()
    router.reset(task_name="hard_multi")

    obs = Observation(
        provider_a_status=0.5,
        provider_b_status=0.5,
        provider_c_status=0.5,
        budget_remaining=1.0,
        queue_backlog=0.0,
        system_latency=0.2,
        step_count=0.0,
    )

    router.choose_action(obs)

    assert router._task_name == "hard_multi"
    assert "task: hard_multi" in router._messages[-2]["content"]


def test_objective_feedback_mode_includes_previous_step_feedback():
    router = LLMRouter(
        api_base_url="https://example.com/v1",
        model_name="test-model",
        api_key="test-key",
        prompt_mode="objective_feedback",
    )
    router._client = _FakeClient()
    router.reset(task_name="hard_multi")

    obs = Observation(
        provider_a_status=0.4,
        provider_b_status=0.7,
        provider_c_status=0.9,
        budget_remaining=0.8,
        queue_backlog=0.1,
        system_latency=0.4,
        step_count=0.5,
        reward=-2.05,
        metadata={
            "action_type": "route_to_a",
            "request_succeeded": False,
            "cost": 0.01,
            "latency_ms": 620.0,
        },
    )

    router.choose_action(obs)

    prompt = router._messages[-2]["content"]
    assert "previous_step_feedback:" in prompt
    assert "previous_action: route_to_a" in prompt
    assert "previous_reward: -2.05" in prompt
    assert "previous_success: false" in prompt
