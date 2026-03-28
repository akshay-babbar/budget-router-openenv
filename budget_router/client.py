from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import Action, EnvState, Observation


class BudgetRouterClient(EnvClient[Action, Observation, EnvState]):
    def _step_payload(self, action: Action) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        observation_payload = payload.get("observation", payload)
        observation = Observation.model_validate(
            {
                **observation_payload,
                "done": payload.get("done", observation_payload.get("done", False)),
                "reward": payload.get("reward", observation_payload.get("reward")),
                "metadata": observation_payload.get("metadata", payload.get("metadata", {})),
            }
        )
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EnvState:
        return EnvState.model_validate(payload)
