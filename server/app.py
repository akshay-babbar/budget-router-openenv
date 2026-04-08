import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from openenv_core.env_server import create_app, create_fastapi_app

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, Observation


try:
    import gradio as gr
    from app_gradio import build_app
except ImportError:
    gr = None
    build_app = None


# ─── Pydantic request/response schemas ───────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=42, description="RNG seed for reproducibility")
    scenario: Optional[str] = Field(
        default="easy",
        description="Scenario name: easy | medium | hard | hard_multi",
    )


class StepRequest(BaseModel):
    action_type: str = Field(
        description="One of: route_to_a | route_to_b | route_to_c | shed_load"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObservationResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool


# ─── Environment ──────────────────────────────────────────────────────────────

env = BudgetRouterEnv(emit_structured_logs=True)
_executor = ThreadPoolExecutor(max_workers=1)

# ─── App factory ──────────────────────────────────────────────────────────────

if os.getenv("ENABLE_OPENENV_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"}:
    _base_app = create_app(env, Action, Observation)
else:
    _base_app = create_fastapi_app(env, Action, Observation)

# We build a *new* FastAPI app that delegates Gradio mounting to the base but
# overrides /reset and /step with correct schemas and seed propagation.
app = FastAPI(title="Budget Router Environment API")


def _serialize(obs: Observation) -> Dict[str, Any]:
    """Flatten dataclass → JSON, strip reward/done/metadata from obs body."""
    obs_dict = asdict(obs)
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)
    return {"observation": obs_dict, "reward": reward, "done": done}


@app.post("/reset", response_model=ObservationResponse, summary="Reset the environment")
async def reset(body: ResetRequest) -> Dict[str, Any]:
    """
    Reset the environment.

    - **seed**: integer seed for deterministic episodes (propagated to the RNG)
    - **scenario**: one of `easy`, `medium`, `hard`, `hard_multi`
    """
    loop = asyncio.get_event_loop()
    obs = await loop.run_in_executor(
        _executor,
        lambda: env.reset(seed=body.seed, scenario=body.scenario),
    )
    return _serialize(obs)


@app.post("/step", response_model=ObservationResponse, summary="Execute one step")
async def step(body: StepRequest) -> Dict[str, Any]:
    """
    Execute one environment step.

    - **action_type**: `route_to_a` | `route_to_b` | `route_to_c` | `shed_load`
    """
    action = Action(action_type=body.action_type, metadata=body.metadata)
    loop = asyncio.get_event_loop()
    obs = await loop.run_in_executor(_executor, lambda: env.step(action))
    return _serialize(obs)


@app.get("/state", summary="Get current environment state")
async def get_state() -> Dict[str, Any]:
    return asdict(env.state)


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


# ─── Gradio dashboard ─────────────────────────────────────────────────────────

if (
    gr is not None
    and build_app is not None
    and os.getenv("ENABLE_GRADIO_DASHBOARD", "true").lower() in {"true", "1", "yes"}
):
    app = gr.mount_gradio_app(app, build_app(), path="/web")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    resolved_port = port or int(os.getenv("PORT", "8000"))
    uvicorn.run(
        app,
        host=host,
        port=resolved_port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
