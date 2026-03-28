from openenv.core.env_server import create_app
import uvicorn

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, Observation

app = create_app(
    BudgetRouterEnv,
    Action,
    Observation,
    env_name="budget_router",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
