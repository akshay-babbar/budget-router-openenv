import os

import uvicorn

from openenv_core.env_server.web_interface import create_web_interface_app

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, Observation

app = create_web_interface_app(
    BudgetRouterEnv(emit_structured_logs=True),
    Action,
    Observation,
)


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
