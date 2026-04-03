from openenv_core.env_server import create_app
import os
import uvicorn

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, Observation

app = create_app(
    BudgetRouterEnv(emit_structured_logs=True),
    Action,
    Observation,
    env_name="budget_router",
)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    web_path = "/web" if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"} else "/docs"
    return {"status": "ok", "web": web_path, "health": "/health"}


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
