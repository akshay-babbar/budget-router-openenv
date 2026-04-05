import os

import uvicorn

from openenv_core.env_server.web_interface import create_web_interface_app

from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, Observation


try:
    import gradio as gr
    from app_gradio import build_app
except ImportError:
    gr = None
    build_app = None

app = create_web_interface_app(
    BudgetRouterEnv(emit_structured_logs=True),
    Action,
    Observation
)

if gr is not None and build_app is not None and os.getenv("ENABLE_GRADIO_DASHBOARD", "true").lower() in {"true", "1", "yes"}:
    app = gr.mount_gradio_app(app, build_app(), path="/dashboard")



@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    web_path = "/web" if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"} else "/docs"
    payload = {"status": "ok", "web": web_path, "health": "/health"}
    if gr is not None and build_app is not None and os.getenv("ENABLE_GRADIO_DASHBOARD", "true").lower() in {"true", "1", "yes"}:
        payload["dashboard"] = "/dashboard"
    return payload


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
