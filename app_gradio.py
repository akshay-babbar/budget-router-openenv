"""
Budget Router — Gradio Visualization Dashboard
Run: python app_gradio.py  (launches on http://localhost:7860)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Protocol

import gradio as gr
import requests
from budget_router.reward import grade_episode

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000"
AUTO_PLAY_DELAY = 0.5
MAX_STEPS = 20
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

SCENARIOS = ["easy", "medium", "hard", "hard_multi"]
ACTION_CHOICES = [
    ("🟢 route_to_a ($0.01)", "route_to_a"),
    ("🟡 route_to_b ($0.05)", "route_to_b"),
    ("🔴 route_to_c ($0.10)", "route_to_c"),
    ("⚪ shed_load (-0.5)", "shed_load"),
]
VALID_ACTIONS = [choice[1] for choice in ACTION_CHOICES]

# Minimal CSS overrides - theme handles most styling via _dark variants
LIGHT_CSS = """
/* Force light color scheme */
:root, .dark {
  color-scheme: light !important;
}

/* Ensure all text is dark */
.gradio-container label,
.gradio-container span {
  color: #1f2937 !important;
}

/* Radio/checkbox label pills - force white background */
.gradio-container .wrap[data-testid="checkbox-group"] label,
.gradio-container .wrap[data-testid="radio-group"] label {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  color: #1f2937 !important;
}
.gradio-container .wrap[data-testid="checkbox-group"] label.selected,
.gradio-container .wrap[data-testid="radio-group"] label.selected {
  background: #eef2ff !important;
  border-color: #4f46e5 !important;
}
"""

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.gray,
).set(
    # Body
    body_background_fill="#f7f8fb",
    body_background_fill_dark="#f7f8fb",
    body_text_color="#1f2937",
    body_text_color_dark="#1f2937",
    # Blocks
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e5e7eb",
    block_border_color_dark="#e5e7eb",
    block_label_text_color="#1f2937",
    block_label_text_color_dark="#1f2937",
    block_title_text_color="#1f2937",
    block_title_text_color_dark="#1f2937",
    # Inputs
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#9ca3af",
    input_border_color_dark="#9ca3af",
    input_placeholder_color="#6b7280",
    input_placeholder_color_dark="#6b7280",
    # Buttons
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_dark="#4f46e5",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#4f46e5",
    button_secondary_background_fill_dark="#4f46e5",
    button_secondary_text_color="#ffffff",
    button_secondary_text_color_dark="#ffffff",
    button_secondary_background_fill_hover="#4338ca",
    button_secondary_background_fill_hover_dark="#4338ca",
    # Radio/Checkbox - THIS IS THE KEY FIX
    checkbox_background_color="#ffffff",
    checkbox_background_color_dark="#ffffff",
    checkbox_border_color="#d1d5db",
    checkbox_border_color_dark="#d1d5db",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_dark="#ffffff",
    checkbox_label_background_fill_hover="#f3f4f6",
    checkbox_label_background_fill_hover_dark="#f3f4f6",
    checkbox_label_background_fill_selected="#eef2ff",
    checkbox_label_background_fill_selected_dark="#eef2ff",
    checkbox_label_border_color="#e5e7eb",
    checkbox_label_border_color_dark="#e5e7eb",
    checkbox_label_text_color="#1f2937",
    checkbox_label_text_color_dark="#1f2937",
)

# ─── API Client ───────────────────────────────────────────────────────────────

class APIClient:
    """Single-responsibility HTTP client for the OpenEnv Budget Router API."""

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, body: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            r = requests.post(f"{self.base_url}{path}", json=body, timeout=15)
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            return None, str(exc)

    def _get(self, path: str) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            r = requests.get(f"{self.base_url}{path}", timeout=10)
            r.raise_for_status()
            return r.json(), None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _normalize(payload: Dict) -> Tuple[Dict, float, Dict, bool]:
        """Handle both flat and observation-wrapped response shapes."""
        obs = payload.get("observation", payload)
        reward = float(payload.get("reward", obs.get("reward", 0.0)) or 0.0)
        meta = payload.get("metadata", obs.get("metadata", {})) or {}
        done = bool(payload.get("done", obs.get("done", False)))
        return obs, reward, meta, done

    def reset(self, seed: int, scenario: str) -> Tuple[Optional[Dict], Optional[str]]:
        data, err = self._post("/reset", {"seed": seed, "scenario": scenario})
        if err:
            return None, err
        obs, _, _, _ = self._normalize(data)
        return obs, None

    def step(self, action_type: str) -> Tuple[Optional[Tuple], Optional[str]]:
        data, err = self._post("/step", {"action_type": action_type})
        if err:
            return None, err
        return self._normalize(data), None

    def state(self) -> Tuple[Optional[Dict], Optional[str]]:
        return self._get("/state")


client = APIClient()

# ─── Policies ─────────────────────────────────────────────────────────────────

class Policy(Protocol):
    def choose_action(self, obs: Dict) -> str:
        ...


class HeuristicPolicy:
    threshold = 0.52

    def choose_action(self, obs: Dict) -> str:
        providers = [
            ("route_to_a", obs.get("provider_a_status", 0)),
            ("route_to_b", obs.get("provider_b_status", 0)),
            ("route_to_c", obs.get("provider_c_status", 0)),
        ]
        candidates = providers[:2] if obs.get("budget_remaining", 1) < 0.10 else providers
        for action, status in candidates:
            if status > self.threshold:
                return action
        return "shed_load"


class LLMPolicy:
    """Thin adapter around an OpenAI-compatible chat completion endpoint."""

    SYSTEM_PROMPT = """You are an incident-response routing agent controlling a
production service. At each step you observe provider health metrics and must
choose exactly one action.

Valid actions — respond with ONLY one of these four strings, nothing else:
  route_to_a
  route_to_b
  route_to_c
  shed_load

Decision rules:
- Prefer providers with windowed success rate above 0.52 and lower cost.
- If budget_remaining is below 0.20, prefer cheaper providers or shed_load.
- shed_load reduces backlog but costs -0.5 reward — use it when all providers
  are degraded, not as a default.
- Never respond with anything other than one of the four action strings above."""

    @staticmethod
    def _parse_action(response_text: str) -> str:
        text = response_text.strip().lower()
        for action in VALID_ACTIONS:
            if action in text:
                return action
        return "shed_load"

    def __init__(self, api_base_url: str, model_name: str, api_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed. Run `pip install openai`.")
        self._client = OpenAI(base_url=api_base_url, api_key=api_key)
        self._model_name = model_name

    def choose_action(self, obs: Dict) -> str:
        obs_text = "\n".join([
            f"provider_a_status: {obs.get('provider_a_status', 0):.3f}",
            f"provider_b_status: {obs.get('provider_b_status', 0):.3f}",
            f"provider_c_status: {obs.get('provider_c_status', 0):.3f}",
            f"budget_remaining:  {obs.get('budget_remaining', 0):.3f}",
            f"queue_backlog:     {obs.get('queue_backlog', 0):.3f}",
            f"system_latency:    {obs.get('system_latency', 0):.3f}",
            f"step_count:        {obs.get('step_count', 0):.3f}",
        ])
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Current observation:\n{obs_text}\n\nYour action:"},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        return self._parse_action(raw)


def get_policy_runner(policy_name: str) -> Tuple[Optional[Policy], Optional[str]]:
    if policy_name == "heuristic":
        return HeuristicPolicy(), None
    if not HF_TOKEN:
        return None, "HF_TOKEN is required for LLM simulation."
    try:
        return LLMPolicy(api_base_url=API_BASE_URL, model_name=MODEL_NAME, api_key=HF_TOKEN), None
    except Exception as exc:
        return None, str(exc)

# ─── Grade Computation ────────────────────────────────────────────────────────

def compute_grade(history: List[Dict]) -> Dict[str, float]:
    canonical_history = [
        {
            "step": h.get("step", 0),
            "action_type": h.get("action", "shed_load"),
            "request_succeeded": h.get("succeeded", False),
            "cost": h.get("cost", 0.0),
            "latency_ms": h.get("latency_ms", 0.0),
            "reward": h.get("reward", 0.0),
            "sla_ceiling_ms": h.get("sla_ceiling_ms", 500.0),
            "initial_budget": h.get("initial_budget", 1.0),
            "degradation_start_step": h.get("degradation_start_step", 999),
            "secondary_degradation_start_step": h.get("secondary_degradation_start_step"),
        }
        for h in history
    ]
    return grade_episode(canonical_history)

# ─── HTML Renderers ───────────────────────────────────────────────────────────

_TABLE_STYLE = "width:100%;border-collapse:collapse;font-size:13px;background:#ffffff;color:#111827"
_HEADER_ROW_STYLE = "background:#fafafa"
_HEADER_CELL_STYLE = "border:1px solid #eee;padding:6px;color:#111827"
_CELL_STYLE = "border:1px solid #eee;padding:6px;text-align:center;color:#111827;background:#ffffff"
_ACTION_CELL_STYLE = "border:1px solid #eee;padding:6px;font-weight:600;color:#111827"

def _join(parts: List[str]) -> str:
    return "".join(parts)

def _th(value: str) -> str:
    return f"<th style='{_HEADER_CELL_STYLE}'>{value}</th>"

def _td(value: str, style: str, colspan: Optional[int] = None) -> str:
    span = f" colspan='{colspan}'" if colspan else ""
    return f"<td{span} style='{style}'>{value}</td>"

def _tr(cells: List[str], style: Optional[str] = None) -> str:
    style_attr = f" style='{style}'" if style else ""
    return f"<tr{style_attr}>" + "".join(cells) + "</tr>"

def _table(head: str, body: str) -> str:
    return f"<table style='{_TABLE_STYLE}'><thead>{head}</thead><tbody>{body}</tbody></table>"

def _bar(value: float, label: str, color: str, show_dollar: bool = False) -> str:
    pct = max(0, min(100, int(value * 100)))
    display = f"${value:.2f}" if show_dollar else f"{pct}%"
    return (
        f'<div style="margin:5px 0;color:#111827">'
        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px;color:#111827">'
        f'<b>{label}</b><span>{display}</span></div>'
        f'<div style="background:#e0e0e0;border-radius:5px;height:20px">'
        f'<div style="width:{pct}%;background:{color};border-radius:5px;height:100%;'
        f'transition:width 0.3s ease"></div></div></div>'
    )

def _budget_color(remaining: float) -> str:
    if remaining > 0.5:  return "#27ae60"
    if remaining > 0.2:  return "#f39c12"
    return "#e74c3c"

def render_providers(obs: Dict) -> str:
    return _join([
        _bar(obs.get("provider_a_status", 0), "Provider A", "#27ae60"),
        _bar(obs.get("provider_b_status", 0), "Provider B", "#e67e22"),
        _bar(obs.get("provider_c_status", 0), "Provider C", "#e74c3c"),
    ])

def render_budget(obs: Dict) -> str:
    b = obs.get("budget_remaining", 1.0)
    return _bar(b, "Budget Remaining", _budget_color(b))

def render_grader(grade: Dict) -> str:
    components = [
        ("Success Score",    grade["success_score"],    "#27ae60"),
        ("Latency Score",    grade["latency_score"],    "#3498db"),
        ("Budget Score",     grade["budget_score"],     "#f39c12"),
        ("SLA Score",        grade["sla_score"],        "#9b59b6"),
        ("Adaptation Score", grade["adaptation_score"], "#e74c3c"),
    ]
    bars = _join([_bar(v, l, c) for l, v, c in components])
    o = grade["overall_score"]
    color = "#27ae60" if o > 0.7 else "#f39c12" if o > 0.4 else "#e74c3c"
    overall_block = (
        f'<div style="text-align:center;font-size:28px;font-weight:bold;'
        f'color:{color};margin-top:12px;padding:8px;border-radius:8px;'
        f'background:rgba(0,0,0,0.04)">Overall Score: {o:.1%}</div>'
    )
    return bars + overall_block

def _GRADER_PENDING() -> str:
    return "<div style='color:#aaa;font-style:italic'>Shown when episode completes.</div>"

def _PROVIDER_EMPTY() -> str:
    return "<div style='color:#aaa;font-style:italic'>Start an episode to see provider health.</div>"

# ─── State Helpers ────────────────────────────────────────────────────────────

def fresh_state() -> Dict:
    return {"obs": {}, "history": [], "cumulative_reward": 0.0, "step": 0, "done": False}


def apply_step_result(state: Dict, action: str, result: Tuple[Dict, float, Dict, bool]) -> Tuple[Dict, Dict, float, bool]:
    obs, reward, meta, done = result
    state["step"] += 1
    state["cumulative_reward"] += reward
    state["history"].append(record_step(state["step"], action, obs, reward, meta))
    state["obs"] = obs
    state["done"] = done
    return state, obs, reward, done


def format_step_status(prefix: str, step: int, action: str, reward: float, done: bool) -> str:
    return (
        f"{prefix} Step {step} · {action} → {reward:+.3f}"
        + (" · DONE" if done else "")
    )


POLICY_META = {
    "heuristic": {"label": "Heuristic", "icon": "⚡"},
    "llm": {"label": "LLM", "icon": "🤖"},
}


def reset_episode(scenario: str, seed: float) -> Tuple[Dict, Optional[Dict], Optional[str]]:
    obs, err = client.reset(int(seed), scenario)
    if err:
        return fresh_state(), None, err
    state = fresh_state()
    state["obs"] = obs
    return state, obs, None


def step_episode(action: str, state: Dict) -> Tuple[Optional[Tuple[Dict, Dict, float, bool]], Optional[str]]:
    result, err = client.step(action)
    if err:
        return None, err
    return apply_step_result(state, action, result), None

def record_step(step: int, action: str, obs: Dict, reward: float, meta: Dict) -> Dict:
    return {
        "step": step,
        "action": action,
        "health_a": obs.get("provider_a_status", 0),
        "health_b": obs.get("provider_b_status", 0),
        "health_c": obs.get("provider_c_status", 0),
        "budget": obs.get("budget_remaining", 0),
        "reward": reward,
        "succeeded": meta.get("request_succeeded", False),
        "cost": meta.get("cost", 0.0),
        "latency_ms": meta.get("latency_ms", 0.0),
        "sla_ceiling_ms": meta.get("sla_ceiling_ms", 500.0),
        "initial_budget": meta.get("initial_budget", 1.0),
        "degradation_start_step": meta.get("degradation_start_step", 999),
        "secondary_degradation_start_step": meta.get("secondary_degradation_start_step"),
    }

_ACTION_COLORS = {
    "route_to_a": "#d5f5e3",
    "route_to_b": "#fef9e7",
    "route_to_c": "#fadbd8",
    "shed_load":  "#f2f3f4",
}

def render_history_table(history: List[Dict]) -> str:
    headers = ["Step", "Action", "Health A", "Health B", "Health C", "Budget", "Reward"]
    head = _tr([_th(h) for h in headers], style=_HEADER_ROW_STYLE)
    if not history:
        body = _tr([
            _td("No steps yet.", "padding:8px;color:#888;text-align:center", colspan=len(headers))
        ])
    else:
        rows = []
        for h in history:
            action = h["action"]
            action_color = _ACTION_COLORS.get(action, "#f2f3f4")
            rows.append(_tr([
                _td(str(h["step"]), _CELL_STYLE),
                _td(action, f"{_ACTION_CELL_STYLE};background:{action_color}"),
                _td(f"{h['health_a']:.2f}", _CELL_STYLE),
                _td(f"{h['health_b']:.2f}", _CELL_STYLE),
                _td(f"{h['health_c']:.2f}", _CELL_STYLE),
                _td(f"{h['budget']:.2f}", _CELL_STYLE),
                _td(f"{h['reward']:+.3f}", _CELL_STYLE),
            ]))
        body = "".join(rows)
    return _table(head, body)

# ─── UI Build ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(title="Budget Router Dashboard", theme=THEME, css=LIGHT_CSS) as demo:

        episode_state = gr.State(fresh_state())

        gr.Markdown(
            "# ⚙️ Budget Router — Live Episode Dashboard\n"
            "_3-provider LLM routing simulator · 20 steps · Budget & reliability constraints_"
        )

        with gr.Row():
            # ── LEFT: Controls ────────────────────────────────────────────────
            with gr.Column(scale=1, min_width=240):
                gr.Markdown("### Controls")
                scenario_sel = gr.Radio(SCENARIOS, value="easy", label="Scenario")
                gr.Markdown("*Select scenario above before running auto-play*")
                seed_inp     = gr.Number(value=42, label="Seed", precision=0)
                start_btn    = gr.Button("▶ Start Episode", variant="primary")

                gr.Markdown("### Action")
                action_sel = gr.Radio(ACTION_CHOICES, value="route_to_a", label="Select Action")
                step_btn   = gr.Button("→ Take Step", variant="secondary")
                auto_btn   = gr.Button("⚡ Run Heuristic Auto-Play")
                llm_btn    = gr.Button("🤖 Run LLM Auto-Play")

                status_box = gr.Textbox(label="Status", interactive=False, lines=2)

            # ── RIGHT: Live State ─────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Live State")
                provider_html = gr.HTML(_PROVIDER_EMPTY())
                budget_html   = gr.HTML()

                with gr.Row():
                    step_md    = gr.Markdown("**Step:** — / 20")
                    action_md  = gr.Markdown("**Last:** —")
                    reward_md  = gr.Markdown("**Cumulative:** 0.000")

        # ── BOTTOM: History ───────────────────────────────────────────────────
        gr.Markdown("### Episode History")
        history_tbl = gr.HTML(render_history_table([]))

        # ── GRADER: shown at episode end ──────────────────────────────────────
        gr.Markdown("### Episode Grade")
        grader_html = gr.HTML(_GRADER_PENDING())

        # ─── Output spec (shared by all 3 buttons) ────────────────────────────
        OUTPUTS = [
            episode_state,  # 0
            status_box,     # 1
            provider_html,  # 2
            budget_html,    # 3
            step_md,        # 4
            action_md,      # 5
            reward_md,      # 6
            history_tbl,    # 7
            grader_html,    # 8
        ]

        def _live_tuple(state: Dict, status: str,
                        obs: Optional[Dict] = None,
                        action: Optional[str] = None,
                        reward: Optional[float] = None) -> tuple:
            """Build the 9-element output tuple from current state."""
            obs = obs or state.get("obs", {})
            grade_html = _GRADER_PENDING()
            if state.get("done") and state.get("history"):
                grade_html = render_grader(compute_grade(state["history"]))

            sign = "+" if (reward or 0) >= 0 else ""
            last = f"**Last:** {action} → {sign}{reward:.3f}" if action else "**Last:** —"

            return (
                state,
                status,
                render_providers(obs) if obs else _PROVIDER_EMPTY(),
                render_budget(obs) if obs else "",
                f"**Step:** {state['step']} / {MAX_STEPS}",
                last,
                f"**Cumulative:** {state['cumulative_reward']:.3f}",
                render_history_table(state["history"]),
                grade_html,
            )

        # ─── Reset handler ────────────────────────────────────────────────────
        def do_reset(scenario: str, seed: float, _state: Dict) -> tuple:
            state, obs, err = reset_episode(scenario, seed)
            if err:
                return _live_tuple(state, f"❌ Reset failed: {err}")
            return _live_tuple(state, f"✅ Started · scenario={scenario} seed={int(seed)}", obs=obs)

        # ─── Manual step handler ──────────────────────────────────────────────
        def do_step(action: str, state: Dict) -> tuple:
            if state.get("done"):
                return _live_tuple(state, "⚠️ Episode done — start a new one.")
            if not state.get("obs"):
                return _live_tuple(state, "⚠️ No active episode — click Start first.")

            step_result, err = step_episode(action, state)
            if err:
                return _live_tuple(state, f"❌ Step failed: {err}")

            state, obs, reward, done = step_result

            status = format_step_status("✅", state["step"], action, reward, done)
            return _live_tuple(state, status, obs=obs, action=action, reward=reward)

        def run_policy_episode(scenario: str, seed: float, policy_name: str):
            policy_runner, policy_err = get_policy_runner(policy_name)
            if policy_err:
                state = fresh_state()
                yield _live_tuple(state, f"❌ {policy_err}")
                return

            state, obs, err = reset_episode(scenario, seed)
            if err:
                yield _live_tuple(state, f"❌ Reset failed: {err}")
                return
            meta = POLICY_META.get(policy_name, {"label": policy_name, "icon": "▶"})
            label = meta["label"]
            icon = meta["icon"]
            yield _live_tuple(state, f"▶ {label} simulation · scenario={scenario} seed={int(seed)}", obs=obs)

            while not state["done"] and state["step"] < MAX_STEPS:
                time.sleep(AUTO_PLAY_DELAY)
                try:
                    action = policy_runner.choose_action(obs) if policy_runner else "shed_load"
                except Exception as exc:
                    yield _live_tuple(state, f"❌ {label} policy error: {exc}")
                    return
                step_result, err = step_episode(action, state)
                if err:
                    yield _live_tuple(state, f"❌ Step error: {err}")
                    return

                state, obs, reward, done = step_result

                status = format_step_status(icon, state["step"], action, reward, done)
                yield _live_tuple(state, status, obs=obs, action=action, reward=reward)

        # ─── Auto-play handlers (streaming generators) ───────────────────────
        def do_auto_play(scenario: str, seed: float, _state: Dict):
            yield from run_policy_episode(scenario, seed, "heuristic")

        def do_llm_play(scenario: str, seed: float, _state: Dict):
            yield from run_policy_episode(scenario, seed, "llm")

        # ─── Wire buttons ─────────────────────────────────────────────────────
        start_btn.click(do_reset,    inputs=[scenario_sel, seed_inp, episode_state], outputs=OUTPUTS)
        step_btn.click( do_step,     inputs=[action_sel,   episode_state],           outputs=OUTPUTS)
        auto_btn.click( do_auto_play,inputs=[scenario_sel, seed_inp, episode_state], outputs=OUTPUTS)
        llm_btn.click(  do_llm_play, inputs=[scenario_sel, seed_inp, episode_state], outputs=OUTPUTS)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_port=7860)
