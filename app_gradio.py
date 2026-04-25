"""
Budget Router — Gradio Visualization Dashboard
Run: python app_gradio.py  (launches on http://localhost:7860)
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import gradio as gr
from budget_router.environment import BudgetRouterEnv
from budget_router.models import Action, ActionType
from budget_router.tasks import TASK_PRESETS

from gradio_ui.config import MAX_STEPS as _MAX_STEPS, POLICY_CHOICES, SCENARIOS
from gradio_ui.policies import get_policy_runner
from gradio_ui.renderers import (
    _kpi_grid,
    render_incident_timeline,
    render_side_panel,
    render_grader_plot,
    _GRADER_PENDING,
    _PROVIDER_EMPTY,
    render_history_table_compare,
)
from gradio_ui.state import fresh_side_state, _observation_to_dict, record_step
from gradio_ui.theme import LIGHT_CSS, THEME

MAX_STEPS = _MAX_STEPS


# Compatibility: preserve module-level MAX_STEPS for callers.

# ─── UI Build ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    def _normalize_seed(seed: object, default: int = 42) -> int:
        if seed is None:
            return default
        try:
            val = float(seed)  # type: ignore[arg-type]
        except Exception:
            return default
        if math.isnan(val) or math.isinf(val):
            return default
        try:
            return int(val)
        except Exception:
            return default

    with gr.Blocks(title="Budget Router — Policy Comparison", theme=THEME, css=LIGHT_CSS) as demo:

        left_state = gr.State(fresh_side_state())
        right_state = gr.State(fresh_side_state())
        run_state = gr.State({"running": False, "scenario": "easy", "seed": 42, "step": 0})

        gr.Markdown(
            "# Budget Router — Policy Comparison\n"
            "_Select 2 policies · start episode · step or fast-forward · compare outcomes_"
        )

        with gr.Row():
            with gr.Column(scale=1):
                left_title = gr.Markdown("## Policy A")
                left_policy = gr.Dropdown(choices=POLICY_CHOICES, value=None, label="Select policy")
                left_status = gr.Textbox(label="Status", interactive=False, lines=2)
                left_providers = gr.HTML(_PROVIDER_EMPTY())
                left_budget = gr.HTML("")
                left_kpis = gr.HTML(
                    _kpi_grid(
                        [
                            ("Step", "—"),
                            ("Last action", "—"),
                            ("Latency (ms)", "—"),
                            ("Budget remaining", "—"),
                            ("Reward", "—"),
                            ("Adaptation", "—"),
                        ]
                    )
                )
                left_badges = gr.HTML("")
                left_summary = gr.HTML(
                    _kpi_grid(
                        [
                            ("Failed %", "—"),
                            ("SLA breach %", "—"),
                            ("Avg latency (ms)", "—"),
                        ]
                    )
                )

            with gr.Column(scale=1):
                right_title = gr.Markdown("## Policy B")
                right_policy = gr.Dropdown(choices=POLICY_CHOICES, value=None, label="Select policy")
                right_status = gr.Textbox(label="Status", interactive=False, lines=2)
                right_providers = gr.HTML(_PROVIDER_EMPTY())
                right_budget = gr.HTML("")
                right_kpis = gr.HTML(
                    _kpi_grid(
                        [
                            ("Step", "—"),
                            ("Last action", "—"),
                            ("Latency (ms)", "—"),
                            ("Budget remaining", "—"),
                            ("Reward", "—"),
                            ("Adaptation", "—"),
                        ]
                    )
                )
                right_badges = gr.HTML("")
                right_summary = gr.HTML(
                    _kpi_grid(
                        [
                            ("Failed %", "—"),
                            ("SLA breach %", "—"),
                            ("Avg latency (ms)", "—"),
                        ]
                    )
                )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Episode Controls")
                scenario_sel = gr.Radio(SCENARIOS, value="easy", label="Scenario")
                seed_inp = gr.Number(value=42, label="Seed", precision=0)
                start_btn = gr.Button("▶ Start Episode", variant="primary", interactive=False)
                with gr.Row():
                    step_btn = gr.Button("→ Step", variant="secondary", interactive=False)
                    fast_btn = gr.Button("⚡ Fast-forward", interactive=False)
                    finish_btn = gr.Button("⏩ Finish Episode", interactive=False)

        gr.Markdown("### Grader score (comparison)")
        grader_plot = gr.Plot()

        with gr.Row(elem_classes=["episode-history-row"]):
            with gr.Column(scale=1):
                left_history_title = gr.Markdown("### Step History — Policy A")
                left_history_tbl = gr.HTML(render_history_table_compare([]), elem_classes=["episode-history-table"])
            with gr.Column(scale=1):
                right_history_title = gr.Markdown("### Step History — Policy B")
                right_history_tbl = gr.HTML(render_history_table_compare([]), elem_classes=["episode-history-table"])

        with gr.Row():
            with gr.Column(scale=1):
                left_grade_title = gr.Markdown("### Episode Grade — Policy A")
                left_grade = gr.HTML(_GRADER_PENDING())
            with gr.Column(scale=1):
                right_grade_title = gr.Markdown("### Episode Grade — Policy B")
                right_grade = gr.HTML(_GRADER_PENDING())

        gr.Markdown("### Incident Timeline")
        incidents_html = gr.HTML(render_incident_timeline("easy"))

        def _render_side(side: Dict, run: Dict, scenario_name: str) -> Tuple[str, str, str, str, str, str, str, str]:
            return render_side_panel(side, run, scenario_name)

        def _render_all(ls: Dict, rs: Dict, run: Dict) -> tuple:
            scenario_name = str(run.get("scenario", "easy") or "easy")
            l_out = _render_side(ls, run, scenario_name)
            r_out = _render_side(rs, run, scenario_name)
            plot = render_grader_plot(ls.get("history", []) or [], rs.get("history", []) or [])
            incidents = render_incident_timeline(scenario_name)

            running = bool(run.get("running", False))
            btn_update = gr.update(interactive=running)
            config_update = gr.update(interactive=(not running))
            return (
                ls,
                rs,
                run,
                l_out[0],
                l_out[1],
                l_out[2],
                l_out[3],
                l_out[4],
                l_out[5],
                r_out[0],
                r_out[1],
                r_out[2],
                r_out[3],
                r_out[4],
                r_out[5],
                l_out[6],
                r_out[6],
                l_out[7],
                r_out[7],
                plot,
                incidents,
                config_update,
                config_update,
                config_update,
                config_update,
                config_update,
                btn_update,
                btn_update,
                btn_update,
            )

        OUTPUTS = [
            left_state,
            right_state,
            run_state,
            left_status,
            left_providers,
            left_budget,
            left_kpis,
            left_badges,
            left_summary,
            right_status,
            right_providers,
            right_budget,
            right_kpis,
            right_badges,
            right_summary,
            left_history_tbl,
            right_history_tbl,
            left_grade,
            right_grade,
            grader_plot,
            incidents_html,
            left_policy,
            right_policy,
            scenario_sel,
            seed_inp,
            start_btn,
            step_btn,
            fast_btn,
            finish_btn,
        ]

        GRADER_PLOT_IDX = OUTPUTS.index(grader_plot)

        def _update_start_enabled(p1: Optional[str], p2: Optional[str], run: Dict):
            left_name = str(p1 or "Policy A")
            right_name = str(p2 or "Policy B")
            running = bool((run or {}).get("running", False))
            ok = (bool(p1) and bool(p2)) and (not running)
            return (
                gr.update(interactive=ok),
                f"## {left_name}",
                f"## {right_name}",
                f"### Step History — {left_name}",
                f"### Step History — {right_name}",
                f"### Episode Grade — {left_name}",
                f"### Episode Grade — {right_name}",
            )

        left_policy.change(
            _update_start_enabled,
            inputs=[left_policy, right_policy, run_state],
            outputs=[start_btn, left_title, right_title, left_history_title, right_history_title, left_grade_title, right_grade_title],
        )
        right_policy.change(
            _update_start_enabled,
            inputs=[left_policy, right_policy, run_state],
            outputs=[start_btn, left_title, right_title, left_history_title, right_history_title, left_grade_title, right_grade_title],
        )

        scenario_sel.change(lambda s: render_incident_timeline(s), inputs=[scenario_sel], outputs=[incidents_html])

        def do_start(p1: str, p2: str, scenario: str, seed: Optional[float], _ls: Dict, _rs: Dict, _run: Dict):
            ls = fresh_side_state()
            rs = fresh_side_state()

            seed_int = _normalize_seed(seed, default=42)

            if not p1 or not p2:
                run = {"running": False, "scenario": scenario, "seed": seed_int, "step": 0}
                ls["status"] = "Select both policies to start."
                rs["status"] = "Select both policies to start."
                return _render_all(ls, rs, run)

            runner_l, err_l = get_policy_runner(p1)
            runner_r, err_r = get_policy_runner(p2)
            if err_l or err_r or runner_l is None or runner_r is None:
                ls["status"] = f"❌ {err_l}" if err_l else ""
                rs["status"] = f"❌ {err_r}" if err_r else ""
                run = {"running": False, "scenario": scenario, "seed": seed_int, "step": 0}
                return _render_all(ls, rs, run)

            env_l = BudgetRouterEnv()
            env_r = BudgetRouterEnv()
            obs_l = env_l.reset(seed=seed_int, scenario=scenario)
            obs_r = env_r.reset(seed=seed_int, scenario=scenario)
            try:
                runner_l.reset(scenario)
            except Exception:
                pass
            try:
                runner_r.reset(scenario)
            except Exception:
                pass

            ls.update(
                {
                    "env": env_l,
                    "policy_name": p1,
                    "policy_runner": runner_l,
                    "obs": _observation_to_dict(obs_l),
                    "status": f"✅ Running · {p1}",
                }
            )
            rs.update(
                {
                    "env": env_r,
                    "policy_name": p2,
                    "policy_runner": runner_r,
                    "obs": _observation_to_dict(obs_r),
                    "status": f"✅ Running · {p2}",
                }
            )
            run = {"running": True, "scenario": scenario, "seed": seed_int, "step": 0}
            return _render_all(ls, rs, run)

        def _apply_local_step(side: Dict, scenario_name: str, global_step: int) -> Dict:
            if side.get("done"):
                return side
            env = side.get("env")
            runner = side.get("policy_runner")
            if env is None or runner is None:
                side["done"] = True
                side["status"] = "❌ Not initialized"
                return side
            try:
                action_str = runner.choose_action(side.get("obs", {}) or {})
            except Exception as exc:
                side["done"] = True
                side["status"] = f"❌ Policy error: {exc}"
                return side

            pre_obs = dict(side.get("obs", {}) or {})
            obs_obj = env.step(Action(action_type=ActionType(action_str)))
            obs = _observation_to_dict(obs_obj)
            reward = float(obs.get("reward", 0.0) or 0.0)
            meta = dict(obs.get("metadata", {}) or {})
            done = bool(obs.get("done", False))
            side["history"].append(record_step(global_step, action_str, obs, reward, meta, health_obs=pre_obs))
            side["obs"] = obs
            side["cumulative_reward"] = float(side.get("cumulative_reward", 0.0) or 0.0) + reward
            side["done"] = done
            side["status"] = "✅ Done" if done else str(side.get("status", ""))
            return side

        def do_step(ls: Dict, rs: Dict, run: Dict):
            if not bool(run.get("running", False)):
                return _render_all(ls, rs, run)
            if int(run.get("step", 0) or 0) >= MAX_STEPS:
                run["running"] = False
                return _render_all(ls, rs, run)

            next_step = int(run.get("step", 0) or 0) + 1
            scenario = str(run.get("scenario", "easy") or "easy")

            ls = _apply_local_step(ls, scenario, next_step)
            rs = _apply_local_step(rs, scenario, next_step)
            run["step"] = next_step

            if next_step >= MAX_STEPS or (ls.get("done") and rs.get("done")):
                run["running"] = False
            return _render_all(ls, rs, run)

        def _stream_to_end(ls: Dict, rs: Dict, run: Dict):
            if not bool(run.get("running", False)):
                yield _render_all(ls, rs, run)
                return

            frozen = _render_all(ls, rs, run)
            frozen_grader_plot = frozen[GRADER_PLOT_IDX]

            while bool(run.get("running", False)) and int(run.get("step", 0) or 0) < MAX_STEPS:
                out = do_step(ls, rs, run)
                ls, rs, run = out[0], out[1], out[2]
                out_list = list(out)
                out_list[GRADER_PLOT_IDX] = frozen_grader_plot
                yield tuple(out_list)
                time.sleep(0.12)
                if not bool(run.get("running", False)):
                    break

            yield _render_all(ls, rs, run)

        def do_fast_forward(ls: Dict, rs: Dict, run: Dict):
            yield from _stream_to_end(ls, rs, run)

        def do_finish(ls: Dict, rs: Dict, run: Dict):
            yield from _stream_to_end(ls, rs, run)

        start_btn.click(do_start, inputs=[left_policy, right_policy, scenario_sel, seed_inp, left_state, right_state, run_state], outputs=OUTPUTS)
        step_btn.click(do_step, inputs=[left_state, right_state, run_state], outputs=OUTPUTS)
        fast_btn.click(do_fast_forward, inputs=[left_state, right_state, run_state], outputs=OUTPUTS)
        finish_btn.click(do_finish, inputs=[left_state, right_state, run_state], outputs=OUTPUTS)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(server_port=7860)
