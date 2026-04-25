from __future__ import annotations

import html
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from budget_router.reward import grade_episode
from budget_router.tasks import TASK_PRESETS

from .config import MAX_STEPS

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
    if remaining > 0.5:
        return "#27ae60"
    if remaining > 0.2:
        return "#f39c12"
    return "#e74c3c"


def render_providers(obs: Dict) -> str:
    return _join(
        [
            _bar(obs.get("provider_a_status", 0), "Provider A", "#27ae60"),
            _bar(obs.get("provider_b_status", 0), "Provider B", "#e67e22"),
            _bar(obs.get("provider_c_status", 0), "Provider C", "#e74c3c"),
        ]
    )


def render_budget(obs: Dict) -> str:
    b = obs.get("budget_remaining", 1.0)
    return _bar(b, "Budget Remaining", _budget_color(b))


def render_grader(grade: Dict) -> str:
    o = grade["overall_score"]
    color = "#27ae60" if o > 0.7 else "#f39c12" if o > 0.4 else "#e74c3c"
    return (
        f'<div style="text-align:center;font-size:28px;font-weight:bold;'
        f'color:{color};margin-top:12px;padding:8px;border-radius:8px;'
        f'background:rgba(0,0,0,0.04)">Overall Score: {o:.1%}</div>'
    )


def _GRADER_PENDING() -> str:
    return "<div style='color:#aaa;font-style:italic'>Shown when episode completes.</div>"


def _REWARD_PENDING() -> str:
    return "<div style='color:#aaa;font-style:italic'>Shown when episode completes.</div>"


def _PROVIDER_EMPTY() -> str:
    return "<div style='color:#aaa;font-style:italic'>Start an episode to see provider health.</div>"


def render_episode_total_reward(history: List[Dict]) -> str:
    total = 0.0
    for h in history or []:
        total += float(h.get("reward", 0.0) or 0.0)
    color = "#27ae60" if total >= 0 else "#e74c3c"
    return (
        f'<div style="text-align:center;font-size:28px;font-weight:bold;'
        f'color:{color};margin-top:12px;padding:8px;border-radius:8px;'
        f'background:rgba(0,0,0,0.04)">Total Reward: {total:+.2f}</div>'
    )


def _task_config(scenario_name: str):
    return TASK_PRESETS.get(scenario_name, TASK_PRESETS["easy"])


def _common_provider_health(config: Any, step: int) -> Dict[str, float]:
    health = {
        "A": float(getattr(config, "reliability_a", 0.0) or 0.0),
        "B": float(getattr(config, "reliability_b", 0.0) or 0.0),
        "C": float(getattr(config, "reliability_c", 0.0) or 0.0),
    }

    start_raw = getattr(config, "degradation_start_step", 999)
    start = 999 if start_raw is None else int(start_raw)
    target = str(getattr(config, "degradation_target", "A") or "A")
    rate = float(getattr(config, "degradation_rate", 0.0) or 0.0)
    if start < 999 and step >= start:
        count = step if start == 0 else max(0, step - start + 1)
        health[target] = max(0.05, float(health.get(target, 1.0)) - rate * count)

    secondary_start = getattr(config, "secondary_degradation_start_step", 999)
    secondary_target = str(getattr(config, "secondary_degradation_target", "") or "")
    secondary_rate = float(getattr(config, "secondary_degradation_rate", 0.0) or 0.0)
    if (
        secondary_target
        and secondary_start is not None
        and int(secondary_start) < 999
        and step >= int(secondary_start)
    ):
        count2 = max(0, step - int(secondary_start) + 1)
        health[secondary_target] = max(
            0.05, float(health.get(secondary_target, 1.0)) - secondary_rate * count2
        )

    return health


def render_common_providers(health: Dict[str, float]) -> str:
    return _join(
        [
            _bar(float(health.get("A", 0.0)), "Provider A", "#27ae60"),
            _bar(float(health.get("B", 0.0)), "Provider B", "#e67e22"),
            _bar(float(health.get("C", 0.0)), "Provider C", "#e74c3c"),
        ]
    )


def _pct(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return 100.0 * (n / d)


def _summary_metrics(history: List[Dict]) -> Dict[str, float]:
    routing = [h for h in history if h.get("action") != "shed_load"]
    failures = sum(1 for h in routing if not bool(h.get("succeeded", False)))
    breaches = sum(
        1
        for h in routing
        if float(h.get("latency_ms", 0.0) or 0.0)
        > float(h.get("sla_ceiling_ms", 500.0) or 500.0)
    )
    latencies = [float(h.get("latency_ms", 0.0) or 0.0) for h in routing]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    return {
        "failed_pct": _pct(failures, len(routing)),
        "sla_breach_pct": _pct(breaches, len(routing)),
        "avg_latency_ms": avg_latency,
    }


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _fmt_pct(ok: int, total: int) -> str:
    if total <= 0:
        return "—"
    return f"{(100.0 * ok / total):.1f}% ({ok}/{total})"


def render_data_quality_panel(history: List[Dict]) -> str:
    try:
        if not history:
            return (
                "<details style='margin-top:10px'>"
                "<summary style='cursor:pointer;font-weight:600;color:#111827'>Data quality (optional)</summary>"
                "<div style='color:#6b7280;margin-top:6px;font-style:italic'>No steps yet.</div>"
                "</details>"
            )

        schema_failed: List[Tuple[int, str]] = []
        consistency_failed: List[Tuple[int, str]] = []

        prev_budget: Optional[float] = None

        for idx, h in enumerate(history):
            step_raw = h.get("step", idx + 1)
            try:
                step = int(step_raw)
            except Exception:
                step = idx + 1

            action = h.get("action")
            budget = h.get("budget")
            meta_raw = h.get("meta_raw")

            schema_errs: List[str] = []
            if action not in {"route_to_a", "route_to_b", "route_to_c", "shed_load"}:
                schema_errs.append(f"invalid action={action!r}")

            if not isinstance(meta_raw, dict):
                schema_errs.append("meta_raw missing/invalid")
                meta_raw = {}

            required_meta_keys = {
                "step",
                "action_type",
                "request_succeeded",
                "cost",
                "latency_ms",
                "sla_ceiling_ms",
                "initial_budget",
                "degradation_start_step",
            }
            missing_meta = [k for k in sorted(required_meta_keys) if k not in meta_raw]
            if missing_meta:
                schema_errs.append("missing meta: " + ", ".join(missing_meta))

            meta_action = meta_raw.get("action_type")
            if meta_action is not None and meta_action != action:
                schema_errs.append(f"meta action_type={meta_action!r} != action={action!r}")

            meta_step = meta_raw.get("step")
            try:
                meta_step_i = int(meta_step) if meta_step is not None else None
            except Exception:
                meta_step_i = None
                schema_errs.append("meta step not int")

            if meta_step_i is not None and meta_step_i != step:
                schema_errs.append(f"meta step={meta_step_i} != history step={step}")

            succeeded_raw = meta_raw.get("request_succeeded")
            cost_raw = meta_raw.get("cost")
            latency_raw = meta_raw.get("latency_ms")
            sla_raw = meta_raw.get("sla_ceiling_ms")
            init_b_raw = meta_raw.get("initial_budget")

            if not isinstance(succeeded_raw, bool):
                schema_errs.append("meta request_succeeded not bool")

            if not _is_finite_number(cost_raw) or float(cost_raw) < 0:
                schema_errs.append("meta cost not finite >= 0")

            if not _is_finite_number(latency_raw) or float(latency_raw) < 0:
                schema_errs.append("meta latency_ms not finite >= 0")

            if not _is_finite_number(sla_raw) or float(sla_raw) <= 0:
                schema_errs.append("meta sla_ceiling_ms not finite > 0")

            if not _is_finite_number(init_b_raw) or float(init_b_raw) <= 0:
                schema_errs.append("meta initial_budget not finite > 0")

            if not _is_finite_number(budget) or float(budget) < 0:
                schema_errs.append("budget not finite >= 0")

            if schema_errs:
                schema_failed.append((step, "; ".join(schema_errs)))
                continue

            b = float(budget)
            c = float(cost_raw)
            ib = float(init_b_raw)

            consistency_errs: List[str] = []
            if step != (idx + 1):
                consistency_errs.append(f"step mismatch: got {step}, expected {idx + 1}")

            if b > 1.0 + 1e-6:
                consistency_errs.append("budget > 1.0")

            if action == "shed_load":
                if succeeded_raw is not False:
                    consistency_errs.append("shed_load succeeded must be False")
                if abs(c - 0.0) > 1e-6:
                    consistency_errs.append("shed_load cost must be 0")
                if abs(float(latency_raw) - 0.0) > 1e-6:
                    consistency_errs.append("shed_load latency_ms must be 0")

            if prev_budget is not None:
                if action == "shed_load":
                    expected = prev_budget
                else:
                    burn = (c / ib) if ib > 0 else 0.0
                    expected = max(0.0, prev_budget - burn)
                if abs(b - expected) > 1e-4:
                    consistency_errs.append(
                        f"budget mismatch: got {b:.4f}, expected {expected:.4f}"
                    )
            prev_budget = b

            if consistency_errs:
                consistency_failed.append((step, "; ".join(consistency_errs)))

        total_steps = len(history)
        schema_ok = total_steps - len(schema_failed)
        consistency_ok = total_steps - len(consistency_failed)

        summary = _kpi_grid(
            [
                ("Schema valid", _fmt_pct(schema_ok, total_steps)),
                ("Consistency OK", _fmt_pct(consistency_ok, total_steps)),
                ("Violations", str(len(schema_failed) + len(consistency_failed))),
            ]
        )

        failures: List[Tuple[int, str, str]] = [
            (s, "schema", msg) for s, msg in schema_failed
        ] + [(s, "consistency", msg) for s, msg in consistency_failed]
        failures.sort(key=lambda x: (x[0], x[1]))

        if not failures:
            fail_html = "<div style='color:#16a34a;margin-top:6px'>No violations detected.</div>"
        else:
            max_rows = 12
            rows = []
            head = _tr([
                _th("Step"),
                _th("Type"),
                _th("Reason"),
            ], style=_HEADER_ROW_STYLE)
            for s, kind, msg in failures[:max_rows]:
                rows.append(
                    _tr(
                        [
                            _td(str(s), _CELL_STYLE),
                            _td(kind, _CELL_STYLE),
                            _td(html.escape(msg), f"{_CELL_STYLE};text-align:left"),
                        ]
                    )
                )
            body = "".join(rows)
            extra = ""
            if len(failures) > max_rows:
                extra = (
                    f"<div style='color:#6b7280;margin-top:6px'>"
                    f"Showing {max_rows} of {len(failures)} violations."
                    f"</div>"
                )
            fail_html = _table(head, body) + extra

        return (
            "<details style='margin-top:10px'>"
            "<summary style='cursor:pointer;font-weight:600;color:#111827'>Data quality (optional)</summary>"
            f"<div style='margin-top:8px'>{summary}</div>"
            f"<div style='margin-top:8px'>{fail_html}</div>"
            "</details>"
        )
    except Exception as exc:
        return (
            "<details style='margin-top:10px'>"
            "<summary style='cursor:pointer;font-weight:600;color:#111827'>Data quality (optional)</summary>"
            f"<div style='color:#dc2626;margin-top:6px'>Error computing data quality: {html.escape(str(exc))}</div>"
            "</details>"
        )


def _step_badges(last: Optional[Dict]) -> str:
    if not last:
        return "<div style='color:#aaa;font-style:italic'>No steps yet.</div>"
    action = str(last.get("action") or "")
    if action == "shed_load":
        return (
            "<div style='display:flex;gap:8px;align-items:center;margin-top:6px'>"
            "<span style='padding:2px 8px;border-radius:999px;background:#6b7280;color:white;font-size:12px;font-weight:600'>SHED LOAD</span>"
            "<span style='padding:2px 8px;border-radius:999px;background:#6b7280;color:white;font-size:12px;font-weight:600'>SLA N/A</span>"
            "</div>"
        )
    succeeded = bool(last.get("succeeded", False))
    latency_ms = float(last.get("latency_ms", 0.0) or 0.0)
    sla_ms = float(last.get("sla_ceiling_ms", 500.0) or 500.0)
    breach = latency_ms > sla_ms
    s_color = "#16a34a" if succeeded else "#dc2626"
    s_text = "SUCCESS" if succeeded else "FAILED"
    b_color = "#dc2626" if breach else "#16a34a"
    b_text = "SLA BREACH" if breach else "SLA OK"
    return (
        "<div style='display:flex;gap:8px;align-items:center;margin-top:6px'>"
        f"<span style='padding:2px 8px;border-radius:999px;background:{s_color};color:white;font-size:12px;font-weight:600'>{s_text}</span>"
        f"<span style='padding:2px 8px;border-radius:999px;background:{b_color};color:white;font-size:12px;font-weight:600'>{b_text}</span>"
        "</div>"
    )


_ACTION_COLORS = {
    "route_to_a": "#d5f5e3",
    "route_to_b": "#fef9e7",
    "route_to_c": "#fadbd8",
    "shed_load": "#f2f3f4",
}


def _normalize_action_label(action_value: Any) -> str:
    raw = str(action_value or "shed_load").strip().lower()
    if not raw:
        return "shed_load"
    matches = re.findall(r"route_to_[abc]|shed_load", raw)
    if matches:
        return matches[0]
    return raw.split()[0]


def render_history_table_compare(history: List[Dict]) -> str:
    headers = [
        "Step",
        "Action",
        "Provider A<br>Health",
        "Provider B<br>Health",
        "Provider C<br>Health",
        "OK",
        "SLA",
        "Latency<br>(ms)",
        "Budget",
        "Reward",
    ]
    head = _tr([_th(h) for h in headers], style=_HEADER_ROW_STYLE)
    table_style = (
        "width:100%;border-collapse:collapse;table-layout:fixed;"
        "font-size:12px;background:#ffffff;color:#111827"
    )
    colgroup = (
        "<colgroup>"
        "<col style='width:9%'>"
        "<col style='width:16%'>"
        "<col style='width:12%'>"
        "<col style='width:12%'>"
        "<col style='width:12%'>"
        "<col style='width:6%'>"
        "<col style='width:6%'>"
        "<col style='width:10%'>"
        "<col style='width:8%'>"
        "<col style='width:9%'>"
        "</colgroup>"
    )
    cell_style = _CELL_STYLE + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px"
    step_style = _CELL_STYLE + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px"
    action_style = _ACTION_CELL_STYLE + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px"
    health_style = _CELL_STYLE + ";white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px"
    if not history:
        body = _tr(
            [_td("No steps yet.", "padding:8px;color:#888;text-align:center", colspan=len(headers))]
        )
    else:
        rows = []
        for h in history:
            action = _normalize_action_label(h.get("action"))
            action_color = _ACTION_COLORS.get(action, "#f2f3f4")
            succeeded = bool(h.get("succeeded", False))
            latency_ms = float(h.get("latency_ms", 0.0) or 0.0)
            sla_ms = float(h.get("sla_ceiling_ms", 500.0) or 500.0)
            breach = latency_ms > sla_ms
            health_a = float(h.get("health_a", 0.0) or 0.0)
            health_b = float(h.get("health_b", 0.0) or 0.0)
            health_c = float(h.get("health_c", 0.0) or 0.0)
            if action == "shed_load":
                ok_cell = "—"
                sla_cell = "—"
            else:
                ok_cell = "✅" if succeeded else "❌"
                sla_cell = "❌" if breach else "✅"
            rows.append(
                _tr(
                    [
                        _td(str(h.get("step", "—")), step_style),
                        _td(action, f"{action_style};background:{action_color}"),
                        _td(f"{health_a:.2f}", health_style),
                        _td(f"{health_b:.2f}", health_style),
                        _td(f"{health_c:.2f}", health_style),
                        _td(ok_cell, cell_style),
                        _td(sla_cell, cell_style),
                        _td(f"{latency_ms:.0f}", cell_style),
                        _td(f"{float(h.get('budget', 0.0) or 0.0):.2f}", cell_style),
                        _td(f"{float(h.get('reward', 0.0) or 0.0):+.3f}", cell_style),
                    ]
                )
            )
        body = "".join(rows)
    return f"<table style='{table_style}'>{colgroup}<thead>{head}</thead><tbody>{body}</tbody></table>"


def _kpi_grid(items: List[Tuple[str, str]]) -> str:
    cards = []
    for label, value in items:
        cards.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{html.escape(str(label))}</div>"
            f"<div class='kpi-value'>{html.escape(str(value))}</div>"
            "</div>"
        )
    return "<div class='kpi-grid'>" + "".join(cards) + "</div>"


def render_incident_timeline(scenario_name: str) -> str:
    config = _task_config(scenario_name)
    start_raw = getattr(config, "degradation_start_step", 999)
    start = 999 if start_raw is None else int(start_raw)
    target = str(getattr(config, "degradation_target", "A") or "A")
    secondary_start = getattr(config, "secondary_degradation_start_step", 999)
    secondary_target = str(getattr(config, "secondary_degradation_target", "") or "")

    items: List[str] = []
    if start < 999:
        items.append(f"<div><b>Step {start}</b>: degradation starts for Provider {target}</div>")
    if secondary_target and secondary_start is not None and int(secondary_start) < 999:
        items.append(
            f"<div><b>Step {int(secondary_start)}</b>: degradation starts for Provider {secondary_target}</div>"
        )
    if not items:
        return "<div style='color:#aaa;font-style:italic'>No configured incidents for this scenario.</div>"
    return "<div style='display:flex;flex-direction:column;gap:6px'>" + "".join(items) + "</div>"


def render_grader_plot(left_hist: List[Dict], right_hist: List[Dict]):
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None

    if go is not None:
        def series(hist: List[Dict]) -> List[float]:
            out: List[float] = []
            for i in range(1, len(hist) + 1):
                out.append(float(compute_grade(hist[:i]).get("overall_score", 0.0)))
            return out

        y1 = series(left_hist)
        y2 = series(right_hist)
        x1 = list(range(1, len(y1) + 1))
        x2 = list(range(1, len(y2) + 1))

        color_a = "#f39c12"
        color_b = "#3498db"

        fig = go.Figure()
        if y1:
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=y1,
                    mode="lines",
                    name="Policy A",
                    line=dict(color=color_a, width=3),
                    hovertemplate="Step %{x}<br>Score %{y:.0%}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x1[-1],
                y=y1[-1],
                text=f"{y1[-1]:.0%}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=color_a, size=12),
            )
        if y2:
            fig.add_trace(
                go.Scatter(
                    x=x2,
                    y=y2,
                    mode="lines",
                    name="Policy B",
                    line=dict(color=color_b, width=3),
                    hovertemplate="Step %{x}<br>Score %{y:.0%}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x2[-1],
                y=y2[-1],
                text=f"{y2[-1]:.0%}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=color_b, size=12),
            )

        fig.update_layout(
            template="plotly_white",
            title=dict(text="Overall score over steps", x=0.0, xanchor="left"),
            margin=dict(l=50, r=20, t=45, b=40),
            height=320,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12, color="#111827"),
        )
        fig.update_xaxes(title_text="Step", showgrid=False, zeroline=False)
        fig.update_yaxes(
            title_text="Overall score",
            tickformat=",.0%",
            range=[0, 1],
            showgrid=True,
            gridcolor="rgba(17,24,39,0.08)",
            zeroline=False,
        )
        return fig

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except Exception:
        return None

    def series(hist: List[Dict]) -> List[float]:
        out: List[float] = []
        for i in range(1, len(hist) + 1):
            out.append(float(compute_grade(hist[:i]).get("overall_score", 0.0)))
        return out

    y1 = series(left_hist)
    y2 = series(right_hist)
    x1 = list(range(1, len(y1) + 1))
    x2 = list(range(1, len(y2) + 1))

    fig = plt.figure(figsize=(8, 3.2), dpi=120)
    ax = fig.add_subplot(111)

    color_a = "#f39c12"
    color_b = "#3498db"
    if y1:
        ax.plot(x1, y1, label="Policy A", linewidth=2.2, color=color_a)
        ax.annotate(
            f"{y1[-1]:.0%}",
            xy=(x1[-1], y1[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color=color_a,
        )
    if y2:
        ax.plot(x2, y2, label="Policy B", linewidth=2.2, color=color_b)
        ax.annotate(
            f"{y2[-1]:.0%}",
            xy=(x2[-1], y2[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color=color_b,
        )

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Step")
    ax.set_ylabel("Overall score")
    ax.set_title("Overall score over steps", loc="left", fontsize=11, fontweight="bold", color="#111827")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e5e7eb")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def render_reward_plot(left_hist: List[Dict], right_hist: List[Dict]):
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None

    def series(hist: List[Dict]) -> Tuple[List[int], List[float]]:
        xs: List[int] = []
        ys: List[float] = []
        total = 0.0
        for i, h in enumerate(hist, start=1):
            total += float(h.get("reward", 0.0) or 0.0)
            xs.append(i)
            ys.append(total)
        return xs, ys

    x1, y1 = series(left_hist)
    x2, y2 = series(right_hist)
    if not y1 and not y2:
        return None

    if go is not None:
        color_a = "#f39c12"
        color_b = "#3498db"
        fig = go.Figure()

        if y1:
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=y1,
                    mode="lines",
                    name="Policy A",
                    line=dict(color=color_a, width=3),
                    hovertemplate="Step %{x}<br>Cumulative %{y:+.2f}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x1[-1],
                y=y1[-1],
                text=f"{y1[-1]:+.2f}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=color_a, size=12),
            )

        if y2:
            fig.add_trace(
                go.Scatter(
                    x=x2,
                    y=y2,
                    mode="lines",
                    name="Policy B",
                    line=dict(color=color_b, width=3),
                    hovertemplate="Step %{x}<br>Cumulative %{y:+.2f}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x2[-1],
                y=y2[-1],
                text=f"{y2[-1]:+.2f}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color=color_b, size=12),
            )

        fig.update_layout(
            template="plotly_white",
            title=dict(text="Cumulative reward over steps", x=0.0, xanchor="left"),
            margin=dict(l=60, r=20, t=45, b=40),
            height=280,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12, color="#111827"),
        )
        fig.update_xaxes(title_text="Step", showgrid=False, zeroline=False)
        fig.update_yaxes(
            title_text="Cumulative reward",
            showgrid=True,
            gridcolor="rgba(17,24,39,0.08)",
            zeroline=False,
        )
        return fig

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure(figsize=(8, 2.8), dpi=120)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e5e7eb")
    ax.tick_params(colors="#374151", labelsize=9)

    if y1:
        ax.plot(x1, y1, label="Policy A", linewidth=2.2, color="#f39c12")
        ax.annotate(
            f"{y1[-1]:+.2f}",
            xy=(x1[-1], y1[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color="#f39c12",
        )
    if y2:
        ax.plot(x2, y2, label="Policy B", linewidth=2.2, color="#3498db")
        ax.annotate(
            f"{y2[-1]:+.2f}",
            xy=(x2[-1], y2[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color="#3498db",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Cumulative reward over steps", loc="left", fontsize=11, fontweight="bold", color="#111827")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def render_budget_consumed_plot(left_hist: List[Dict], right_hist: List[Dict]):
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None

    if go is not None:
        def series(hist: List[Dict]) -> List[float]:
            if not hist:
                return []
            initial = float(hist[0].get("initial_budget", 0.0) or 0.0)
            consumed: List[float] = []
            total = 0.0
            for h in hist:
                total += float(h.get("cost", 0.0) or 0.0)
                if initial > 0:
                    consumed.append(min(initial, total))
                else:
                    consumed.append(total)
            return consumed

        y1 = series(left_hist)
        y2 = series(right_hist)
        if not y1 and not y2:
            return None

        x1 = list(range(1, len(y1) + 1))
        x2 = list(range(1, len(y2) + 1))

        fig = go.Figure()
        if y1:
            fig.add_trace(
                go.Scatter(
                    x=x1,
                    y=y1,
                    mode="lines",
                    name="Policy A",
                    line=dict(color="#f39c12", width=3),
                    hovertemplate="Step %{x}<br>Consumed $%{y:,.0f}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x1[-1],
                y=y1[-1],
                text=f"${y1[-1]:,.0f}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color="#f39c12", size=12),
            )
        if y2:
            fig.add_trace(
                go.Scatter(
                    x=x2,
                    y=y2,
                    mode="lines",
                    name="Policy B",
                    line=dict(color="#3498db", width=3),
                    hovertemplate="Step %{x}<br>Consumed $%{y:,.0f}<extra></extra>",
                )
            )
            fig.add_annotation(
                x=x2[-1],
                y=y2[-1],
                text=f"${y2[-1]:,.0f}",
                showarrow=False,
                xanchor="left",
                xshift=8,
                font=dict(color="#3498db", size=12),
            )

        fig.update_layout(
            template="plotly_white",
            title=dict(text="Budget consumed over steps", x=0.0, xanchor="left"),
            margin=dict(l=60, r=20, t=45, b=40),
            height=280,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial", size=12, color="#111827"),
        )
        fig.update_xaxes(title_text="Step", showgrid=False, zeroline=False)
        fig.update_yaxes(
            title_text="Budget consumed ($)",
            tickprefix="$",
            separatethousands=True,
            showgrid=True,
            gridcolor="rgba(17,24,39,0.08)",
            zeroline=False,
        )
        return fig

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import StrMethodFormatter
    except Exception:
        return None

    def series(hist: List[Dict]) -> List[float]:
        if not hist:
            return []
        initial = float(hist[0].get("initial_budget", 0.0) or 0.0)
        consumed: List[float] = []
        total = 0.0
        for h in hist:
            total += float(h.get("cost", 0.0) or 0.0)
            if initial > 0:
                consumed.append(min(initial, total))
            else:
                consumed.append(total)
        return consumed

    y1 = series(left_hist)
    y2 = series(right_hist)
    if not y1 and not y2:
        return None

    x1 = list(range(1, len(y1) + 1))
    x2 = list(range(1, len(y2) + 1))

    fig = plt.figure(figsize=(8, 2.8), dpi=120)
    ax = fig.add_subplot(111)

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if y1:
        ax.plot(x1, y1, label="Policy A", linewidth=2.2, color="#f39c12")
        ax.fill_between(x1, y1, alpha=0.10, color="#f39c12")
        ax.annotate(
            f"${y1[-1]:,.0f}",
            xy=(x1[-1], y1[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color="#f39c12",
        )
    if y2:
        ax.plot(x2, y2, label="Policy B", linewidth=2.2, color="#3498db")
        ax.fill_between(x2, y2, alpha=0.08, color="#3498db")
        ax.annotate(
            f"${y2[-1]:,.0f}",
            xy=(x2[-1], y2[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight=600,
            color="#3498db",
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Budget consumed ($)")
    ax.set_title("Budget consumed over steps", loc="left", fontsize=11, fontweight="bold", color="#111827")
    ax.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(False, axis="x")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e5e7eb")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


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
            rows.append(
                _tr(
                    [
                        _td(str(h["step"]), _CELL_STYLE),
                        _td(action, f"{_ACTION_CELL_STYLE};background:{action_color}"),
                        _td(f"{h['health_a']:.2f}", _CELL_STYLE),
                        _td(f"{h['health_b']:.2f}", _CELL_STYLE),
                        _td(f"{h['health_c']:.2f}", _CELL_STYLE),
                        _td(f"{h['budget']:.2f}", _CELL_STYLE),
                        _td(f"{h['reward']:+.3f}", _CELL_STYLE),
                    ]
                )
            )
        body = "".join(rows)
    return _table(head, body)


def render_side_panel(side: Dict, run: Dict, scenario_name: str) -> Tuple[str, str, str, str, str, str, str, str]:
    scenario_cfg = _task_config(scenario_name)
    global_step = int(run.get("step", 0) or 0)
    common = _common_provider_health(scenario_cfg, global_step)

    obs = side.get("obs", {}) or {}
    history = side.get("history", []) or []
    last = history[-1] if history else None

    grade = compute_grade(history) if history else {}
    adaptation = grade.get("adaptation_score") if history else None
    latency = float(last.get("latency_ms", 0.0) or 0.0) if last else None
    last_action = str(last.get("action")) if last else "—"
    budget_val = float(obs.get("budget_remaining", 0.0) or 0.0)
    reward_val = float(last.get("reward", 0.0) or 0.0) if last else None

    kpis = _kpi_grid(
        [
            ("Step", f"{global_step} / {MAX_STEPS}"),
            ("Last action", last_action),
            ("Latency (ms)", (f"{latency:.0f}" if latency is not None else "—")),
            ("Budget remaining", f"{budget_val:.2f}"),
            ("Reward", ((f"{reward_val:+.3f}") if reward_val is not None else "—")),
            ("Adaptation", ((f"{float(adaptation):.3f}") if adaptation is not None else "—")),
        ]
    )

    metrics = _summary_metrics(history)
    summary = _kpi_grid(
        [
            ("Failed %", f"{metrics['failed_pct']:.1f}%"),
            ("SLA breach %", f"{metrics['sla_breach_pct']:.1f}%"),
            ("Avg latency (ms)", f"{metrics['avg_latency_ms']:.1f}"),
        ]
    )
    summary = summary + render_data_quality_panel(history)

    grade_html = _GRADER_PENDING()
    if side.get("done") and history:
        grade_html = render_grader(grade)

    return (
        str(side.get("status", "")),
        render_common_providers(common) if global_step > 0 else _PROVIDER_EMPTY(),
        render_budget(obs) if obs else "",
        kpis,
        _step_badges(last),
        summary,
        render_history_table_compare(history),
        grade_html,
    )
