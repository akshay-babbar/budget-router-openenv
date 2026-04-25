from __future__ import annotations

import gradio as gr

# Minimal CSS overrides - theme handles most styling via _dark variants
LIGHT_CSS = """
/* Force light color scheme */
:root, .dark {
  color-scheme: light !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-primary-dark: #ffffff !important;
  --background-fill-secondary: #ffffff !important;
  --background-fill-secondary-dark: #ffffff !important;
}

/* Ensure all text is dark */
.gradio-container label,
.gradio-container span {
  color: #1f2937 !important;
}

/* Radio/checkbox label pills - force white background */
.gradio-container .wrap[data-testid=\"checkbox-group\"] label,
.gradio-container .wrap[data-testid=\"radio-group\"] label {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  color: #1f2937 !important;
}
.gradio-container .wrap[data-testid=\"checkbox-group\"] label.selected,
.gradio-container .wrap[data-testid=\"radio-group\"] label.selected {
  background: #eef2ff !important;
  border-color: #4f46e5 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: #1f2937 !important;
  background: #ffffff !important;
}
 
.gradio-container [data-testid=\"dropdown\"],
.gradio-container [data-testid=\"dropdown\"] * {
  background: #ffffff !important;
  color: #1f2937 !important;
}

/* Dropdown menu is sometimes portaled outside .gradio-container, so also target by role globally */
[role=\"listbox\"] {
  background: #ffffff !important;
  color: #1f2937 !important;
}

[role=\"option\"] {
  background: #ffffff !important;
  color: #1f2937 !important;
}

[role=\"option\"]:hover,
[role=\"option\"][aria-selected=\"true\"],
[role=\"option\"][data-highlighted],
[role=\"option\"][data-selected],
[role=\"option\"][data-state=\"checked\"],
[role=\"option\"].selected {
  background: #ffffff !important;
  color: #1f2937 !important;
}

/* KPI cards */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 6px;
}

.kpi-card {
  border: 1px solid #e5e7eb;
  background: #ffffff;
  border-radius: 10px;
  padding: 10px 12px;
}

.kpi-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 4px;
}

.kpi-value {
  font-size: 14px;
  font-weight: 600;
  color: #111827;
  line-height: 1.2;
  font-variant-numeric: tabular-nums;
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
