"""
Interactive overlay-plot widget for the downstream evaluation notebook.

Provides show_overlay_widget(), which renders a checkbox + button UI above
the overlay plots so the reader can select which tasks to display per model.
"""

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from .plots import plot_overlay

_TASK_LABELS = {"m_mmlu": "M-MMLU", "xcopa": "XCOPA", "belebele": "Belebele", "m_arc": "M-ARC"}
_TASKS       = ["m_mmlu", "xcopa", "belebele", "m_arc"]


def show_overlay_widget(models: list, data: dict) -> None:
    """
    Display an interactive overlay-plot selector.

    Parameters
    ----------
    models : list of model keys (e.g. ["fuxi", "apertus"])
    data   : dict keyed by model key; each value must contain the keys
             produced by the notebook's data-loading cells:
             cfg, df_eval, df_layer, df_grokking, checkpoints_all,
             token_counts, langs_sorted, eval_available, df_geometry.
    """
    checks = {
        m: {
            task: widgets.Checkbox(value=(task == "m_mmlu"),
                                   description=_TASK_LABELS[task],
                                   style={"description_width": "initial"},
                                   layout=widgets.Layout(width="auto", margin="0 16px 0 0"))
            for task in _TASKS
        }
        for m in models
    }

    header = widgets.HTML(
        "<div style='font-size:11px; font-weight:600; color:#888; "
        "text-transform:uppercase; letter-spacing:0.6px; "
        "border-bottom:1px solid #e0e0e0; padding-bottom:5px; margin-bottom:7px;'>"
        "Task selection</div>"
    )

    rows = [
        widgets.HBox(
            [widgets.HTML(
                f"<span style='font-size:13px; font-weight:500; color:#333; "
                f"min-width:190px; display:inline-block; padding-top:3px;'>"
                f"{data[m]['cfg']['model_label']}</span>"
            )]
            + list(checks[m].values()),
            layout=widgets.Layout(align_items="center", margin="0 0 3px 0")
        )
        for m in models
    ]

    btn       = widgets.Button(description="Generate plots",   button_style="primary",
                               layout=widgets.Layout(margin="0 8px 0 0", height="28px"))
    reset_btn = widgets.Button(description="Reset to default", button_style="warning",
                               layout=widgets.Layout(height="28px"))
    controls  = widgets.VBox(
        [header] + rows + [widgets.HBox([btn, reset_btn],
                                        layout=widgets.Layout(margin="8px 0 0 0"))],
        layout=widgets.Layout(padding="8px 16px 8px 16px", border="none", width="auto")
    )

    def draw(btn_event=None):
        btn.disabled = True
        btn.description = "⏳ Generating..."
        try:
            clear_output(wait=True)
            display(controls)
            plt.rcParams["figure.edgecolor"] = "white"
            for m in models:
                d   = data[m]
                cfg = d["cfg"]
                if not d["eval_available"]:
                    print(f"[{cfg['model_label']}] No eval data.")
                    continue
                selected = {t: cfg["task_languages"][t]
                            for t in _TASKS
                            if checks[m][t].value and t in cfg["task_languages"]}
                if not selected:
                    print(f"[{cfg['model_label']}] No tasks selected.")
                    continue
                plot_overlay(d["df_eval"], d["df_layer"], d["df_grokking"],
                             selected, cfg["random_chance"],
                             d["checkpoints_all"], d["token_counts"], d["langs_sorted"],
                             cfg["model_label"], cfg["plots_dir"], model_key=m,
                             df_geometry=d.get("df_geometry"))
        finally:
            btn.disabled = False
            btn.description = "Generate plots"

    def reset(btn_event=None):
        for m in models:
            for task, cb in checks[m].items():
                cb.value = (task == "m_mmlu")
        draw()

    btn.on_click(draw)
    reset_btn.on_click(reset)
    draw()
