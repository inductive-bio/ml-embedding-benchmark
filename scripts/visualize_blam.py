"""Visualizations for scripts/blam_analysis.py's posterior summaries.

Six outputs:
  - A static bar chart (PDF) of per-embedding latent ability (theta), with
    error bars from the posterior std dev.
  - An interactive HTML dashboard (--html) with a toggle between that same
    theta ranking, the per-endpoint specialization (u_ie), and (when
    --posterior-input's file exists) the raw per-endpoint MAE: pick an
    endpoint, see every embedding's u_mean/u_std or MAE ranked for it.
  - An animated GIF of just the theta ranking (--theta-gif), stepping down
    the ranked list one embedding at a time (top/best to bottom/worst) with
    that row highlighted -- embeds natively into tools like Google Slides.
  - A static bar chart (PNG, --compare-ecfp) of P(embedding > ECFP) per
    the paper's delta_{i,ECFP} construction (App. C.5), with a two-sided
    p-value analog for significance.
  - An animated GIF counterpart to that same chart (--compare-ecfp-gif),
    stepping through embeddings one at a time like --theta-gif.
  - A standalone interactive HTML explorer (--compare-ecfp-endpoints) of that
    same P(embedding > ECFP) question, but per existing endpoint rather than
    aggregated -- pick an endpoint, see which embeddings actually beat ECFP
    there and whether it's significant.

Usage:
    python scripts/visualize_blam.py
    python scripts/visualize_blam.py --input data/blam/theta_summary.csv --output data/blam/theta_ranking.pdf
    python scripts/visualize_blam.py --html
    python scripts/visualize_blam.py --theta-gif
    python scripts/visualize_blam.py --compare-ecfp
    python scripts/visualize_blam.py --compare-ecfp-gif
    python scripts/visualize_blam.py --compare-ecfp-endpoints
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

# Diverging pair (see dataviz skill, references/palette.md): theta is a
# sum-to-zero latent ability, so its sign (above/below the mean embedding)
# is the thing worth encoding, not just its magnitude.
COLOR_ABOVE = "#2a78d6"  # blue
COLOR_BELOW = "#e34948"  # red
COLOR_ERRORBAR = "#52514e"  # secondary ink
COLOR_BASELINE = "#c3c2b7"  # axis/baseline
COLOR_GRID = "#e1e0d9"  # hairline gridline
COLOR_TEXT_MUTED = "#898781"
COLOR_TEXT_PRIMARY = "#0b0b0b"
ACTIVE_WASH = "#fce8a8"  # highlight band behind the animated GIFs' active row
SURFACE = "#ffffff"

ENDPOINT_PREFIX = "reg_openadmet_"
FINETUNED_SUFFIX = "_finetuned"


def load_display_names(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def resolve_display(name: str, display_names: dict) -> tuple:
    """Maps a raw embedding/entity name to (display_name, is_finetuned).
    Fine-tuned entities are the base entity's name with a '_finetuned' suffix
    (see blam_analysis.py's --finetuned-models) -- stripped here to look up
    the base model's display name, then re-marked via a '(fine-tuned)' suffix
    and the returned is_finetuned flag (callers bold on it)."""
    is_finetuned = name.endswith(FINETUNED_SUFFIX)
    base = name[: -len(FINETUNED_SUFFIX)] if is_finetuned else name
    display = display_names.get(base, base)
    if is_finetuned:
        display += " (fine-tuned)"
    return display, is_finetuned


def plot_theta(df: pd.DataFrame, output: str, display_names: dict = None) -> None:
    display_names = display_names or {}
    df = df.sort_values("theta_mean", ascending=True).reset_index(drop=True)
    colors = [COLOR_ABOVE if v >= 0 else COLOR_BELOW for v in df["theta_mean"]]
    labels, bold_flags = zip(*(resolve_display(n, display_names) for n in df["embedding"]))

    fig_height = max(4, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(df))
    ax.barh(
        y, df["theta_mean"], height=0.6, color=colors,
        xerr=df["theta_std"], ecolor=COLOR_ERRORBAR, capsize=2, error_kw={"elinewidth": 1},
    )

    ax.axvline(0, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    for label, is_finetuned in zip(ax.get_yticklabels(), bold_flags):
        if is_finetuned:
            label.set_fontweight("bold")
    ax.set_ylim(-0.6, len(df) - 0.4)

    ax.set_xlabel("theta (latent ability, higher = better)", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label="Above average"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label="Below average"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=False,
        fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


def load_posterior(path: str) -> dict:
    # blam_analysis.py pickles posterior.joblib's "stats" entries as
    # EndpointStat under whatever module ran it (__main__), so joblib can
    # only unpickle them once that name is registered here. Import deferred
    # so plain theta/dashboard runs skip blam_analysis's jax/numpyro cost.
    import sys

    from blam_analysis import EndpointStat
    sys.modules["__main__"].EndpointStat = EndpointStat
    return joblib.load(path)


def compute_mae_summary(posterior: dict) -> pd.DataFrame:
    """Raw per-(endpoint, embedding) MAE -- EndpointStat.y_hat is exactly this
    (the paper's App. B.5 Step 2 per-embedding mean |error|), computed
    before any Bayesian fitting. mae_se is the standard error of that mean,
    from V_e's diagonal (Step 3's sampling covariance)."""
    rows = []
    for stat in posterior["stats"]:
        se = np.sqrt(np.diag(stat.V))
        for j, embedding in enumerate(stat.embeddings):
            rows.append({
                "endpoint": stat.name,
                "embedding": embedding,
                "mae_mean": float(stat.y_hat[j]),
                "mae_se": float(se[j]),
            })
    return pd.DataFrame(rows)


def compute_ecfp_comparison(posterior: dict, baseline: str, seed: int = 0) -> pd.DataFrame:
    """The paper's new-endpoint P(i > j) construction (Eq. 8, App. C.5): for each
    posterior draw s, sample a hypothetical new endpoint's discrimination and
    specialization noise from that draw's hyper-parameters, then

        P^(s) = Phi(beta_new^(s) * delta_{i,baseline}^(s) / (sqrt(2) * omega_new^(s)))

    averaged over draws. This is the probability embedding i beats `baseline`
    on a new, unseen endpoint, not just that its overall ability theta_i is
    higher -- it also accounts for how much a random endpoint's
    discrimination/noise could still favor either side.

    p_value is a two-sided p-value analog (Makowski et al. 2019's
    p_direction -> p-value mapping): p = 2 * min(P, 1-P), so p -> 0 as the
    posterior concentrates on one model winning and p -> 1 at P = 0.5."""
    global_embeddings = posterior["global_embeddings"]
    if baseline not in global_embeddings:
        raise ValueError(f"baseline '{baseline}' not found among fitted embeddings: {global_embeddings}")
    samples = posterior["samples"]
    theta = np.asarray(samples["theta"])  # (S, M)
    theta_baseline = theta[:, global_embeddings.index(baseline)]

    # One hypothetical new endpoint per posterior draw, shared across every
    # embedding compared against `baseline` in that draw -- matching the
    # doc's single beta_new^(s), omega_new^(s) per draw s.
    rng = np.random.default_rng(seed)
    sigma_beta = np.asarray(samples["sigma_beta"])  # (S,)
    sigma_omega = np.asarray(samples["sigma_omega"])  # (S,)
    beta_new = np.abs(rng.normal(0.0, sigma_beta))
    omega_new = np.abs(rng.normal(0.0, sigma_omega))

    rows = []
    for i, embedding in enumerate(global_embeddings):
        if embedding == baseline:
            continue
        delta = theta[:, i] - theta_baseline
        prob_draws = ndtr(beta_new * delta / (np.sqrt(2) * omega_new))
        prob_beat = float(prob_draws.mean())
        rows.append({
            "embedding": embedding,
            "prob_beat_baseline": prob_beat,
            "p_value": 2 * min(prob_beat, 1 - prob_beat),
            "delta_mean": float(delta.mean()),
        })
    return pd.DataFrame(rows).sort_values("prob_beat_baseline").reset_index(drop=True)


def compute_endpoint_ecfp_comparison(posterior: dict, baseline: str) -> pd.DataFrame:
    """Per-existing-endpoint P(i > baseline): did i actually outperform baseline
    on the endpoint e data already collected, given everything the model has
    learned about it -- NOT compute_ecfp_comparison's "would i beat baseline
    on a hypothetical new endpoint" question (that one legitimately
    integrates out u_ie/u_je as fresh, never-observed noise via resampled
    beta_new/omega_new). Reusing that same Phi(beta_e*delta/(sqrt2*omega_e))
    construction here with endpoint e's own fitted beta_e/omega_e would be a
    subtler version of the same mistake, since blam_analysis.py's
    compute_specialization() already sampled u_ie/u_je from their posterior
    conditional given endpoint e's actual data (posterior["u_by_endpoint"]).

    So this plugs those real per-draw u_ie/u_je into the paper's margin
    decomposition (Eq. 58):

        d_ije^(s) = beta_e^(s) * (theta_i^(s) - theta_j^(s)) + (u_je^(s) - u_ie^(s))

    and P(i > baseline) is the empirical fraction of draws with d_ije^(s) > 0
    -- no Phi() closed form needed, since it's just counting how often the
    model's own posterior says i came out ahead here.

    Long-form output: one row per (endpoint, embedding) actually fitted
    together on that endpoint (a handful of embeddings are missing from a
    handful of endpoints, so this only computes pairs that both exist
    there)."""
    global_embeddings = posterior["global_embeddings"]
    if baseline not in global_embeddings:
        raise ValueError(f"baseline '{baseline}' not found among fitted embeddings: {global_embeddings}")
    samples = posterior["samples"]
    u_by_endpoint = posterior["u_by_endpoint"]

    rows = []
    for stat in posterior["stats"]:
        if baseline not in stat.embeddings:
            continue
        beta_e = np.asarray(samples[f"beta__{stat.name}"])  # (S,)
        u_e = u_by_endpoint[stat.name]  # (S, n_e), column order matches stat.embeddings
        baseline_col = stat.embeddings.index(baseline)
        theta_baseline = stat.theta_samples[:, baseline_col]  # (S,)
        u_baseline = u_e[:, baseline_col]  # (S,)
        for col, embedding in enumerate(stat.embeddings):
            if embedding == baseline:
                continue
            d = beta_e * (stat.theta_samples[:, col] - theta_baseline) + (u_baseline - u_e[:, col])
            prob_beat = float((d > 0).mean())
            rows.append({
                "endpoint": stat.name,
                "embedding": embedding,
                "prob_beat_baseline": prob_beat,
                "p_value": 2 * min(prob_beat, 1 - prob_beat),
            })
    return pd.DataFrame(rows)


def plot_ecfp_comparison(df: pd.DataFrame, output: str, baseline: str, alpha: float, display_names: dict = None) -> None:
    display_names = display_names or {}

    def bar_color(row):
        if row["p_value"] >= alpha:
            return COLOR_TEXT_MUTED
        return COLOR_ABOVE if row["prob_beat_baseline"] >= 0.5 else COLOR_BELOW

    colors = [bar_color(row) for _, row in df.iterrows()]
    labels, bold_flags = zip(*(resolve_display(n, display_names) for n in df["embedding"]))
    baseline_label, _ = resolve_display(baseline, display_names)

    fig_height = max(4, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    y = range(len(df))
    ax.barh(y, df["prob_beat_baseline"], height=0.6, color=colors)

    ax.axvline(0.5, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    for label, is_finetuned in zip(ax.get_yticklabels(), bold_flags):
        if is_finetuned:
            label.set_fontweight("bold")
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])  # 0.5 must be a labeled tick -- it's the no-difference reference line

    ax.set_xlabel(f"P(embedding beats {baseline_label}) over BLAM posterior draws", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label=f"Significantly better (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label=f"Significantly worse (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TEXT_MUTED, label=f"Not significant (p ≥ {alpha:g})"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=False,
        fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"Wrote {output}")


def plot_ecfp_comparison_gif(
    df: pd.DataFrame, output: str, baseline: str, alpha: float, display_names: dict = None,
    frame_ms: int = 900, dpi: int = 110,
) -> None:
    """Animated-GIF counterpart to plot_ecfp_comparison: same P(embedding
    beats baseline) ranking for a hypothetical new/unseen endpoint
    (compute_ecfp_comparison's output), stepping top (most likely to beat
    baseline) to bottom with that row highlighted -- see plot_theta_gif's
    docstring for why a GIF over an HTML page."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    display_names = display_names or {}

    def bar_color(row):
        if row["p_value"] >= alpha:
            return COLOR_TEXT_MUTED
        return COLOR_ABOVE if row["prob_beat_baseline"] >= 0.5 else COLOR_BELOW

    colors = [bar_color(row) for _, row in df.iterrows()]
    labels, bold_flags = zip(*(resolve_display(n, display_names) for n in df["embedding"]))
    baseline_label, _ = resolve_display(baseline, display_names)
    values = df["prob_beat_baseline"].to_numpy()
    p_values = df["p_value"].to_numpy()
    n = len(df)

    fig_height = max(4, 0.18 * n)
    fig = plt.figure(figsize=(10.5, fig_height), dpi=dpi, facecolor=SURFACE)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(facecolor=SURFACE)

    y = range(n)
    bars = ax.barh(y, values, height=0.6, color=colors, zorder=2)

    ax.axvline(0.5, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    for label, is_finetuned in zip(ax.get_yticklabels(), bold_flags):
        if is_finetuned:
            label.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])  # 0.5 must be a labeled tick -- it's the no-difference reference line

    ax.set_xlabel(f"P(embedding beats {baseline_label}) over BLAM posterior draws", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)

    ax.set_title(
        f"BLAM: probability each embedding beats {baseline_label} (new/unseen endpoint)", loc="left",
        color=COLOR_TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label=f"Significantly better (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label=f"Significantly worse (p < {alpha:g})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_TEXT_MUTED, label=f"Not significant (p ≥ {alpha:g})"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=False,
        fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )
    fig.tight_layout()

    def render_frame() -> Image.Image:
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return Image.fromarray(buf, mode="RGBA").convert("RGB")

    tick_labels = ax.get_yticklabels()
    frames = [render_frame()]  # static lead-in frame, no row highlighted
    for row in range(n - 1, -1, -1):  # top (most likely to beat baseline) to bottom, matching the ranking's reading order
        span = ax.axhspan(row - 0.5, row + 0.5, color=ACTIVE_WASH, zorder=1)
        bars.patches[row].set_edgecolor(COLOR_TEXT_PRIMARY)
        bars.patches[row].set_linewidth(1.5)
        tick_labels[row].set_fontweight("bold")
        value_text = f"P={values[row]:.3f}, p={p_values[row]:.3f}"
        ann = ax.annotate(
            value_text, xy=(values[row], row), xytext=(8 if values[row] >= 0.5 else -8, 0),
            textcoords="offset points", va="center", ha="left" if values[row] >= 0.5 else "right",
            fontsize=8, fontweight="bold", color=COLOR_TEXT_PRIMARY,
        )

        frames.append(render_frame())

        span.remove()
        ann.remove()
        bars.patches[row].set_edgecolor("none")
        bars.patches[row].set_linewidth(0)
        tick_labels[row].set_fontweight("bold" if bold_flags[row] else "normal")

    plt.close(fig)

    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True,
    )
    print(f"Wrote {output} ({n} frames)")


def endpoint_short_label(name: str) -> str:
    if name.startswith(ENDPOINT_PREFIX):
        name = name[len(ENDPOINT_PREFIX):]
    return name.replace("_", " ")


def named_row(raw_name, value, std, display_names: dict) -> dict:
    display, bold = resolve_display(raw_name, display_names)
    return {"name": display, "value": float(value), "std": float(std), "bold": bold}


def compute_theta_rows(theta_df: pd.DataFrame, display_names: dict = None) -> list:
    """Theta rows shared by the full dashboard's "Latent ability" mode and the
    standalone theta-animation page: one row per embedding, ranked best-first."""
    display_names = display_names or {}
    return [
        named_row(r.embedding, r.theta_mean, r.theta_std, display_names)
        for r in theta_df.sort_values("theta_mean", ascending=False).itertuples(index=False)
    ]


def build_dashboard_payload(
    theta_df: pd.DataFrame, spec_df: pd.DataFrame, endpoint_df: pd.DataFrame, mae_df: pd.DataFrame = None,
    display_names: dict = None,
) -> dict:
    """Rows for all dashboard modes, sharing a common {name, value, std, bold}
    shape so the JS chart/table renderers don't need to know which mode
    they're drawing. Endpoints keep endpoint_df's row order (dataset
    groupings stay together); each endpoint's embeddings/mae rows are
    pre-sorted (u_mean descending, mae_mean ascending) -- the chart re-sorts
    per endpoint rather than pinning a fixed row order. mae_df is optional
    since it requires posterior.joblib (unlike theta/spec/endpoint, which
    only need blam_analysis.py's CSV outputs) -- omitted, the dashboard just
    doesn't offer the raw-MAE mode."""
    display_names = display_names or {}

    theta_rows = compute_theta_rows(theta_df, display_names)

    endpoints = []
    for row in endpoint_df.itertuples(index=False):
        rows = spec_df[spec_df["endpoint"] == row.endpoint].sort_values("u_mean", ascending=False)
        entry = {
            "key": row.endpoint,
            "label": endpoint_short_label(row.endpoint),
            "mu_mean": float(row.mu_mean),
            "embeddings": [named_row(r.embedding, r.u_mean, r.u_std, display_names) for r in rows.itertuples(index=False)],
        }
        if mae_df is not None:
            mrows = mae_df[mae_df["endpoint"] == row.endpoint].sort_values("mae_mean", ascending=True)
            entry["mae"] = [named_row(r.embedding, r.mae_mean, r.mae_se, display_names) for r in mrows.itertuples(index=False)]
        endpoints.append(entry)
    return {"theta": theta_rows, "endpoints": endpoints, "has_mae": mae_df is not None}


def render_dashboard_html(payload: dict) -> str:
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BLAM dashboard</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --surface-2: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --strong: #2a78d6;
    --weak: #e34948;
    --neutral: #898781;
    --accent-wash: rgba(42,120,214,0.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --surface-2: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --strong: #3987e5;
      --weak: #e66767;
      --neutral: #a3a19a;
      --accent-wash: rgba(57,135,229,0.16);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface-2);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .layout {{
    display: flex;
    min-height: 100vh;
  }}
  .sidebar {{
    width: 260px;
    flex-shrink: 0;
    background: var(--surface-1);
    border-right: 1px solid var(--border);
    padding: 16px 0;
    overflow-y: auto;
  }}
  .sidebar h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin: 0 16px 8px;
  }}
  .endpoint-btn {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    border-left: 3px solid transparent;
    padding: 7px 16px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
    font-family: inherit;
  }}
  .endpoint-btn:hover {{ background: var(--accent-wash); }}
  .endpoint-btn.active {{ background: var(--accent-wash); border-left-color: var(--strong); }}
  .endpoint-metric {{ color: var(--text-muted); font-size: 11px; flex-shrink: 0; padding-left: 8px; }}
  main {{ flex: 1; padding: 24px 32px; min-width: 0; }}
  .mode-toggle {{ display: flex; gap: 6px; margin-bottom: 18px; }}
  .mode-btn {{
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 6px 12px; cursor: pointer;
    font-family: inherit;
  }}
  .mode-btn.active {{ color: var(--text-primary); border-color: var(--strong); background: var(--accent-wash); }}
  h1 {{ font-size: 16px; font-weight: 700; margin: 0 0 2px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 12px; margin: 0 0 20px; max-width: 640px; line-height: 1.5; }}
  .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .view-toggle {{
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
    font-family: inherit;
  }}
  .chart-wrap {{ overflow-x: auto; }}
  svg text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-primary); }}
  .row-label {{ fill: var(--text-primary); font-size: 11px; }}
  .tick-label {{ fill: var(--text-muted); font-size: 10px; }}
  .bar-strong {{ fill: var(--strong); }}
  .bar-weak {{ fill: var(--weak); }}
  .bar-neutral {{ fill: var(--neutral); }}
  .bar-strong.hovered, .bar-weak.hovered, .bar-neutral.hovered {{ filter: brightness(1.15); }}
  .errorbar {{ stroke: var(--text-secondary); stroke-width: 1; }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .baseline {{ stroke: var(--baseline); stroke-width: 1; }}
  .hit-row {{ fill: transparent; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  table th, table td {{ text-align: left; padding: 5px 12px; border-bottom: 1px solid var(--gridline); }}
  table th {{ color: var(--text-muted); font-weight: 500; }}
  #tooltip {{
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; box-shadow: 0 2px 8px var(--border);
  }}
  #tooltip .tt-value {{ color: var(--text-primary); font-weight: 700; }}
  #tooltip .tt-name {{ color: var(--text-secondary); }}
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar" id="sidebar">
    <h2>Endpoints</h2>
    <div id="endpoint-list"></div>
  </nav>
  <main>
    <div class="mode-toggle" role="tablist">
      <button class="mode-btn active" id="mode-theta" data-mode="theta">Latent ability</button>
      <button class="mode-btn" id="mode-specialization" data-mode="specialization">Per-endpoint specialization</button>
      <button class="mode-btn" id="mode-mae" data-mode="mae" style="display:none">Raw MAE per endpoint</button>
    </div>
    <h1 id="chart-title"></h1>
    <p class="subtitle" id="chart-subtitle"></p>
    <div class="toolbar">
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:var(--strong)"></span><span id="legend-strong"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--weak)"></span><span id="legend-weak"></span></span>
      </div>
      <button class="view-toggle" id="view-toggle">View as table</button>
    </div>
    <div class="chart-wrap" id="chart-wrap">
      <svg id="chart" width="100%"></svg>
    </div>
    <div id="table-wrap" style="display:none"></div>
  </main>
</div>
<div id="tooltip"></div>
<script>
const DATA = {data_json};

const CHART_WIDTH = 920;
const LEFT_MARGIN = 220;
const RIGHT_MARGIN = 60;
const TOP_MARGIN = 12;
const BOTTOM_MARGIN = 30;
const ROW_H = 16;
const BAR_H = 11;
const CORNER_R = 4;

const MODES = {{
  theta: {{
    title: "BLAM latent ability (theta) per embedding",
    subtitle: "theta: overall latent ability across all endpoints (sum-to-zero across the "
      + "compared embeddings). Higher = better. Error bars are the posterior std dev.",
    strongLabel: "Above average (better)",
    weakLabel: "Below average (worse)",
    axisMode: "diverging",
    isStrong: (v, i) => v >= 0,
    symbol: "\\u03b8",
  }},
  specialization: {{
    title: "Per-embedding specialization",
    subtitle: "u (specialization): how much each embedding's error on this endpoint deviates "
      + "from what its overall latent ability alone would predict. Negative = does better than "
      + "expected here (a relative strength for this endpoint); positive = does worse than "
      + "expected here. Error bars are the posterior std dev.",
    strongLabel: "Stronger than expected here",
    weakLabel: "Weaker than expected here",
    axisMode: "diverging",
    isStrong: (v, i) => v < 0,
    symbol: "u",
  }},
  mae: {{
    title: "Raw MAE per endpoint",
    subtitle: "Mean absolute error of each embedding's best-performing head (rf/ridge/knn/tabicl, "
      + "picked by cross-validation score) on this endpoint's held-out test compounds -- the "
      + "un-modeled numbers the BLAM fit is built from. Lower is better. Error bars are the "
      + "standard error of the mean.",
    strongLabel: "Best (lowest MAE)",
    weakLabel: "",
    axisMode: "zero",
    isStrong: (v, i) => i === 0,
    falseClass: "bar-neutral",
    symbol: "MAE",
  }},
}};

function barClassFor(r, i) {{
  const cfg = MODES[currentMode];
  return cfg.isStrong(r.value, i) ? "bar-strong" : (cfg.falseClass || "bar-weak");
}}

let currentMode = "theta";
let currentKey = DATA.endpoints[0].key;
let showTable = false;

function formatValue(v, maxAbs) {{
  const decimals = maxAbs < 0.01 ? 4 : (maxAbs < 0.1 ? 3 : 2);
  return v.toFixed(decimals);
}}

function niceTicks(maxAbs) {{
  return [-maxAbs, -maxAbs / 2, 0, maxAbs / 2, maxAbs];
}}

function svgEl(tag, attrs) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}}

function roundedBarPath(baselineX, endX, y, h, r) {{
  const left = Math.min(baselineX, endX);
  const right = Math.max(baselineX, endX);
  const rr = Math.min(r, (right - left) / 2, h / 2);
  if (endX >= baselineX) {{
    return `M ${{left}},${{y}} L ${{right - rr}},${{y}} Q ${{right}},${{y}} ${{right}},${{y + rr}} `
         + `L ${{right}},${{y + h - rr}} Q ${{right}},${{y + h}} ${{right - rr}},${{y + h}} L ${{left}},${{y + h}} Z`;
  }}
  return `M ${{right}},${{y}} L ${{left + rr}},${{y}} Q ${{left}},${{y}} ${{left}},${{y + rr}} `
       + `L ${{left}},${{y + h - rr}} Q ${{left}},${{y + h}} ${{left + rr}},${{y + h}} L ${{right}},${{y + h}} Z`;
}}

function renderSidebar() {{
  const list = document.getElementById("endpoint-list");
  list.innerHTML = "";
  for (const ep of DATA.endpoints) {{
    const btn = document.createElement("button");
    btn.className = "endpoint-btn" + (ep.key === currentKey ? " active" : "");
    btn.dataset.key = ep.key;
    const name = document.createElement("span");
    name.textContent = ep.label;
    const metric = document.createElement("span");
    metric.className = "endpoint-metric";
    metric.textContent = "\\u03bc " + ep.mu_mean.toFixed(2);
    metric.title = "Endpoint difficulty (mean log\\u2081\\u2080 MAE)";
    btn.appendChild(name);
    btn.appendChild(metric);
    btn.addEventListener("click", () => selectEndpoint(ep.key));
    list.appendChild(btn);
  }}
}}

function getCurrentRows() {{
  if (currentMode === "theta") return DATA.theta;
  if (currentMode === "mae") return DATA.endpoints.find(e => e.key === currentKey).mae;
  return DATA.endpoints.find(e => e.key === currentKey).embeddings;
}}

function updateHeader() {{
  const cfg = MODES[currentMode];
  let title = cfg.title;
  if (currentMode === "specialization" || currentMode === "mae") {{
    title += " \\u2013 " + DATA.endpoints.find(e => e.key === currentKey).label;
  }}
  document.getElementById("chart-title").textContent = title;
  document.getElementById("chart-subtitle").textContent = cfg.subtitle;
  document.getElementById("legend-strong").textContent = cfg.strongLabel;
  document.getElementById("legend-weak").textContent = cfg.weakLabel;
  document.querySelector(".legend-item:nth-child(2)").style.display = cfg.weakLabel ? "flex" : "none";
}}

function renderCurrent() {{
  const rows = getCurrentRows();
  if (showTable) renderTable(rows); else renderChart(rows);
}}

function switchMode(mode) {{
  currentMode = mode;
  for (const btn of document.querySelectorAll(".mode-btn")) {{
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }}
  document.getElementById("sidebar").style.display = mode === "theta" ? "none" : "block";
  updateHeader();
  renderCurrent();
}}

function selectEndpoint(key) {{
  currentKey = key;
  for (const btn of document.querySelectorAll(".endpoint-btn")) {{
    btn.classList.toggle("active", btn.dataset.key === key);
  }}
  updateHeader();
  renderCurrent();
}}

function renderChart(rows) {{
  const svg = document.getElementById("chart");
  svg.innerHTML = "";
  const n = rows.length;
  const plotWidth = CHART_WIDTH - LEFT_MARGIN - RIGHT_MARGIN;
  const zeroBased = MODES[currentMode].axisMode === "zero";
  const maxAbs = zeroBased
    ? Math.max(...rows.map(r => r.value + r.std), 1e-9)
    : Math.max(...rows.map(r => Math.abs(r.value) + r.std), 1e-9);
  const height = TOP_MARGIN + n * ROW_H + BOTTOM_MARGIN;
  svg.setAttribute("viewBox", `0 0 ${{CHART_WIDTH}} ${{height}}`);
  svg.setAttribute("height", height);

  const baselineX = zeroBased ? LEFT_MARGIN : LEFT_MARGIN + plotWidth / 2;
  const scale = zeroBased
    ? (v => LEFT_MARGIN + (v / maxAbs) * plotWidth)
    : (v => baselineX + (v / maxAbs) * (plotWidth / 2));
  const ticks = zeroBased ? [0, maxAbs / 4, maxAbs / 2, (3 * maxAbs) / 4, maxAbs] : niceTicks(maxAbs);
  const plotBottom = TOP_MARGIN + n * ROW_H;

  for (const t of ticks) {{
    const x = scale(t);
    svg.appendChild(svgEl("line", {{
      class: "gridline", x1: x, x2: x, y1: TOP_MARGIN, y2: plotBottom,
    }}));
    const label = svgEl("text", {{
      class: "tick-label", x: x, y: plotBottom + 16, "text-anchor": "middle",
    }});
    label.textContent = formatValue(t, maxAbs);
    svg.appendChild(label);
  }}

  rows.forEach((r, i) => {{
    const y = TOP_MARGIN + i * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const endX = scale(r.value);
    const cls = barClassFor(r, i);

    const hit = svgEl("rect", {{
      class: "hit-row", x: 0, y: y, width: CHART_WIDTH, height: ROW_H,
    }});

    const bar = svgEl("path", {{ class: cls, d: roundedBarPath(baselineX, endX, barY, BAR_H, CORNER_R) }});

    const loX = scale(r.value - r.std);
    const hiX = scale(r.value + r.std);
    const cy = y + ROW_H / 2;
    const errGroup = svgEl("g", {{ class: "errorbar" }});
    errGroup.appendChild(svgEl("line", {{ x1: loX, x2: hiX, y1: cy, y2: cy }}));
    errGroup.appendChild(svgEl("line", {{ x1: loX, x2: loX, y1: cy - 3, y2: cy + 3 }}));
    errGroup.appendChild(svgEl("line", {{ x1: hiX, x2: hiX, y1: cy - 3, y2: cy + 3 }}));

    const label = svgEl("text", {{
      class: "row-label", x: LEFT_MARGIN - 10, y: cy + 4, "text-anchor": "end",
      "font-weight": r.bold ? "700" : "400",
    }});
    label.textContent = r.name;

    hit.addEventListener("pointerenter", () => bar.classList.add("hovered"));
    hit.addEventListener("pointerleave", () => {{ bar.classList.remove("hovered"); hideTooltip(); }});
    hit.addEventListener("pointermove", ev => showTooltip(ev, r, maxAbs));

    // hit-row goes on top (last) so it wins hit-testing over the bar/label
    // beneath it -- pointerenter/leave don't bubble between siblings, so a
    // listener on a lower rect would never fire while the pointer is over
    // the (higher, unlistened) bar.
    svg.appendChild(bar);
    svg.appendChild(errGroup);
    svg.appendChild(label);
    svg.appendChild(hit);
  }});

  svg.appendChild(svgEl("line", {{
    class: "baseline", x1: baselineX, x2: baselineX, y1: TOP_MARGIN, y2: plotBottom,
  }}));
}}

function showTooltip(ev, r, maxAbs) {{
  const tt = document.getElementById("tooltip");
  tt.innerHTML = "";
  const name = document.createElement("div");
  name.className = "tt-name";
  name.textContent = r.name;
  const value = document.createElement("div");
  value.className = "tt-value";
  value.textContent = `${{MODES[currentMode].symbol}} = ${{formatValue(r.value, maxAbs)}} \\u00b1 ${{formatValue(r.std, maxAbs)}}`;
  tt.appendChild(name);
  tt.appendChild(value);
  tt.style.display = "block";
  tt.style.left = (ev.clientX + 14) + "px";
  tt.style.top = (ev.clientY + 14) + "px";
}}

function hideTooltip() {{
  document.getElementById("tooltip").style.display = "none";
}}

function renderTable(rows) {{
  const wrap = document.getElementById("table-wrap");
  wrap.innerHTML = "";
  const symbol = MODES[currentMode].symbol;
  const table = document.createElement("table");
  const thead = document.createElement("tr");
  for (const h of ["", "Embedding", `${{symbol}}_mean`, `${{symbol}}_std`]) {{
    const th = document.createElement("th");
    th.textContent = h;
    thead.appendChild(th);
  }}
  table.appendChild(thead);
  rows.forEach((r, i) => {{
    const tr = document.createElement("tr");
    const swatchCell = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(--${{barClassFor(r, i).replace("bar-", "")}})`;
    swatchCell.appendChild(swatch);
    const nameCell = document.createElement("td");
    nameCell.textContent = r.name;
    nameCell.style.fontWeight = r.bold ? "700" : "400";
    const meanCell = document.createElement("td");
    meanCell.textContent = r.value.toFixed(4);
    const stdCell = document.createElement("td");
    stdCell.textContent = r.std.toFixed(4);
    tr.appendChild(swatchCell);
    tr.appendChild(nameCell);
    tr.appendChild(meanCell);
    tr.appendChild(stdCell);
    table.appendChild(tr);
  }});
  wrap.appendChild(table);
}}

document.getElementById("view-toggle").addEventListener("click", () => {{
  showTable = !showTable;
  document.getElementById("view-toggle").textContent = showTable ? "View as chart" : "View as table";
  document.getElementById("chart-wrap").style.display = showTable ? "none" : "block";
  document.getElementById("table-wrap").style.display = showTable ? "block" : "none";
  renderCurrent();
}});

document.getElementById("mode-theta").addEventListener("click", () => switchMode("theta"));
document.getElementById("mode-specialization").addEventListener("click", () => switchMode("specialization"));
document.getElementById("mode-mae").addEventListener("click", () => switchMode("mae"));
if (DATA.has_mae) document.getElementById("mode-mae").style.display = "inline-block";

renderSidebar();
switchMode("theta");
</script>
</body>
</html>
"""


def write_dashboard_html(
    theta_df: pd.DataFrame, spec_df: pd.DataFrame, endpoint_df: pd.DataFrame, output: str, mae_df: pd.DataFrame = None,
    display_names: dict = None,
) -> None:
    payload = build_dashboard_payload(theta_df, spec_df, endpoint_df, mae_df, display_names)
    with open(output, "w") as f:
        f.write(render_dashboard_html(payload))
    print(f"Wrote {output}")


def plot_theta_gif(
    df: pd.DataFrame, output: str, display_names: dict = None, frame_ms: int = 900, dpi: int = 110,
) -> None:
    """Animated GIF of plot_theta's ranking: one frame per embedding, walking
    top (best) to bottom (worst) with that row highlighted and its value
    called out. Unlike the HTML dashboard, a GIF is something Google Slides
    embeds natively (Insert > Image) and autoplays in Present mode."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    display_names = display_names or {}
    df = df.sort_values("theta_mean", ascending=True).reset_index(drop=True)
    labels, bold_flags = zip(*(resolve_display(n, display_names) for n in df["embedding"]))
    values = df["theta_mean"].to_numpy()
    stds = df["theta_std"].to_numpy()
    n = len(df)
    colors = [COLOR_ABOVE if v >= 0 else COLOR_BELOW for v in values]

    fig_height = max(4, 0.18 * n)
    fig = plt.figure(figsize=(9, fig_height), dpi=dpi, facecolor=SURFACE)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(facecolor=SURFACE)

    y = range(n)
    bars = ax.barh(
        y, values, height=0.6, color=colors,
        xerr=stds, ecolor=COLOR_ERRORBAR, capsize=2, error_kw={"elinewidth": 1}, zorder=2,
    )

    ax.axvline(0, color=COLOR_BASELINE, linewidth=1, zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_TEXT_PRIMARY)
    for label, is_finetuned in zip(ax.get_yticklabels(), bold_flags):
        if is_finetuned:
            label.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 0.4)

    ax.set_xlabel("theta (latent ability, higher = better)", color=COLOR_TEXT_MUTED, fontsize=9)
    ax.tick_params(axis="x", colors=COLOR_TEXT_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)

    ax.set_title(
        "BLAM latent ability (theta) per embedding", loc="left",
        color=COLOR_TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ABOVE, label="Above average"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BELOW, label="Below average"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", frameon=False,
        fontsize=8, labelcolor=COLOR_TEXT_MUTED,
    )
    fig.tight_layout()

    def render_frame() -> Image.Image:
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        return Image.fromarray(buf, mode="RGBA").convert("RGB")

    tick_labels = ax.get_yticklabels()
    frames = [render_frame()]  # static lead-in frame, no row highlighted
    for row in range(n - 1, -1, -1):  # top (best) to bottom (worst), matching the ranking's reading order
        span = ax.axhspan(row - 0.5, row + 0.5, color=ACTIVE_WASH, zorder=1)
        bars.patches[row].set_edgecolor(COLOR_TEXT_PRIMARY)
        bars.patches[row].set_linewidth(1.5)
        tick_labels[row].set_fontweight("bold")
        value_text = f"{values[row]:+.3f} ± {stds[row]:.3f}"
        ann = ax.annotate(
            value_text, xy=(values[row], row), xytext=(8 if values[row] >= 0 else -8, 0),
            textcoords="offset points", va="center", ha="left" if values[row] >= 0 else "right",
            fontsize=8, fontweight="bold", color=COLOR_TEXT_PRIMARY,
        )

        frames.append(render_frame())

        span.remove()
        ann.remove()
        bars.patches[row].set_edgecolor("none")
        bars.patches[row].set_linewidth(0)
        tick_labels[row].set_fontweight("bold" if bold_flags[row] else "normal")

    plt.close(fig)

    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0, optimize=True,
    )
    print(f"Wrote {output} ({n} frames)")


def _comparison_row_dicts(df: pd.DataFrame, display_names: dict) -> list:
    rows = df.sort_values("prob_beat_baseline", ascending=False)
    row_dicts = []
    for r in rows.itertuples(index=False):
        display, bold = resolve_display(r.embedding, display_names)
        row_dicts.append({"name": display, "value": float(r.prob_beat_baseline), "p_value": float(r.p_value), "bold": bold})
    return row_dicts


def build_endpoint_comparison_payload(
    comparison_df: pd.DataFrame, endpoint_df: pd.DataFrame, baseline: str, alpha: float, display_names: dict = None,
    new_endpoint_df: pd.DataFrame = None,
) -> dict:
    """Rows for the standalone per-endpoint ECFP-comparison explorer. Endpoints
    keep endpoint_df's row order (matches the main dashboard); each endpoint's
    rows are pre-sorted by prob_beat_baseline descending, same convention as
    build_dashboard_payload's specialization rows (row 0 = top of chart).

    new_endpoint_df (compute_ecfp_comparison's output, the hypothetical-new-
    endpoint construction) is optional and, when given, is prepended as a
    synthetic endpoint entry -- letting the same sidebar toggle between "did i
    beat baseline on this real endpoint" (every other entry, from
    compute_endpoint_ecfp_comparison) and "would i beat baseline on an
    unseen endpoint" (this one) without leaving the page. It has no mu_mean
    (no real endpoint difficulty exists for a hypothetical endpoint)."""
    display_names = display_names or {}
    baseline_label, _ = resolve_display(baseline, display_names)
    endpoints = []
    if new_endpoint_df is not None:
        endpoints.append({
            "key": "__new_endpoint__",
            "label": "New / unseen endpoint",
            "mu_mean": None,
            "rows": _comparison_row_dicts(new_endpoint_df, display_names),
            "synthetic": True,
        })
    for row in endpoint_df.itertuples(index=False):
        rows = comparison_df[comparison_df["endpoint"] == row.endpoint]
        endpoints.append({
            "key": row.endpoint,
            "label": endpoint_short_label(row.endpoint),
            "mu_mean": float(row.mu_mean),
            "rows": _comparison_row_dicts(rows, display_names),
        })
    return {"baseline": baseline_label, "alpha": alpha, "endpoints": endpoints}


def render_endpoint_comparison_html(payload: dict) -> str:
    data_json = json.dumps(payload).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BLAM ECFP comparison per endpoint</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --surface-2: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --strong: #2a78d6;
    --weak: #e34948;
    --neutral: #898781;
    --accent-wash: rgba(42,120,214,0.12);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --surface-2: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --strong: #3987e5;
      --weak: #e66767;
      --neutral: #a3a19a;
      --accent-wash: rgba(57,135,229,0.16);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--surface-2);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 260px;
    flex-shrink: 0;
    background: var(--surface-1);
    border-right: 1px solid var(--border);
    padding: 16px 0;
    overflow-y: auto;
  }}
  .sidebar h2 {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-muted);
    margin: 0 16px 8px;
  }}
  .endpoint-btn {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    border-left: 3px solid transparent;
    padding: 7px 16px;
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
    font-family: inherit;
  }}
  .endpoint-btn:hover {{ background: var(--accent-wash); }}
  .endpoint-btn.active {{ background: var(--accent-wash); border-left-color: var(--strong); }}
  .endpoint-btn.synthetic {{ font-style: italic; border-bottom: 1px solid var(--border); margin-bottom: 4px; padding-bottom: 9px; }}
  .endpoint-metric {{ color: var(--text-muted); font-size: 11px; flex-shrink: 0; padding-left: 8px; }}
  main {{ flex: 1; padding: 24px 32px; min-width: 0; }}
  h1 {{ font-size: 16px; font-weight: 700; margin: 0 0 2px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 12px; margin: 0 0 20px; max-width: 640px; line-height: 1.5; }}
  .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .view-toggle {{
    font-size: 12px; color: var(--text-secondary); background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
    font-family: inherit;
  }}
  .chart-wrap {{ overflow-x: auto; }}
  svg text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--text-primary); }}
  .row-label {{ fill: var(--text-primary); font-size: 11px; }}
  .tick-label {{ fill: var(--text-muted); font-size: 10px; }}
  .bar-strong {{ fill: var(--strong); }}
  .bar-weak {{ fill: var(--weak); }}
  .bar-neutral {{ fill: var(--neutral); }}
  .bar-strong.hovered, .bar-weak.hovered, .bar-neutral.hovered {{ filter: brightness(1.15); }}
  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .baseline {{ stroke: var(--baseline); stroke-width: 1; }}
  .hit-row {{ fill: transparent; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  table th, table td {{ text-align: left; padding: 5px 12px; border-bottom: 1px solid var(--gridline); }}
  table th {{ color: var(--text-muted); font-weight: 500; }}
  #tooltip {{
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
    padding: 6px 10px; font-size: 12px; box-shadow: 0 2px 8px var(--border);
  }}
  #tooltip .tt-value {{ color: var(--text-primary); font-weight: 700; }}
  #tooltip .tt-name {{ color: var(--text-secondary); }}
</style>
</head>
<body>
<div class="layout">
  <nav class="sidebar" id="sidebar">
    <h2>Endpoints</h2>
    <div id="endpoint-list"></div>
  </nav>
  <main>
    <h1 id="chart-title"></h1>
    <p class="subtitle" id="chart-subtitle"></p>
    <div class="toolbar">
      <div class="legend">
        <span class="legend-item"><span class="swatch" style="background:var(--strong)"></span><span id="legend-strong"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--weak)"></span><span id="legend-weak"></span></span>
        <span class="legend-item"><span class="swatch" style="background:var(--neutral)"></span><span id="legend-neutral"></span></span>
      </div>
      <button class="view-toggle" id="view-toggle">View as table</button>
    </div>
    <div class="chart-wrap" id="chart-wrap">
      <svg id="chart" width="100%"></svg>
    </div>
    <div id="table-wrap" style="display:none"></div>
  </main>
</div>
<div id="tooltip"></div>
<script>
const DATA = {data_json};

const CHART_WIDTH = 920;
const LEFT_MARGIN = 220;
const RIGHT_MARGIN = 60;
const TOP_MARGIN = 12;
const BOTTOM_MARGIN = 30;
const ROW_H = 16;
const BAR_H = 11;
const CORNER_R = 4;
const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1;
const BASELINE_VALUE = 0.5;

let currentKey = DATA.endpoints[0].key;
let showTable = false;

function svgEl(tag, attrs) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}}

function roundedBarPath(baselineX, endX, y, h, r) {{
  const left = Math.min(baselineX, endX);
  const right = Math.max(baselineX, endX);
  const rr = Math.min(r, (right - left) / 2, h / 2);
  if (endX >= baselineX) {{
    return `M ${{left}},${{y}} L ${{right - rr}},${{y}} Q ${{right}},${{y}} ${{right}},${{y + rr}} `
         + `L ${{right}},${{y + h - rr}} Q ${{right}},${{y + h}} ${{right - rr}},${{y + h}} L ${{left}},${{y + h}} Z`;
  }}
  return `M ${{right}},${{y}} L ${{left + rr}},${{y}} Q ${{left}},${{y}} ${{left}},${{y + rr}} `
       + `L ${{left}},${{y + h - rr}} Q ${{left}},${{y + h}} ${{left + rr}},${{y + h}} L ${{right}},${{y + h}} Z`;
}}

function currentEndpoint() {{
  return DATA.endpoints.find(e => e.key === currentKey);
}}

function barClass(r) {{
  if (r.p_value >= DATA.alpha) return "bar-neutral";
  return r.value >= BASELINE_VALUE ? "bar-strong" : "bar-weak";
}}

function renderSidebar() {{
  const list = document.getElementById("endpoint-list");
  list.innerHTML = "";
  for (const ep of DATA.endpoints) {{
    const btn = document.createElement("button");
    btn.className = "endpoint-btn" + (ep.key === currentKey ? " active" : "") + (ep.synthetic ? " synthetic" : "");
    btn.dataset.key = ep.key;
    const name = document.createElement("span");
    name.textContent = ep.label;
    const metric = document.createElement("span");
    metric.className = "endpoint-metric";
    metric.textContent = ep.mu_mean === null ? "" : "\\u03bc " + ep.mu_mean.toFixed(2);
    metric.title = "Endpoint difficulty (mean log\\u2081\\u2080 MAE)";
    btn.appendChild(name);
    btn.appendChild(metric);
    btn.addEventListener("click", () => selectEndpoint(ep.key));
    list.appendChild(btn);
  }}
}}

function updateHeader() {{
  const ep = currentEndpoint();
  document.getElementById("chart-title").textContent = `P(embedding beats ${{DATA.baseline}}) \\u2013 ${{ep.label}}`;
  document.getElementById("chart-subtitle").textContent = ep.synthetic
    ? "Per posterior draw, a fresh discrimination/noise (beta_new, omega_new) is sampled from the fitted "
      + "hyper-priors and P = Phi(beta_new * (theta_i - theta_" + DATA.baseline + ") / (sqrt(2) * omega_new)) -- "
      + "this is the model's answer for a hypothetical endpoint it has never seen (the paper's Eq. 8). "
      + " Significance uses a two-sided p-value analog: p = 2 * min(P, 1-P), flagged "
      + `at alpha = ${{DATA.alpha}}.`
    : "Per posterior draw, using this endpoint's own fitted theta/beta_e and already-recovered specialization "
      + "(u_ie), P = fraction of draws where i's actual margin over " + DATA.baseline + " on this endpoint's data "
      + "is positive (the paper's d_ije = beta_e*(theta_i-theta_j) + (u_je-u_ie)) -- this reflects "
      + "what was actually observed here, not a hypothetical rerun. Significance uses a two-sided p-value "
      + `analog: p = 2 * min(P, 1-P), flagged at alpha = ${{DATA.alpha}}.`;
  document.getElementById("legend-strong").textContent = `Significantly better (p < ${{DATA.alpha}})`;
  document.getElementById("legend-weak").textContent = `Significantly worse (p < ${{DATA.alpha}})`;
  document.getElementById("legend-neutral").textContent = `Not significant (p \\u2265 ${{DATA.alpha}})`;
}}

function renderCurrent() {{
  const rows = currentEndpoint().rows;
  if (showTable) renderTable(rows); else renderChart(rows);
}}

function selectEndpoint(key) {{
  currentKey = key;
  for (const btn of document.querySelectorAll(".endpoint-btn")) {{
    btn.classList.toggle("active", btn.dataset.key === key);
  }}
  updateHeader();
  renderCurrent();
}}

function renderChart(rows) {{
  const svg = document.getElementById("chart");
  svg.innerHTML = "";
  const n = rows.length;
  const plotWidth = CHART_WIDTH - LEFT_MARGIN - RIGHT_MARGIN;
  const height = TOP_MARGIN + n * ROW_H + BOTTOM_MARGIN;
  svg.setAttribute("viewBox", `0 0 ${{CHART_WIDTH}} ${{height}}`);
  svg.setAttribute("height", height);

  const scale = v => LEFT_MARGIN + ((v - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotWidth;
  const baselineX = scale(BASELINE_VALUE);
  const plotBottom = TOP_MARGIN + n * ROW_H;

  for (const t of [0, 0.25, 0.5, 0.75, 1.0]) {{
    const x = scale(t);
    svg.appendChild(svgEl("line", {{
      class: "gridline", x1: x, x2: x, y1: TOP_MARGIN, y2: plotBottom,
    }}));
    const label = svgEl("text", {{
      class: "tick-label", x: x, y: plotBottom + 16, "text-anchor": "middle",
    }});
    label.textContent = t.toFixed(2);
    svg.appendChild(label);
  }}

  rows.forEach((r, i) => {{
    const y = TOP_MARGIN + i * ROW_H;
    const barY = y + (ROW_H - BAR_H) / 2;
    const endX = scale(r.value);

    const hit = svgEl("rect", {{
      class: "hit-row", x: 0, y: y, width: CHART_WIDTH, height: ROW_H,
    }});

    const bar = svgEl("path", {{ class: barClass(r), d: roundedBarPath(baselineX, endX, barY, BAR_H, CORNER_R) }});

    const label = svgEl("text", {{
      class: "row-label", x: LEFT_MARGIN - 10, y: y + ROW_H / 2 + 4, "text-anchor": "end",
      "font-weight": r.bold ? "700" : "400",
    }});
    label.textContent = r.name;

    hit.addEventListener("pointerenter", () => bar.classList.add("hovered"));
    hit.addEventListener("pointerleave", () => {{ bar.classList.remove("hovered"); hideTooltip(); }});
    hit.addEventListener("pointermove", ev => showTooltip(ev, r));

    // hit-row goes on top (last) so it wins hit-testing over the bar/label
    // beneath it -- pointerenter/leave don't bubble between siblings, so a
    // listener on a lower rect would never fire while the pointer is over
    // the (higher, unlistened) bar.
    svg.appendChild(bar);
    svg.appendChild(label);
    svg.appendChild(hit);
  }});

  svg.appendChild(svgEl("line", {{
    class: "baseline", x1: baselineX, x2: baselineX, y1: TOP_MARGIN, y2: plotBottom,
  }}));
}}

function showTooltip(ev, r) {{
  const tt = document.getElementById("tooltip");
  tt.innerHTML = "";
  const name = document.createElement("div");
  name.className = "tt-name";
  name.textContent = r.name;
  const value = document.createElement("div");
  value.className = "tt-value";
  value.textContent = `P(beats ${{DATA.baseline}}) = ${{r.value.toFixed(3)}}, p = ${{r.p_value.toFixed(3)}}`;
  tt.appendChild(name);
  tt.appendChild(value);
  tt.style.display = "block";
  tt.style.left = (ev.clientX + 14) + "px";
  tt.style.top = (ev.clientY + 14) + "px";
}}

function hideTooltip() {{
  document.getElementById("tooltip").style.display = "none";
}}

function renderTable(rows) {{
  const wrap = document.getElementById("table-wrap");
  wrap.innerHTML = "";
  const table = document.createElement("table");
  const thead = document.createElement("tr");
  for (const h of ["", "Embedding", `P(beats ${{DATA.baseline}})`, "p-value"]) {{
    const th = document.createElement("th");
    th.textContent = h;
    thead.appendChild(th);
  }}
  table.appendChild(thead);
  for (const r of rows) {{
    const tr = document.createElement("tr");
    const swatchCell = document.createElement("td");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(${{barClass(r) === "bar-strong" ? "--strong" : barClass(r) === "bar-weak" ? "--weak" : "--neutral"}})`;
    swatchCell.appendChild(swatch);
    const nameCell = document.createElement("td");
    nameCell.textContent = r.name;
    nameCell.style.fontWeight = r.bold ? "700" : "400";
    const valueCell = document.createElement("td");
    valueCell.textContent = r.value.toFixed(4);
    const pCell = document.createElement("td");
    pCell.textContent = r.p_value.toFixed(4);
    tr.appendChild(swatchCell);
    tr.appendChild(nameCell);
    tr.appendChild(valueCell);
    tr.appendChild(pCell);
    table.appendChild(tr);
  }}
  wrap.appendChild(table);
}}

document.getElementById("view-toggle").addEventListener("click", () => {{
  showTable = !showTable;
  document.getElementById("view-toggle").textContent = showTable ? "View as chart" : "View as table";
  document.getElementById("chart-wrap").style.display = showTable ? "none" : "block";
  document.getElementById("table-wrap").style.display = showTable ? "block" : "none";
  renderCurrent();
}});

renderSidebar();
updateHeader();
renderCurrent();
</script>
</body>
</html>
"""


def write_endpoint_comparison_html(
    comparison_df: pd.DataFrame, endpoint_df: pd.DataFrame, baseline: str, alpha: float, output: str,
    display_names: dict = None, new_endpoint_df: pd.DataFrame = None,
) -> None:
    payload = build_endpoint_comparison_payload(comparison_df, endpoint_df, baseline, alpha, display_names, new_endpoint_df)
    with open(output, "w") as f:
        f.write(render_endpoint_comparison_html(payload))
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/blam/theta_summary.csv")
    parser.add_argument("--output", default="data/blam/theta_ranking.pdf")
    parser.add_argument("--html", action="store_true", help="Also write an interactive dashboard (theta ranking + per-endpoint specialization)")
    parser.add_argument("--specialization-input", default="data/blam/specialization_summary.csv")
    parser.add_argument("--endpoint-input", default="data/blam/endpoint_summary.csv")
    parser.add_argument("--html-output", default="data/blam/blam_dashboard.html")
    parser.add_argument(
        "--theta-gif", action="store_true",
        help="Write an animated GIF of just the theta ranking, stepping down the ranked list one "
        "embedding at a time (top/best to bottom/worst) with that row highlighted -- embeds natively "
        "into Google Slides (Insert > Image) and autoplays in Present mode, unlike an HTML page",
    )
    parser.add_argument("--theta-gif-output", default="data/blam/theta_ranking.gif")
    parser.add_argument("--theta-gif-frame-ms", type=int, default=900, help="Milliseconds each --theta-gif frame is shown")
    parser.add_argument(
        "--compare-ecfp", action="store_true",
        help="Also write a bar chart of P(embedding > baseline) and a significance p-value, per the paper's App. C.5",
    )
    parser.add_argument("--posterior-input", default="data/blam/posterior.joblib")
    parser.add_argument(
        "--ecfp-baseline", default="ECFP",
        help="Embedding name to use as the baseline for --compare-ecfp (default: ECFP, the uncounted fingerprint)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for --compare-ecfp's p-value")
    parser.add_argument("--compare-output", default="data/blam/ecfp_comparison.png")
    parser.add_argument("--compare-seed", type=int, default=0, help="Seed for --compare-ecfp's new-endpoint Monte Carlo draws")
    parser.add_argument(
        "--compare-ecfp-gif", action="store_true",
        help="Write an animated GIF of P(embedding > baseline) for the hypothetical new/unseen endpoint "
        "(compute_ecfp_comparison), stepping through embeddings one at a time -- embeds natively into "
        "Google Slides (Insert > Image), unlike --compare-ecfp-endpoints' HTML explorer",
    )
    parser.add_argument("--compare-ecfp-gif-output", default="data/blam/ecfp_comparison.gif")
    parser.add_argument("--compare-ecfp-gif-frame-ms", type=int, default=900, help="Milliseconds each --compare-ecfp-gif frame is shown")
    parser.add_argument(
        "--compare-ecfp-endpoints", action="store_true",
        help="Write a standalone HTML explorer of P(embedding > baseline) per endpoint, with a significance p-value",
    )
    parser.add_argument("--endpoint-compare-output", default="data/blam/ecfp_endpoint_comparison.html")
    parser.add_argument(
        "--display-names", default=None,
        help="Path to a JSON {raw_name: display_name} map (see data/blam_v2/model_display_names.json) used to "
        "relabel embeddings everywhere they're plotted; fine-tuned entities ('<name>_finetuned') look up their "
        "base name and get a ' (fine-tuned)' suffix plus bold styling. Names missing from the map fall back to "
        "the raw name.",
    )
    args = parser.parse_args()

    display_names = load_display_names(args.display_names)

    df = pd.read_csv(args.input)
    plot_theta(df, args.output, display_names)

    need_posterior = args.compare_ecfp or args.compare_ecfp_gif or args.compare_ecfp_endpoints or (
        args.html and os.path.exists(args.posterior_input)
    )
    posterior = load_posterior(args.posterior_input) if need_posterior else None

    if args.html:
        spec_df = pd.read_csv(args.specialization_input)
        endpoint_df = pd.read_csv(args.endpoint_input)
        mae_df = compute_mae_summary(posterior) if posterior is not None else None
        write_dashboard_html(df, spec_df, endpoint_df, args.html_output, mae_df, display_names)

    if args.theta_gif:
        plot_theta_gif(df, args.theta_gif_output, display_names, args.theta_gif_frame_ms)

    if args.compare_ecfp:
        comparison_df = compute_ecfp_comparison(posterior, args.ecfp_baseline, args.compare_seed)
        plot_ecfp_comparison(comparison_df, args.compare_output, args.ecfp_baseline, args.alpha, display_names)

    if args.compare_ecfp_gif:
        comparison_df = compute_ecfp_comparison(posterior, args.ecfp_baseline, args.compare_seed)
        plot_ecfp_comparison_gif(
            comparison_df, args.compare_ecfp_gif_output, args.ecfp_baseline, args.alpha, display_names,
            args.compare_ecfp_gif_frame_ms,
        )

    if args.compare_ecfp_endpoints:
        endpoint_df = pd.read_csv(args.endpoint_input)
        endpoint_comparison_df = compute_endpoint_ecfp_comparison(posterior, args.ecfp_baseline)
        new_endpoint_df = compute_ecfp_comparison(posterior, args.ecfp_baseline, args.compare_seed)
        write_endpoint_comparison_html(
            endpoint_comparison_df, endpoint_df, args.ecfp_baseline, args.alpha, args.endpoint_compare_output, display_names,
            new_endpoint_df,
        )


if __name__ == "__main__":
    main()
