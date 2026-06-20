import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use('Agg')

IEEE_STYLE = {
    'figure.figsize': (3.5, 2.5),
    'figure.dpi': 300,
    'font.size': 8,
    'font.family': 'serif',
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
MARKERS = ['o', 's', '^', 'D', 'v']

# Distinct palette for plots that may show many models (>5) on one axis.
# tab10 gives 10 perceptually distinct colours; markers cycle independently so
# even an 11th+ series stays distinguishable.
DISTINCT_COLORS = list(plt.get_cmap('tab10').colors)
DISTINCT_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']


def _series_style(i: int):
    """Return a (color, marker) pair that stays distinct for many series."""
    return (
        DISTINCT_COLORS[i % len(DISTINCT_COLORS)],
        DISTINCT_MARKERS[i % len(DISTINCT_MARKERS)],
    )


def setup_ieee_style():
    plt.rcParams.update(IEEE_STYLE)


def save_figure(fig, filename: str, output_dir: str = "results/figures"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in ['pdf', 'png']:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    
    plt.close(fig)
    return output_path / f"{filename}.png"


def _draw_pdr_axis(ax, pdr_data: dict):
    """Render grouped PDR-by-level bars on a given axis (auto-scaled, signed).

    pdr_data maps model -> {level(int): pdr_percent}. Values are already in %
    and may be negative (model improved under perturbation).
    """
    models = list(pdr_data.keys())
    levels = [1, 2, 3]
    x = np.arange(len(models))
    width = 0.25

    all_vals = []
    for i, lvl in enumerate(levels):
        values = [pdr_data[m].get(lvl, 0) for m in models]
        all_vals.extend(values)
        ax.bar(x + i * width, values, width, label=f'Level {lvl}',
               color=COLORS[i], edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('PDR (%)')
    ax.set_title('Performance Drop Rate by level  (secondary: shape only)', fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', frameon=True, ncol=3, fontsize=6)
    lo, hi = (min(all_vals), max(all_vals)) if all_vals else (-1, 1)
    pad = max(1.0, (hi - lo) * 0.25)
    ax.set_ylim(lo - pad, hi + pad)


def plot_pdr_bars(pdr_data: dict, output_dir: str = "results/figures") -> Path:
    """Standalone (corrected) PDR-by-level chart. Retained for compatibility;
    the headline figure is plot_robustness_comparison()."""
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    _draw_pdr_axis(ax, pdr_data)
    ax.set_xlabel('Model')
    plt.tight_layout()
    return save_figure(fig, "pdr_comparison", output_dir)


def plot_robustness_comparison(
    wlr_data: dict,
    pdr_data: dict,
    output_dir: str = "results/figures",
    mean_data: dict | None = None,
) -> Path:
    """Headline robustness figure.

    Top (primary):  Worst-Level Robustness (WLR = min composite over L0-L3) per
                    model as bars (the worst-case number), with the mean-level
                    score (average-case companion) overlaid as a marker. Reporting
                    worst-case beside average-case is the convention in WILDS
                    (worst-group vs. average accuracy) and AdvGLUE.
    Bottom (2nd):   PDR-by-level (shape diagnostic only), auto-scaled & signed.

    wlr_data maps model -> WLR in [0, 1]; pdr_data maps model -> {level: pdr%};
    mean_data (optional) maps model -> mean-level composite in [0, 1].
    """
    setup_ieee_style()
    ranked = sorted(wlr_data.items(), key=lambda kv: kv[1], reverse=True)
    models = [m for m, _ in ranked]
    wlr = [v for _, v in ranked]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 5.2), gridspec_kw={'height_ratios': [1.1, 1]}
    )

    # --- primary: WLR bars (worst-case) + mean-level markers (average-case) ---
    x = np.arange(len(models))
    bars = ax_top.bar(x, wlr, color=DISTINCT_COLORS[0], edgecolor='black',
                      linewidth=0.5, label='WLR (worst case)', zorder=2)
    for bar, v in zip(bars, wlr):
        ax_top.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6)

    y_top = max(wlr) if wlr else 1
    if mean_data:
        means = [mean_data.get(m, 0.0) for m in models]
        y_top = max(y_top, max(means) if means else y_top)
        ax_top.scatter(x, means, marker='D', s=28, color=DISTINCT_COLORS[3],
                       edgecolor='black', linewidth=0.5, zorder=3,
                       label='Mean level (avg case)')
        for xi, mv in zip(x, means):
            ax_top.text(xi, mv + 0.006, f'{mv:.3f}', ha='center', va='bottom',
                        fontsize=5.5, color=DISTINCT_COLORS[3])
        ax_top.legend(loc='lower left', frameon=True, fontsize=6, ncol=2)

    ax_top.set_ylabel('Composite score')
    ax_top.set_title('Robustness ranking (primary): worst-case (bar) and average-case (marker)',
                     fontsize=8.5)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(models, rotation=15, ha='right')
    ax_top.set_ylim(0, y_top * 1.18)

    # --- secondary: PDR shape, reordered to match the WLR ranking ---
    pdr_ordered = {m: pdr_data.get(m, {}) for m in models}
    _draw_pdr_axis(ax_bot, pdr_ordered)
    ax_bot.set_xlabel('Model (ordered by WLR)')

    plt.tight_layout()
    return save_figure(fig, "robustness_comparison", output_dir)


def plot_constraint_breakdown(
    cpr_by_type: dict,
    output_dir: str = "results/figures",
) -> Path:
    """Per-constraint-type compliance, broken out from the averaged CPR.

    Averaging keyword/length/format into a single CPR hides that keyword and
    format are saturated (~1.0) while length is the only discriminating
    constraint. This grouped bar reports the three separately so the headline
    CPR is not read as a single capability axis.

    cpr_by_type maps model -> {"keyword": rate, "length": rate, "format": rate},
    each rate a pass fraction in [0, 1] averaged over all levels.
    """
    setup_ieee_style()
    models = list(cpr_by_type.keys())
    types = ["keyword", "length", "format"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, t in enumerate(types):
        vals = [cpr_by_type[m].get(t, 0.0) * 100 for m in models]
        ax.bar(x + i * width, vals, width, label=t.capitalize(),
               color=COLORS[i], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Compliance (%)')
    ax.set_xlabel('Model')
    ax.set_title('Constraint compliance by type (averaged over L0-L3)', fontsize=8.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True, ncol=3, fontsize=6)

    plt.tight_layout()
    return save_figure(fig, "constraint_breakdown", output_dir)


def plot_heatmap(matrix_data: dict, output_dir: str = "results/figures") -> Path:
    setup_ieee_style()
    
    models = list(matrix_data.keys())
    levels = ['Clean', 'Mild', 'Jaksel', 'Adversarial']
    
    data = []
    for model in models:
        row = [matrix_data[model].get(i, 0) * 100 for i in range(4)]
        data.append(row)
    
    data = np.array(data)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(levels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(levels)
    ax.set_yticklabels(models)
    
    for i in range(len(models)):
        for j in range(len(levels)):
            text_color = 'white' if data[i, j] < 50 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color=text_color, fontsize=7)
    
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Model')
    ax.set_title('Constraint Pass Rate (%)')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pass Rate (%)', size=7)
    
    plt.tight_layout()
    return save_figure(fig, "pass_rate_heatmap", output_dir)


def plot_perturbation_comparison(results: dict, output_dir: str = "results/figures") -> Path:
    setup_ieee_style()
    
    models = list(results.keys())
    levels = [0, 1, 2, 3]
    level_labels = ['Clean', 'Mild', 'Jaksel', 'Adversarial']
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    for i, model in enumerate(models):
        scores = [results[model].get(lvl, 0) * 100 for lvl in levels]
        color, marker = _series_style(i)
        ax.plot(levels, scores, marker=marker, label=model, color=color, linewidth=1.5, markersize=5)
    
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Across Perturbation Levels')
    ax.set_xticks(levels)
    ax.set_xticklabels(level_labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', frameon=True)
    
    plt.tight_layout()
    return save_figure(fig, "perturbation_trend", output_dir)


def plot_perturbation_examples(
    dataset: list[dict],
    num_examples: int = 5,
    output_dir: str = "results/figures",
    seed: int = 42,
) -> Path:
    """Create a table showing perturbation examples side-by-side.
    
    Args:
        dataset: List of dictionaries containing prompt data with perturbations
        num_examples: Number of random examples to display (default: 5)
        output_dir: Output directory for saving figures (default: "results/figures")
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Path to saved PNG file
    """
    import random
    import re
    import textwrap

    random.seed(seed)
    setup_ieee_style()

    samples = random.sample(dataset, min(num_examples, len(dataset)))

    levels = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]
    level_labels = ["Clean (L0)", "Mild (L1)", "Jaksel (L2)", "Adversarial (L3)"]

    # Strip glyphs outside Latin + common punctuation/currency (e.g. leaked CJK
    # source text) that the serif font cannot render and would draw as tofu
    # boxes. Replace each run with a compact marker so the cell stays readable.
    _non_latin = re.compile(r'[^\x00-ɏ‐-‧‰- ₠-⃏]+')
    WRAP, MAXCH = 50, 260

    def _prep(text) -> str:
        text = _non_latin.sub(' […] ', str(text) if text else "N/A").strip()
        text = re.sub(r'\s+', ' ', text)
        if len(text) > MAXCH:
            text = text[:MAXCH - 1].rstrip() + "…"
        return textwrap.fill(text, width=WRAP) or "N/A"

    table_data = [
        [
            _prep((s.get("perturbations") or {}).get(lk) if isinstance(s.get("perturbations"), dict) else None)
            for lk in levels
        ]
        for s in samples
    ]

    # Size the canvas to the wrapped content so there is no dead whitespace.
    row_lines = [max(cell.count('\n') + 1 for cell in row) for row in table_data]
    header_u = 1.6
    pad = 0.9  # vertical breathing room per data row, in line-units
    total_u = header_u + sum(rl + pad for rl in row_lines)
    fig_h = max(2.5, total_u * 0.30)

    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=level_labels,
        loc='center',
        cellLoc='left',
        colLoc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Equal column widths; per-row heights proportional to wrapped line count.
    unit = 1.0 / total_u
    for j in range(len(level_labels)):
        cell = table[(0, j)]
        cell.set_height(header_u * unit)
        cell.set_width(0.25)
        cell.set_facecolor('#2E86AB')
        cell.set_text_props(weight='bold', color='white', ha='center')
        cell.set_edgecolor('#1b5f7e')

    for r, rl in enumerate(row_lines, start=1):
        for j in range(len(level_labels)):
            cell = table[(r, j)]
            cell.set_height((rl + pad) * unit)
            cell.set_width(0.25)
            cell.set_facecolor('#f4f7f9' if r % 2 == 0 else 'white')
            cell.set_edgecolor('#cfd8dc')
            cell.get_text().set_verticalalignment('center')
            cell.PAD = 0.04

    ax.set_title(f"Perturbation Level Examples ({len(samples)} Random Prompts)",
                 fontsize=11, pad=12, weight='bold')

    plt.tight_layout()
    return save_figure(fig, "perturbation_examples", output_dir)
