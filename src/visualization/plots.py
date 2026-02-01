import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
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


def plot_pdr_bars(pdr_data: dict, output_dir: str = "results/figures") -> Path:
    setup_ieee_style()
    
    models = list(pdr_data.keys())
    levels = ['Level 1', 'Level 2', 'Level 3']
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(7, 2.8))
    
    for i, level in enumerate(levels):
        level_key = f"level_{i+1}"
        values = [pdr_data[m].get(level_key, 0) * 100 for m in models]
        bars = ax.bar(x + i * width, values, width, label=level, color=COLORS[i], edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance Drop Rate (%)')
    ax.set_title('PDR by Perturbation Level')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', frameon=True)
    ax.set_ylim(0, max([max(pdr_data[m].values()) * 100 for m in models]) * 1.3 + 5)
    
    plt.tight_layout()
    return save_figure(fig, "pdr_comparison", output_dir)


def plot_radar_chart(skill_data: dict, output_dir: str = "results/figures") -> Path:
    setup_ieee_style()
    
    categories = list(next(iter(skill_data.values())).keys())
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    for i, (model, scores) in enumerate(skill_data.items()):
        values = [scores[cat] * 100 for cat in categories]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=1.5, label=model, color=COLORS[i % len(COLORS)], markersize=4)
        ax.fill(angles, values, alpha=0.15, color=COLORS[i % len(COLORS)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=7)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True)
    
    plt.tight_layout()
    return save_figure(fig, "skill_radar", output_dir)


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
        ax.plot(levels, scores, marker=MARKERS[i % len(MARKERS)], label=model, color=COLORS[i % len(COLORS)], linewidth=1.5, markersize=5)
    
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
    
    random.seed(seed)
    setup_ieee_style()
    
    samples = random.sample(dataset, min(num_examples, len(dataset)))
    
    levels = ["level_0_clean", "level_1_mild", "level_2_jaksel", "level_3_adversarial"]
    level_labels = ["Clean", "Mild", "Jaksel", "Adversarial"]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = []
    for sample in samples:
        row = []
        for level_key in levels:
            perturbations = sample.get("perturbations", {})
            if isinstance(perturbations, dict):
                text = perturbations.get(level_key, "N/A")
            else:
                text = "N/A"
            
            text = str(text) if text else "N/A"
            
            if len(text) > 60:
                text = text[:57] + "..."
            
            row.append(text)
        table_data.append(row)
    
    table = ax.table(
        cellText=table_data,
        colLabels=level_labels,
        loc='center',
        cellLoc='left',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 2.5)
    
    for i in range(len(level_labels)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(level_labels)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax.set_title("Perturbation Level Examples (5 Random Prompts)", fontsize=10, pad=20, weight='bold')
    
    plt.tight_layout()
    return save_figure(fig, "perturbation_examples", output_dir)
