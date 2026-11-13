"""Generate lightweight SVG figures from the frozen validation JSON files."""
from __future__ import annotations

import html
import json
import statistics
from pathlib import Path


VARIANTS = ["B", "E0", "E1", "E2", "E3"]
COLORS = {"B": "#4C78A8", "E0": "#F58518", "E1": "#54A24B", "E2": "#E45756", "E3": "#72B7B2"}
TASKS = [
    ("classification", "accuracy", "Classification Acc"),
    ("extraction", "f1", "Extraction F1"),
    ("retrieval", "mrr", "Retrieval MRR"),
    ("reranking", "mrr", "Reranking MRR"),
]


def _load(folder: Path) -> tuple[list[int], dict[tuple[int, str], dict]]:
    env = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    seeds = [int(s) for s in env["seeds"]]
    records = {
        (seed, variant): json.loads(
            (folder / f"seed_{seed}" / f"{variant}_metrics.json").read_text(encoding="utf-8")
        )
        for seed in seeds
        for variant in VARIANTS
    }
    return seeds, records


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{_esc(title)}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124} .grid{stroke:#d9d9d9;stroke-width:1} .axis{stroke:#555;stroke-width:1.2} .caption{font-size:13px} .small{font-size:11px} .title{font-size:18px;font-weight:600}</style>',
    ]


def _text(x: float, y: float, value: object, cls: str = "caption", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{_esc(value)}</text>'


def _task_figure(seeds: list[int], records: dict[tuple[int, str], dict], output: Path) -> None:
    width, height = 1040, 620
    left, top, chart_w, chart_h = 75, 80, 900, 430
    parts = _svg_start(width, height, "Downstream task means across ablations")
    parts.append(_text(width / 2, 35, "Downstream task means (percent; mean +/- seed std)", "title", "middle"))
    parts.append(_text(width / 2, height - 18, "Ablation group", "small", "middle"))
    parts.append(_text(18, top + chart_h / 2, "Score (%)", "small", "middle"))
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(_text(left - 10, y + 4, tick, "small", "end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis"/>')
    bar_width, gap = 30, 5
    group_width = chart_w / len(VARIANTS)
    for i, variant in enumerate(VARIANTS):
        group_x = left + i * group_width
        parts.append(_text(group_x + group_width / 2, top + chart_h + 22, variant, "caption", "middle"))
        for j, (task, metric, label) in enumerate(TASKS):
            values = []
            for seed in seeds:
                row = records[(seed, variant)][task]
                values.append((row["en"][metric] + row["zh"][metric]) / 2 * 100)
            mean, std = _mean_std(values)
            x = group_x + (group_width - (len(TASKS) * bar_width + (len(TASKS) - 1) * gap)) / 2 + j * (bar_width + gap)
            y = top + chart_h - chart_h * mean / 100
            h = chart_h * mean / 100
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{COLORS[variant]}" opacity="{0.55 + 0.1 * j:.2f}"/>')
            err = chart_h * std / 100
            cx = x + bar_width / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{max(top, y - err):.1f}" x2="{cx:.1f}" y2="{min(top + chart_h, y + err):.1f}" stroke="#333" stroke-width="1"/>')
            parts.append(f'<line x1="{cx - 4:.1f}" y1="{max(top, y - err):.1f}" x2="{cx + 4:.1f}" y2="{max(top, y - err):.1f}" stroke="#333"/>')
            parts.append(f'<line x1="{cx - 4:.1f}" y1="{min(top + chart_h, y + err):.1f}" x2="{cx + 4:.1f}" y2="{min(top + chart_h, y + err):.1f}" stroke="#333"/>')
    legend_y = height - 48
    for j, (_, _, label) in enumerate(TASKS):
        x = 150 + j * 205
        parts.append(f'<rect x="{x}" y="{legend_y - 11}" width="13" height="13" fill="#777" opacity="{0.55 + 0.1 * j:.2f}"/>')
        parts.append(_text(x + 19, legend_y, label, "small"))
    parts.append(_text(width - 12, 60, "Error bars: sample std across seeds", "small", "end"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _line_panel(parts: list[str], x: float, y: float, width: float, height: float, title: str, series: dict[str, list[float]], xlabels: list[str], ymax: float = 100.0, reference: float | None = None) -> None:
    parts.append(_text(x + width / 2, y - 18, title, "caption", "middle"))
    ticks = [ymax * i / 4 for i in range(5)]
    for tick in ticks:
        yy = y + height - height * tick / ymax
        parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" class="grid"/>')
        if x in (65, 70, 575):
            label = f"{tick:.2f}" if ymax < 10 else f"{tick:.0f}"
            parts.append(_text(x - 9, yy + 4, label, "small", "end"))
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>')
    parts.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/>')
    positions = [x + width * i / (len(xlabels) - 1) for i in range(len(xlabels))]
    for px, label in zip(positions, xlabels):
        parts.append(_text(px, y + height + 19, label, "small", "middle"))
    if reference is not None:
        yy = y + height - height * reference / ymax
        parts.append(f'<line x1="{x}" y1="{yy:.1f}" x2="{x + width}" y2="{yy:.1f}" stroke="#888" stroke-dasharray="5,4"/>')
    for variant, values in series.items():
        points = []
        for px, value in zip(positions, values):
            py = y + height - height * value / ymax
            points.append(f"{px:.1f},{py:.1f}")
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{COLORS[variant]}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[variant]}" stroke-width="2"/>')


def _long_figure(seeds: list[int], records: dict[tuple[int, str], dict], output: Path) -> None:
    width, height = 1040, 470
    parts = _svg_start(width, height, "Long evidence counterfactual pair accuracy")
    parts.append(_text(width / 2, 32, "Long evidence: counterfactual pair accuracy", "title", "middle"))
    lengths = ["256", "1024", "8192"]
    for idx, location in enumerate(["near", "far", "removed"]):
        x = 65 + idx * 325
        series = {
            variant: [statistics.mean([records[(seed, variant)]["long_evidence"][f"{length}_{location}"]["counterfactual_pair_accuracy"] for seed in seeds]) * 100 for length in lengths]
            for variant in VARIANTS
        }
        _line_panel(parts, x, 78, 270, 285, location, series, lengths, 100, 25)
    parts.append(_text(20, 220, "Pair accuracy (%)", "small", "middle"))
    parts.append(_text(75, 410, "Dashed line: 25% random reference for two paired guesses", "small"))
    for idx, variant in enumerate(VARIANTS):
        x = 470 + (idx % 3) * 170
        y = 410 + (idx // 3) * 22
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 20}" y2="{y - 4}" stroke="{COLORS[variant]}" stroke-width="3"/>')
        parts.append(_text(x + 27, y, variant, "small"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _deployment_figure(seeds: list[int], records: dict[tuple[int, str], dict], output: Path) -> None:
    width, height = 1040, 470
    parts = _svg_start(width, height, "Deployment latency and throughput")
    parts.append(_text(width / 2, 32, "Deployment cost by input length (mean across seeds)", "title", "middle"))
    lengths = ["256", "1024", "8192"]
    variants = ["E0", "E2", "E3"]
    latency = {v: [statistics.mean([records[(seed, v)]["deployment"][n]["median_ms"] for seed in seeds]) for n in lengths] for v in variants}
    throughput = {v: [statistics.mean([records[(seed, v)]["deployment"][n]["tokens_per_second"] for seed in seeds]) for n in lengths] for v in variants}
    max_latency = max(max(values) for values in latency.values()) * 1.12
    max_throughput = max(max(values) for values in throughput.values()) * 1.12
    _line_panel(parts, 70, 78, 385, 285, "Median latency (ms)", latency, lengths, max_latency)
    _line_panel(parts, 575, 78, 385, 285, "Throughput (tokens/s)", throughput, lengths, max_throughput)
    for idx, variant in enumerate(variants):
        x = 395 + idx * 100
        y = 410
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 20}" y2="{y - 4}" stroke="{COLORS[variant]}" stroke-width="3"/>')
        parts.append(_text(x + 27, y, variant, "small"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def build_all(folder: Path, output: Path) -> list[str]:
    """Build all report figures and return report-relative paths."""
    output.mkdir(parents=True, exist_ok=True)
    seeds, records = _load(folder)
    _task_figure(seeds, records, output / "task_means.svg")
    _long_figure(seeds, records, output / "long_evidence.svg")
    _deployment_figure(seeds, records, output / "deployment_cost.svg")
    return [
        "results/figures/task_means.svg",
        "results/figures/long_evidence.svg",
        "results/figures/deployment_cost.svg",
    ]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print("Generated:", ", ".join(build_all(root / "results/validation", root / "results/figures")))
