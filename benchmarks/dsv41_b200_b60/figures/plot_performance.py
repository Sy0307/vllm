"""Render the B200 performance figure from existing benchmark summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    data = json.loads((here / "performance_data.json").read_text())
    font = args.font
    if font is None:
        candidates = [
            Path("/System/Library/Fonts/STHeiti Medium.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
        font = next((p for p in candidates if p.exists()), None)
    if font:
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = "DejaVu Sans"

    plt.rcParams.update({
        "font.family": family,
        "font.size": 14,
        "axes.unicode_minus": False,
        "axes.edgecolor": "#c8d0da",
        "axes.labelcolor": "#334155",
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "svg.fonttype": "path",
    })
    ink, muted, grid = "#0f172a", "#64748b", "#e5eaf0"
    colors = {2048: "#7563d9", 2432: "#3291cf", 2816: "#049b82"}
    runs = {r["concurrency"]: r for r in data["runs"]}
    best = runs[2816]
    fig = plt.figure(figsize=(14, 9.4), facecolor="white")
    fig.text(.055, .945, "DeepSeek V4.1 Flash  /  B200 性能概览",
             fontsize=26, fontweight="bold", color=ink)
    fig.text(.055, .902, "AgentX  ·  16 GPU（8 Prefill + 8 Decode）  ·  2026-11-03 整理",
             fontsize=14, color=muted)
    fig.text(.055, .852, "当前最高吞吐配置：并发 2816", fontsize=14,
             color=colors[2816], fontweight="bold")

    cards = [
        ("100.2k", "全程输出 tok/s · 16 GPU"),
        ("6,264", "全程输出 tok/s/GPU"),
        ("112.45", "全程平均 per-user TPS"),
        ("1.55 s", "稳态平均首 token 等待"),
    ]
    for i, (value, label) in enumerate(cards):
        x = .055 + i * .235
        patch = FancyBboxPatch((x, .737), .218, .09,
                              boxstyle="round,pad=0.008,rounding_size=0.007",
                              transform=fig.transFigure, facecolor="#f7f9fc",
                              edgecolor="#e7ecf2", linewidth=.8)
        fig.add_artist(patch)
        fig.text(x + .015, .783, value, fontsize=27, fontweight="bold", color=ink)
        fig.text(x + .015, .751, label, fontsize=12.5, color=muted)

    left = fig.add_axes([.088, .257, .47, .374])
    right = fig.add_axes([.678, .257, .27, .374])
    fig.text(.065, .681, "吞吐与用户生成速度", fontsize=19,
             fontweight="bold", color=ink)
    fig.text(.654, .681, "首 token 等待分位数", fontsize=19,
             fontweight="bold", color=ink)

    for ax in [left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=13, length=0, pad=8)
        ax.set_axisbelow(True)
    left.grid(color=grid, linewidth=.8)
    left.set_xlim(103, 158)
    left.set_ylim(4650, 6680)
    left.set_xticks([110, 120, 130, 140, 150])
    left.set_yticks([5000, 5500, 6000, 6500])
    left.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    left.set_xlabel("Per-user TPS（全程均值，tok/s/user）", labelpad=13, fontsize=14)
    left.set_ylabel("输出吞吐（tok/s/GPU）", labelpad=12, fontsize=14)
    left.text(.98, .98, "越往右上越好", ha="right", va="top",
              transform=left.transAxes, color=muted, fontsize=12)

    positions = {2048: (123, 4930), 2432: (132, 5800), 2816: (115.6, 6360)}
    for concurrency in [2048, 2432, 2816]:
        r = runs[concurrency]
        x, y = r["per_user_tps"]["avg"], r["output_tps"] / 16
        left.scatter(x, y, s=180 if concurrency == 2816 else 115,
                     marker="D" if concurrency == 2816 else "o",
                     color=colors[concurrency], edgecolor="white", linewidth=1.8, zorder=5)
        label = f"并发 {concurrency}" + (" · 当前最高吞吐" if concurrency == 2816 else "")
        if concurrency == 2432:
            label += " *"
        label += f"\n{r['output_tps']/1000:.1f}k tok/s · {x:.1f} TPS/user"
        left.annotate(label, xy=(x, y), xytext=positions[concurrency],
                      fontsize=13, color=colors[concurrency], fontweight="bold",
                      ha="left", va="center", linespacing=1.7,
                      arrowprops={"arrowstyle": "-", "color": colors[concurrency],
                                  "lw": 1.0, "connectionstyle": "arc3,rad=0"})

    percentiles = ["p50", "p90", "p95", "p99"]
    ys = [3, 2, 1, 0]
    for y, stat in zip(ys, percentiles):
        low = runs[2048]["ttft_ms"][stat] / 1000
        high = runs[2816]["ttft_ms"][stat] / 1000
        right.plot([low, high], [y, y], color=grid, lw=4.5, zorder=1)
        for concurrency, value, offset in [(2048, low, -17), (2816, high, 12)]:
            right.scatter(value, y, s=80, color=colors[concurrency], zorder=3)
            right.annotate(f"{value:.2f}", (value, y), xytext=(0, offset),
                           textcoords="offset points", ha="center", va="center",
                           fontsize=12.5, color=colors[concurrency], fontweight="bold")
    right.set_yticks(ys, [x.upper() for x in percentiles])
    right.set_ylim(-.48, 3.48)
    right.set_xlim(-.6, 26.7)
    right.set_xticks([0, 5, 10, 15, 20, 25])
    right.grid(axis="x", color=grid, linewidth=.8)
    right.set_xlabel("首 token 等待（秒，越小越好）", labelpad=13, fontsize=13)
    handles = [Line2D([], [], linestyle="", marker="o", markersize=7,
                      color=colors[c], label=f"并发 {c}") for c in [2048, 2816]]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.654, .636),
               frameon=False, fontsize=11.8, handletextpad=.3, ncols=2,
               columnspacing=1.2)

    fig.text(.065, .164, "读图：并发 2816 的整机吞吐更高；并发 2048 的用户生成速度更快、首 token 等待更短。",
             fontsize=14, color=ink)
    fig.text(.065, .119, "口径：图中散点与分位数均取正式全程；左图用户速度不含首 token 等待。稳态卡片取第 600–1800 秒。",
             fontsize=11.8, color=muted)
    fig.text(.065, .085, "* 并发 2432 该轮有 4 个正式错误。各点为不同实测配置，不是只改变并发的扫描；投机接受率采用固定模拟值。",
             fontsize=11.8, color=muted)
    fig.text(.065, .044, "图形视角参考 InferenceX  ·  数据：本项目 AIPerf 原始导出  ·  当前最高吞吐运行：pb-b60-1102a",
             fontsize=10.8, color=muted)

    fig.savefig(here / "performance_overview.png", dpi=200, facecolor="white")
    fig.savefig(here / "performance_overview.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
