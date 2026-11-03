"""Render the existing B200 results using InferenceX's documented dark theme."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def oklch_hex(lightness, chroma, hue):
    """Convert the upstream OKLCH theme tokens to clipped sRGB."""
    angle = math.radians(hue)
    a, b = chroma * math.cos(angle), chroma * math.sin(angle)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    ]
    rgb = [
        12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        for c in rgb
    ]
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02x}" for c in rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="Chinese fallback font")
    parser.add_argument("--latin-font", type=Path, help="Optional DM Sans font")
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
    families = []
    for path in [args.latin_font, font]:
        if path:
            font_manager.fontManager.addfont(str(path))
            families.append(font_manager.FontProperties(fname=str(path)).get_name())
    families.append("DejaVu Sans")

    # InferenceX globals.css dark theme; see style_reference.json for pinned sources.
    bg, ink, muted = "#131416", "#eaebec", "#b4b9bc"
    border, grid, gold = "#656b72", "#222426", "#f7b041"
    # generateVendorColors: one NVIDIA series, dark theme -> L=.64 C=.15 H=145.
    green = oklch_hex(.64, .15, 145)
    # InferenceX's dark --overlay-run-{1,2,3,0} comparison tokens.
    colors = {
        "p50": oklch_hex(.75, .20, 190),
        "p90": oklch_hex(.78, .20, 90),
        "p95": oklch_hex(.70, .22, 290),
        "p99": oklch_hex(.72, .22, 25),
    }
    plt.rcParams.update({
        "font.family": families,
        "font.size": 13,
        "axes.unicode_minus": False,
        "axes.facecolor": bg,
        "axes.edgecolor": border,
        "axes.labelcolor": ink,
        "text.color": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "svg.fonttype": "path",
        "savefig.facecolor": bg,
    })
    runs = data["runs"]
    baseline = sorted(
        [r for r in runs if r["series"] == "same_configuration_concurrency_comparison"],
        key=lambda r: r["concurrency"],
    )
    historical = next(r for r in runs if r["series"] == "historical_internal_baseline")
    best = max(runs, key=lambda r: r["output_tps"])
    fig = plt.figure(figsize=(15, 9.6), facecolor=bg)
    fig.text(.055, .940, "DeepSeek V4.1 Flash  /  B200", fontsize=26, color=ink)
    fig.text(.945, .945, "PERFORMANCE", fontsize=12, color=gold, ha="right")
    fig.text(.055, .897,
             "AgentX  ·  16 GPU（8 Prefill + 8 Decode）  ·  正式全程（含突发阶段）",
             fontsize=13.5, color=muted)
    fig.add_artist(Line2D([.055, .945], [.867, .867],
                         transform=fig.transFigure, color=border, lw=.8))

    stats = [
        (f"{best['output_tps']/1000:.1f}k", "整机输出 tok/s", green),
        (f"{best['output_tps']/best['gpu_count']:,.0f}",
         "输出 tok/s/GPU · 按全部 16 卡", green),
        (f"{best['per_user_tps']['p50']:.2f}", "P50 per-user TPS", ink),
        (f"{best['ttft_ms']['p50']/1000:.2f} s", "P50 TTFT", gold),
    ]
    for i, (value, label, color) in enumerate(stats):
        x = .055 + i * .235
        fig.text(x, .800, value, fontsize=28, color=color)
        fig.text(x, .767, label, fontsize=11.5, color=muted)
        if i:
            fig.add_artist(Line2D([x - .022, x - .022], [.765, .833],
                                 transform=fig.transFigure, color=grid, lw=1))
    fig.text(.945, .851, "以上指标：11/02 配置 · 并发 2816",
             fontsize=10.5, color=muted, ha="right")

    fig.text(.080, .704, "实测配置对比：每卡吞吐 vs. 用户速度", fontsize=17, color=ink)
    fig.text(.682, .704, "同配置 TTFT 随并发变化", fontsize=16, color=ink)
    left = fig.add_axes([.080, .292, .470, .345])
    right = fig.add_axes([.682, .292, .263, .345])
    for ax in [left, right]:
        ax.set_axisbelow(True)
        ax.grid(color=grid, linewidth=.85)
        ax.tick_params(axis="both", labelsize=12, length=4, width=.7, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
            spine.set_color(border)

    curve = baseline
    xs = [r["per_user_tps"]["p50"] for r in curve]
    ys = [r["output_tps"] / r["gpu_count"] for r in curve]
    left.plot(xs, ys, color=green, linewidth=2.6, linestyle="-", marker="o",
              markersize=8, markeredgecolor=bg, markeredgewidth=1.4, zorder=3)
    left.scatter([best["per_user_tps"]["p50"]],
                 [best["output_tps"] / best["gpu_count"]],
                 s=100, marker="D", facecolors=gold, edgecolors=bg, linewidths=1.1,
                 zorder=4)
    offsets = {2816: (26, -30), 2432: (10, 18), 2048: (-8, -30)}
    for r in [*curve, best]:
        concurrency = r["concurrency"]
        latest = r is best
        suffix = "*" if r["formal_errors"] else ""
        note = "11/02 · " if latest else ""
        text = (f"{note}并发 {concurrency}{suffix}\n"
                f"{r['output_tps']/r['gpu_count']/1000:.2f}k tok/s/GPU")
        left.annotate(
            text, (r["per_user_tps"]["p50"], r["output_tps"] / r["gpu_count"]),
            xytext=(16, 48) if latest else offsets[concurrency],
            textcoords="offset points",
            ha="right" if concurrency == 2048 else "left", va="center",
            fontsize=11.5, color=ink, linespacing=1.5,
            bbox={"boxstyle": "round,pad=.35,rounding_size=.12",
                  "facecolor": bg if latest else "#246c39",
                  "edgecolor": gold if latest else green, "linewidth": .8},
            arrowprops={"arrowstyle": "-", "color": gold if latest else green,
                        "linewidth": .8, "shrinkA": 4, "shrinkB": 6},
        )
    hx = historical["per_user_tps"]["p50"]
    hy = historical["output_tps"] / historical["gpu_count"]
    left.scatter([hx], [hy], s=75, marker="s", facecolors="none",
                 edgecolors=muted, linewidths=1.5, zorder=4)
    left.annotate(
        "10/24 旧内部基线\n4.87k tok/s/GPU", (hx, hy),
        xytext=(-12, -22), textcoords="offset points", ha="right", va="center",
        fontsize=11, color=muted, linespacing=1.5,
        arrowprops={"arrowstyle": "-", "color": muted, "linewidth": .8},
    )
    left.set_xlim(88, 150)
    left.set_ylim(4350, 7100)
    left.set_xticks([90, 100, 110, 120, 130, 140, 150])
    left.set_yticks([4500, 5000, 5500, 6000, 6500, 7000])
    left.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:g}k"))
    left.set_xlabel("Per-user TPS（P50，tok/s/user）", labelpad=12, fontsize=13)
    left.set_ylabel("Output token throughput（tok/s/GPU）", labelpad=12, fontsize=12)
    left.text(.98, .97, "越往右上越好", transform=left.transAxes,
              ha="right", va="top", color=muted, fontsize=10.5)
    left.legend(
        handles=[Line2D([], [], color=green, marker="o", linewidth=2.5,
                        markersize=6, linestyle="-",
                        label="10/30–31 同配置，仅改变并发"),
                 Line2D([], [], color=gold, marker="D", linestyle="none",
                        markersize=6, label="11/02 · 共享限步状态 + 整步预算")],
        loc="lower left", bbox_to_anchor=(-.01, 1.025), frameon=False,
        fontsize=10.5, labelcolor=ink, borderaxespad=0, ncol=1,
    )

    cx = [r["concurrency"] for r in baseline]
    for stat in ["p99", "p95", "p90", "p50"]:
        values = [r["ttft_ms"][stat] / 1000 for r in baseline]
        right.plot(cx, values, color=colors[stat], marker="o", linestyle="-",
                   markersize=5.5, linewidth=2.1,
                   markeredgecolor=bg, markeredgewidth=.6)
        right.annotate(
            f"{stat.upper()}  {values[-1]:.2f}s", (cx[-1], values[-1]),
            xytext=(9, 0), textcoords="offset points", va="center",
            fontsize=10.8, color=colors[stat],
        )
    right.set_xlim(1990, 3310)
    right.set_ylim(0, 27)
    right.set_xticks(cx, ["2048", "2432*", "2816"])
    right.set_yticks([0, 5, 10, 15, 20, 25])
    right.set_xlabel("客户端并发", labelpad=12, fontsize=13)
    right.set_ylabel("秒 · 越小越好", labelpad=9, fontsize=12)
    right.text(0, 1.045, "P50 / P90 / P95 / P99",
               transform=right.transAxes, color=muted, fontsize=11)

    fig.add_artist(Line2D([.055, .945], [.205, .205],
                         transform=fig.transFigure, color=border, lw=.7))
    fig.text(.055, .163,
             "同配置下，并发提高时整机吞吐上升、单用户速度下降；11/02 配置另列。",
             fontsize=14, color=ink)
    fig.text(.055, .119,
             "口径：用户速度用 P50，避免均值极端值的影响；不含 TTFT。每卡吞吐按全部 16 卡计算。",
             fontsize=11.5, color=muted)
    fig.text(.055, .084,
             "灰点：10/24 旧内部基线（异机、异版本，1 错误）。* 并发 2432 有 4 个正式错误。",
             fontsize=11.5, color=muted)
    fig.text(.055, .039, "InferenceX 深色主题 · 投机接受率为固定模拟值，非质量评测",
             fontsize=10.3, color=muted)
    fig.text(.945, .039, "数据：本项目 AIPerf 导出  ·  整理：2026-11-03",
             fontsize=10.3, color=muted, ha="right")
    fig.savefig(here / "performance_overview.png", dpi=220, facecolor=bg)
    svg = here / "performance_overview.svg"
    fig.savefig(svg, facecolor=bg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
