"""Render line charts from existing B200 benchmark summaries; no benchmark runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


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
        "font.size": 13,
        "axes.unicode_minus": False,
        "axes.edgecolor": "#c8d0da",
        "axes.labelcolor": "#334155",
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "svg.fonttype": "path",
    })
    ink, muted, grid = "#0f172a", "#64748b", "#e5eaf0"
    colors = {
        "avg": "#049b82", "p10": "#94a3b8", "p50": "#3291cf",
        "p90": "#7563d9", "p95": "#d38a28", "p99": "#ca5473",
    }
    runs = sorted(data["runs"], key=lambda r: r["concurrency"])
    best = max(runs, key=lambda r: r["output_tps"])
    xs = [r["concurrency"] for r in runs]
    fig = plt.figure(figsize=(16, 9.8), facecolor="white")
    fig.text(.05, .945, "DeepSeek V4.1 Flash  /  B200 性能趋势",
             fontsize=27, color=ink)
    fig.text(.05, .900, "AgentX  ·  16 GPU（8 Prefill + 8 Decode）  ·  2026-11-03 整理",
             fontsize=14, color=muted)
    fig.text(.05, .852, "当前最高吞吐配置：并发 2816  ·  以下均为正式全程，包含突发阶段",
             fontsize=14, color=colors["avg"])

    cards = [
        (f"{best['output_tps']/1000:.1f}k", "整机输出 tok/s · 16 GPU"),
        (f"{best['output_tps']/best['gpu_count']:,.0f}", "输出 tok/s/GPU · 按全部 16 卡"),
        (f"{best['per_user_tps']['avg']:.2f}", "平均 per-user TPS"),
        (f"{best['ttft_ms']['avg']/1000:.2f} s", "平均首 token 等待（TTFT）"),
    ]
    for i, (value, label) in enumerate(cards):
        x = .05 + i * .238
        fig.add_artist(FancyBboxPatch(
            (x, .735), .222, .09, transform=fig.transFigure,
            boxstyle="round,pad=0.007,rounding_size=0.007",
            facecolor="#f7f9fc", edgecolor="#e7ecf2", linewidth=.8,
        ))
        fig.text(x + .013, .780, value, fontsize=28, color=ink)
        fig.text(x + .013, .748, label, fontsize=12, color=muted)

    axes = [fig.add_axes([x, .280, .232, .365]) for x in [.069, .393, .717]]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=12.5, length=0, pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=grid, linewidth=.8)
        ax.set_xlim(1990, 3240)
        ax.set_xticks(xs, [f"{x}" + ("*" if x == 2432 else "") for x in xs])
        ax.set_xlabel("客户端并发", labelpad=12, fontsize=13)

    throughput, user, ttft = axes
    titles = ["整机输出吞吐 ↑", "用户生成速度 ↑", "首 token 等待 ↓"]
    units = ["k tok/s", "tok/s/user · 不含 TTFT", "秒"]
    for x, title, unit in zip([.069, .393, .717], titles, units):
        fig.text(x, .688, title, fontsize=19, color=ink)
        fig.text(x, .653, unit, fontsize=12, color=muted)

    values = [r["output_tps"] / 1000 for r in runs]
    throughput.plot(xs, values, color=colors["avg"], marker="o", markersize=8,
                    linewidth=3, markeredgecolor="white", markeredgewidth=1.3)
    throughput.set_ylim(0, 112)
    throughput.set_yticks([0, 25, 50, 75, 100])
    throughput.fill_between(xs, values, color=colors["avg"], alpha=.055)
    for x, y in zip(xs, values):
        throughput.annotate(f"{y:.1f}k", (x, y), xytext=(0, 13),
                            textcoords="offset points", ha="center", fontsize=14,
                            color=colors["avg"])
    throughput.text(.05, .08, "83.1k → 100.2k tok/s", transform=throughput.transAxes,
                    fontsize=13, color=ink)

    # Low TPS percentiles describe slower requests. No smoothing/interpolation.
    for stat in ["p99", "p90", "avg", "p50", "p10"]:
        values = [r["per_user_tps"][stat] for r in runs]
        user.plot(xs, values, color=colors[stat], marker="o", markersize=5.5,
                  linewidth=2.5 if stat == "avg" else 1.8,
                  linestyle="--" if stat == "p10" else "-")
        label = "均值" if stat == "avg" else stat.upper()
        user.annotate(f"{label} {values[-1]:.1f}", (xs[-1], values[-1]),
                      xytext=(9, 0), textcoords="offset points", va="center",
                      fontsize=11.5, color=colors[stat])
    user.set_xlim(1990, 3240)
    user.set_ylim(65, 265)
    user.set_yticks([75, 100, 150, 200, 250])

    for stat in ["p99", "p95", "p90", "p50"]:
        values = [r["ttft_ms"][stat] / 1000 for r in runs]
        ttft.plot(xs, values, color=colors[stat], marker="o", markersize=5.5,
                  linewidth=2)
        ttft.annotate(f"{stat.upper()} {values[-1]:.2f}", (xs[-1], values[-1]),
                      xytext=(9, 0), textcoords="offset points", va="center",
                      fontsize=11.5, color=colors[stat])
    ttft.set_xlim(1990, 3240)
    ttft.set_ylim(0, 27)
    ttft.set_yticks([0, 5, 10, 15, 20, 25])

    fig.text(.05, .175, "并发 2816：吞吐最高；并发 2048：用户生成更快，首 token 等待更短。",
             fontsize=15, color=ink)
    fig.text(.05, .123, "口径：均为正式全程。Per-user TPS 先逐请求计算，再汇总；低分位代表较慢请求。",
             fontsize=12, color=muted)
    fig.text(.05, .088, "* 并发 2432 有 4 个正式错误。各点对应不同实测配置，连线仅便于比较，不代表只改变并发的扫描。",
             fontsize=12, color=muted)
    fig.text(.05, .045, "视觉风格参考 InferenceX  ·  数据：本项目 AIPerf 原始导出  ·  投机接受率为固定模拟值，不代表质量评测。",
             fontsize=11.5, color=muted)

    fig.savefig(here / "performance_overview.png", dpi=200, facecolor="white")
    fig.savefig(here / "performance_overview.svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
