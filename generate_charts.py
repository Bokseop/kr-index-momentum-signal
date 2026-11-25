import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, run_backtest, calc_metrics
from run_robustness_analysis import eval_variant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity\brain\3a9e90af-e676-4f93-8d95-ae0d04ca7c15"

# Windows Korean font setup
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def make_chart1_equity_and_states(res_orig, res_cal24, res_ma50, res_ma100, s4_orig):
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 12.5), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.5, 1.4, 1.1], "hspace": 0.14}
    )

    # Panel 1: Cumulative Equity (Log scale or normal -> normal in 만원 is super readable here)
    ax1 = axes[0]
    ax1.plot(res_orig.index, res_orig["strat_equity"] / 1e4, label="원본 4단계 FSM (과열 2.0%/1.8%, ISA)", color="#1f77b4", lw=2.2)
    ax1.plot(res_cal24.index, res_cal24["strat_equity"] / 1e4, label="보정 4단계 FSM (과열 2.4%/2.1%, ISA)", color="#d62728", lw=2.0)
    ax1.plot(res_ma50.index, res_ma50["strat_equity"] / 1e4, label="귀무가설: 단순 MA120 (50%/0%)", color="#2ca02c", lw=1.4, ls="--")
    ax1.plot(res_orig.index, res_orig["bh_equity"] / 1e4, label="코스닥150 단순보유 (B&H)", color="#7f7f7f", lw=1.3, alpha=0.85)
    ax1.plot(res_orig.index, res_orig["kofr_equity"] / 1e4, label="KOFR/CD91 현금보유", color="#ff7f0e", lw=1.2, ls=":")

    # Mark KOFR actual listing date
    kofr_start = pd.Timestamp("2022-04-26")
    ax1.axvline(kofr_start, color="#555555", ls="-.", lw=1.0, alpha=0.7)
    ax1.text(kofr_start + pd.Timedelta(days=25), ax1.get_ylim()[1] * 0.92 if ax1.get_ylim()[1] > 0 else 2400,
             "KOFR ETF 상장\n(2022-04-26)", fontsize=8.5, color="#444444", va="top")

    ax1.set_title("코스닥150 4단계 상태 머신(FSM) 실데이터 백테스트 (2016-06 ~ 2026-11, 초기자본 1,000만 원)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("자산 (만 원)", fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92, ncol=2)

    # Panel 2: Drawdown (%)
    ax2 = axes[1]
    for df_r, col, lbl, c, ls in [
        (res_orig, "strat_equity", "원본 FSM (MDD -19.6%)", "#1f77b4", "-"),
        (res_cal24, "strat_equity", "보정 FSM 2.4% (MDD -23.6%)", "#d62728", "-"),
        (res_orig, "bh_equity", "코스닥150 B&H (MDD -57.7%)", "#7f7f7f", "-"),
    ]:
        dd = (df_r[col] / df_r[col].cummax() - 1.0) * 100.0
        if "B&H" in lbl:
            ax2.fill_between(df_r.index, dd, 0, color=c, alpha=0.18, label=lbl)
        else:
            ax2.plot(df_r.index, dd, color=c, lw=1.4, ls=ls, label=lbl)
    ax2.axvline(kofr_start, color="#555555", ls="-.", lw=1.0, alpha=0.7)
    ax2.set_ylabel("낙폭 DD (%)", fontsize=10)
    ax2.set_ylim(-63, 3)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9, ncol=3)

    # Panel 3: Real FreeSIS KOSDAQ Credit Ratio (%) vs Thresholds
    ax3 = axes[2]
    ax3.plot(res_orig.index, res_orig["cr_ratio"], color="#333333", lw=1.2, label="실제 코스닥 신용비율 (FreeSIS, shift(2))")
    ax3.axhline(2.0, color="#1f77b4", ls="--", lw=1.1, label="원본 과열선 2.0% (전체의 70%가 초과)")
    ax3.axhline(1.8, color="#1f77b4", ls=":", lw=1.0, label="원본 진정선 1.8%")
    ax3.axhline(2.4, color="#d62728", ls="--", lw=1.1, label="보정 과열선 2.4% (상위 ~25%)")
    ax3.fill_between(res_orig.index, 2.0, res_orig["cr_ratio"], where=(res_orig["cr_ratio"] >= 2.0), color="#1f77b4", alpha=0.10)
    ax3.fill_between(res_orig.index, 2.4, res_orig["cr_ratio"], where=(res_orig["cr_ratio"] >= 2.4), color="#d62728", alpha=0.18)
    ax3.set_ylabel("신용비율 (%)", fontsize=10)
    ax3.set_ylim(1.2, 3.0)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    # Panel 4: State Timeline (Original vs Calibrated 2.4%)
    ax4 = axes[3]
    state_colors = {1: "#2ca02c", 2: "#ffbf00", 3: "#d9d9d9", 4: "#9467bd"}
    dates_num = mdates.date2num(res_orig.index)

    for y_pos, series, tag in [(0.55, res_orig["state"], "원본(2.0%)"), (0.05, res_cal24["state"], "보정(2.4%)")]:
        st_vals = series.values
        start_i = 0
        for i in range(1, len(st_vals) + 1):
            if i == len(st_vals) or st_vals[i] != st_vals[start_i]:
                x0 = dates_num[start_i]
                x1 = dates_num[i - 1] + 1.5
                ax4.barh(y_pos, x1 - x0, left=x0, height=0.38, color=state_colors[st_vals[start_i]], align="edge", edgecolor="none")
                start_i = i
    ax4.set_yticks([0.24, 0.74])
    ax4.set_yticklabels(["보정(2.4%)", "원본(2.0%)"], fontsize=9)
    ax4.set_ylim(0, 1.0)
    ax4.set_ylabel("국면(State)", fontsize=10)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legend_patches = [
        Patch(facecolor="#2ca02c", label="S1 정상상승(100%)"),
        Patch(facecolor="#ffbf00", label="S2 과열축소(50%)"),
        Patch(facecolor="#d9d9d9", label="S3 하락대피(0%)"),
        Patch(facecolor="#9467bd", label="S4 투매역발상(30%)"),
    ]
    ax4.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.58), ncol=4, fontsize=9, frameon=False)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08)
    out_path = os.path.join(BASE_DIR, "chart_equity_and_states.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, "chart_equity_and_states.png"))
    print("Saved:", out_path)


def make_chart2_sensitivity_and_ablation():
    sens = pd.read_csv(os.path.join(BASE_DIR, "sensitivity_grid.csv"))
    ma_order = [60, 80, 100, 120, 150, 180, 200]
    cr_order = ["1.8%", "2.0%", "2.2%", "2.4%", "2.6%", "20d급증만"]

    piv_cagr = sens.pivot(index="ma_long", columns="cr_overheat", values="cagr").reindex(index=ma_order, columns=cr_order)
    piv_calmar = sens.pivot(index="ma_long", columns="cr_overheat", values="calmar").reindex(index=ma_order, columns=cr_order)
    piv_mdd = sens.pivot(index="ma_long", columns="cr_overheat", values="mdd").reindex(index=ma_order, columns=cr_order)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    configs = [
        (axes[0], piv_cagr, "연복리 수익률 CAGR (%)", "YlGn", ".1f", "%"),
        (axes[1], piv_mdd, "최대낙폭 MDD (%)", "RdYlGn", ".1f", "%"),
        (axes[2], piv_calmar, "칼마비율 (CAGR / |MDD|)", "YlGnBu", ".2f", ""),
    ]

    for ax, piv, title, cmap, fmt, suf in configs:
        data = piv.values.astype(float)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks(range(len(cr_order)))
        ax.set_xticklabels(cr_order, fontsize=9, rotation=20)
        ax.set_yticks(range(len(ma_order)))
        ax.set_yticklabels([f"MA{m}" for m in ma_order], fontsize=9.5)
        ax.set_xlabel("신용비율 과열 기준 (cr_overheat_pct)", fontsize=9.5)
        if ax == axes[0]:
            ax.set_ylabel("장기 이평선 기간 (ma_long)", fontsize=9.5)

        vmin, vmax = np.nanmin(data), np.nanmax(data)
        mid = (vmin + vmax) / 2.0
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                val = data[r, c]
                # Choose readable text color
                if cmap == "RdYlGn":
                    txt_c = "white" if val < vmin + 0.25 * (vmax - vmin) else "black"
                else:
                    txt_c = "white" if val > mid + 0.15 * (vmax - vmin) else "black"
                # Highlight original (MA120, 2.0%) with box
                ax.text(c, r, f"{val:{fmt}}{suf}", ha="center", va="center", fontsize=8.8, color=txt_c,
                        fontweight="bold" if (ma_order[r] == 120 and cr_order[c] in ("2.0%", "2.2%", "2.4%")) else "normal")
                if ma_order[r] == 120 and cr_order[c] == "2.0%":
                    rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="#000000", lw=2.2)
                    ax.add_patch(rect)

    fig.suptitle("파라미터 강건성(과최적화) 히트맵 — 동일 구간(2016-10 ~ 2026-11) | 검은 테두리 = 원본 설정(MA120, 2.0%)", fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(BASE_DIR, "chart_sensitivity_heatmap.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, "chart_sensitivity_heatmap.png"))
    print("Saved:", out_path)


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True}
    base_df = load_market_data(base_cfg)
    _, res_orig, s4_orig = eval_variant(base_cfg, base_df=base_df)
    _, res_cal24, _ = eval_variant({**base_cfg, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}, base_df=base_df)
    _, res_ma50, _ = eval_variant(base_cfg, base_df=base_df, fixed_bull_weight=0.5)
    _, res_ma100, _ = eval_variant(base_cfg, base_df=base_df, fixed_bull_weight=1.0)

    make_chart1_equity_and_states(res_orig, res_cal24, res_ma50, res_ma100, s4_orig)
    make_chart2_sensitivity_and_ablation()


if __name__ == "__main__":
    main()
