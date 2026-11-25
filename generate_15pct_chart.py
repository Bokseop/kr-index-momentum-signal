import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, calc_metrics
from fsm_engine_v2 import load_aligned_etf, run_fsm_generalized
from test_15pct_strategies import run_single_ma_trend, run_global_dual_momentum

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity\brain\3a9e90af-e676-4f93-8d95-ae0d04ca7c15"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}
    df0 = load_market_data(base_cfg)
    lev2x = load_aligned_etf("kodex_kosdaq150_lev2x_nav.csv", "lev2x", "2016-01-01", "2026-11-24")
    inv1x = load_aligned_etf("kodex_kosdaq150_inv1x_nav.csv", "inv1x", "2016-01-01", "2026-11-24")
    nq = load_aligned_etf("tiger_nasdaq100_nav.csv", "nq", "2015-01-01", "2026-11-24")
    k200 = load_aligned_etf("kodex_200_nav.csv", "k200", "2015-01-01", "2026-11-24")
    gold = load_aligned_etf("kodex_gold_nav.csv", "gold", "2015-01-01", "2026-11-24")

    df = df0.join([lev2x, inv1x, nq, k200, gold], how="inner").copy()
    kofr_s = df["kofr_ret"].copy()
    kofr_s.iloc[0] = 0.0

    # Key series
    bh_kq1 = df["etf_ret"].copy(); bh_kq1.iloc[0] = 0.0
    bh_kq2 = df["lev2x_ret"].copy(); bh_kq2.iloc[0] = 0.0
    bh_nq = df["nq_ret"].copy(); bh_nq.iloc[0] = 0.0

    cfg_orig = {**base_cfg, "cr_overheat_pct": 2.0, "cr_calm_pct": 1.8}
    r_orig = run_fsm_generalized(df, cfg_orig, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})["strat_ret"]
    r_cal = run_fsm_generalized(df, base_cfg, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})["strat_ret"]
    r_1b = run_fsm_generalized(df, base_cfg, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})["strat_ret"]
    r_2b = run_fsm_generalized(df, base_cfg, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0.5, 0), 4: (0.3, 0, 0, 0)})["strat_ret"]
    r_nq_ma, _, _ = run_single_ma_trend(df["nq_close"], df["nq_ret"], df["kofr_ret"], ma_len=120, fee=0.0008)
    r_3c = 0.5 * r_cal + 0.5 * r_nq_ma
    r_3g = 0.5 * r_1b + 0.5 * r_nq_ma
    r_3f, _ = run_global_dual_momentum(df, fee=0.0008, top_k=2, mom_days=120)
    r_hyb = 0.4 * r_cal + 0.6 * r_3f

    # Metrics for Frontier
    points = [
        ("코스닥150 B&H", bh_kq1, "#888888", "o", (-75, 10)),
        ("원본 FSM (2.0%)", r_orig, "#7aa6c2", "s", (10, -6)),
        ("보정 FSM 1x (2.4%)", r_cal, "#1f77b4", "s", (10, -6)),
        ("경로1: 코스닥 S1 2x", r_1b, "#ff7f0e", "^", (-35, -30)),
        ("경로2: S1 2x + 인버스", r_2b, "#d62728", "^", (-20, 12)),
        ("50:50 노레버(코스닥+나스닥)", r_3c, "#2ca02c", "D", (10, -14)),
        ("추천A: 40%FSM + 60%글로벌Top2\n(노레버 1x)", r_hyb, "#008080", "*", (12, -32)),
        ("추천B: 50%FSM(2x) + 50%나스닥\n(저레버)", r_3g, "#9467bd", "*", (-72, 18)),
        ("추천C: 글로벌 모멘텀 Top-2\n(노레버 1x)", r_3f, "#c0392b", "*", (12, -8)),
        ("나스닥100 B&H", bh_nq, "#555555", "o", (10, -2)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Left: Efficient Frontier (|MDD| vs CAGR)
    ax1.axhline(15.0, color="#c0392b", ls="--", lw=1.3, alpha=0.8, label="목표 CAGR 15% 선")
    ax1.axvline(25.0, color="#2c3e50", ls=":", lw=1.1, alpha=0.6, label="실전 심리적 MDD 한계 (-25%)")
    ax1.axvspan(0, 25, ymin=(15.0 - (-2)) / (24 - (-2)), ymax=1.0, color="#2ecc71", alpha=0.10, label="최적 골든존 (CAGR≥15%, MDD≤25%)")

    for lbl, r_s, col, mk, offset in points:
        eq = 10_000_000 * (1 + r_s).cumprod()
        m = calc_metrics(eq, r_s, kofr_s)
        x = abs(m["mdd_pct"])
        y = m["cagr_pct"]
        sz = 230 if mk == "*" else 85
        ax1.scatter(x, y, s=sz, color=col, marker=mk, edgecolors="black", linewidths=0.8, zorder=5)
        ax1.annotate(
            f"{lbl}\n({y:+.1f}% / -{x:.1f}%)",
            xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=8.2, color="#222222",
            fontweight="bold" if mk == "*" else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.72)
        )

    ax1.set_title("위험-수익 효율적 프런티어 (|MDD| vs CAGR, 2016~2026 실데이터)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlabel("최대낙폭 |MDD| (%) — 왼쪽일수록 안전", fontsize=10)
    ax1.set_ylabel("연복리 수익률 CAGR (%) — 위쪽일수록 고수익", fontsize=10)
    ax1.set_xlim(10, 65)
    ax1.set_ylim(-2, 24)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    # Right: Log-scale Equity Curve
    curves = [
        ("추천C: 글로벌 듀얼모멘텀 Top-2 (1x, +18.3%)", r_3f, "#c0392b", "-", 2.2),
        ("추천B: 50%코스닥FSM(S1 2x)+50%나스닥 (+15.5%)", r_3g, "#9467bd", "-", 2.2),
        ("추천A: 40%코스닥FSM(1x)+60%글로벌Top2 (+15.2%)", r_hyb, "#008080", "-", 2.2),
        ("경로1: 코스닥150 FSM S1 2x 단독 (+14.1%, MDD -48%)", r_1b, "#ff7f0e", "--", 1.5),
        ("보정 FSM 1x 단독 (+9.7%, MDD -24%)", r_cal, "#1f77b4", "--", 1.4),
        ("코스닥150 1x 단순보유 (+1.1%, MDD -58%)", bh_kq1, "#7f7f7f", "-", 1.2),
        ("코스닥150 2x 단순보유 (-9.1%, MDD -88%)", bh_kq2, "#aaaaaa", ":", 1.2),
    ]
    for lbl, r_s, col, ls, lw in curves:
        eq = 1000 * (1 + r_s).cumprod()  # 만원 단위
        ax2.plot(eq.index, eq, label=lbl, color=col, ls=ls, lw=lw)

    ax2.set_yscale("log")
    ax2.set_yticks([500, 1000, 2000, 3000, 4000, 6000])
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}만"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_title("10.3년 누적 자산 곡선 (초기자본 1,000만 원, 로그 스케일)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylabel("자산 (만 원, 로그)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.22)
    ax2.legend(loc="upper left", fontsize=8.3, framealpha=0.92)

    fig.suptitle("연평균 15% CAGR 달성 경로 실데이터 비교 (KOFIA 공식 2016-08 ~ 2026-11, ISA 기준, 편도 8bp)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = os.path.join(BASE_DIR, "chart_paths_to_15pct.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    shutil.copy2(out, os.path.join(ARTIFACT_DIR, "chart_paths_to_15pct.png"))
    print("Saved:", out)


if __name__ == "__main__":
    main()
