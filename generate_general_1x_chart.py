import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, calc_metrics
from general_account_1x_study import load_etf_tax_aware, simulate_general_account_portfolio, build_ma_signal
from verify_general_account_robustness import build_monthly_mom_with_daily_ma_stop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity\brain\3a9e90af-e676-4f93-8d95-ae0d04ca7c15"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True}
    df_base = load_market_data(base_cfg)
    etf_files = {
        "kq": ("kodex_kosdaq150_nav.csv", False),
        "k200": ("kodex_200_nav.csv", False),
        "nq": ("tiger_nasdaq100_nav.csv", True),
    }
    loaded = {k: load_etf_tax_aware(fn, is_domestic_equity=(not is_tax)) for k, (fn, is_tax) in etf_files.items()}
    common_idx = df_base.index
    for k in loaded:
        common_idx = common_idx.intersection(loaded[k].index)
    common_idx = common_idx.sort_values()

    kofr_ret = df_base.loc[common_idx, "kofr_ret"].copy()
    kofr_ret.iloc[0] = 0.0
    kofr_at = kofr_ret * (1.0 - 0.154)

    full_closes = {k: loaded[k]["adj_close"] for k in loaded}
    price_rets = {k: loaded[k].loc[common_idx, "price_ret"] for k in loaded}
    total_rets = {k: loaded[k].loc[common_idx, "total_ret"] for k in loaded}
    div_rets = {k: loaded[k].loc[common_idx, "div_ret"] for k in loaded}
    is_taxable = {k: etf_files[k][1] for k in loaded}

    specs = [
        ("해법1-A: 국내2지수(코스피200+코스닥150) 6M Top1 + 일간MA100 (비과세)",
         build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], 120, 100, 1), "#c0392b", "-", 2.3),
        ("해법1-B: 국내2지수(코스피200+코스닥150) 6M Top1 + 일간MA60 (비과세)",
         build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], 120, 60, 1), "#e67e22", "--", 1.7),
        ("해법2: 한·미 3지수(200+코스닥150+나스닥) 6M Top2 + 일간MA100 (세후)",
         build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], 120, 100, 2), "#1f77b4", "-", 2.3),
        ("해법2-안정: 한·미 3지수 6M Top2 + 일간MA80 (세후, MDD -19%)",
         build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], 120, 80, 2), "#008080", "--", 1.7),
        ("참고: KODEX 200 단순보유 (비과세)",
         pd.DataFrame({"k200": 1.0}, index=common_idx), "#555555", ":", 1.3),
        ("참고: KODEX 코스닥150 단순보유 (비과세)",
         pd.DataFrame({"kq": 1.0}, index=common_idx), "#aaaaaa", ":", 1.2),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.2], "hspace": 0.12})

    for lbl, W, col, ls, lw in specs:
        eq, ret, _, tax_p, _ = simulate_general_account_portfolio(
            common_idx, W.loc[common_idx], price_rets, total_rets, div_rets, is_taxable, kofr_ret
        )
        m = calc_metrics(eq, ret, kofr_at)
        dd = (eq / eq.cummax() - 1.0) * 100.0
        tag = f"{lbl}  [{m['cagr_pct']:+.1f}% / MDD {m['mdd_pct']:.1f}%]"
        ax1.plot(eq.index, eq / 1e4, label=tag, color=col, ls=ls, lw=lw)
        if "단순보유" in lbl and "코스닥" in lbl:
            ax2.fill_between(dd.index, dd, 0, color=col, alpha=0.18, label="코스닥150 B&H (MDD -57.8%)")
        elif "단순보유" not in lbl:
            ax2.plot(dd.index, dd, color=col, ls=ls, lw=1.4, label=lbl.split(":")[0] + f" ({m['mdd_pct']:.1f}%)")

    ax1.set_yscale("log")
    ax1.set_yticks([1000, 2000, 3000, 5000, 8000, 10000])
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}만"))
    ax1.set_title("일반계좌 · 1x 노레버리지 · 지수 전용 세후 백테스트 (2016-06 ~ 2026-11, 초기 1,000만 원, 편도 8bp)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_ylabel("세후 순자산 (만 원, 로그)", fontsize=10)
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(loc="upper left", fontsize=8.4, framealpha=0.92)

    ax2.set_ylabel("낙폭 DD (%)", fontsize=10)
    ax2.set_ylim(-62, 3)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.92, ncol=3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = os.path.join(BASE_DIR, "chart_general_account_1x.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    shutil.copy2(out, os.path.join(ARTIFACT_DIR, "chart_general_account_1x.png"))
    print("Saved:", out)


if __name__ == "__main__":
    main()
