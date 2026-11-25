import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, calc_metrics
from general_account_1x_study import (
    load_etf_tax_aware,
    simulate_general_account_portfolio,
    build_ma_signal,
    build_dual_momentum_weights,
)

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_monthly_mom_with_daily_ma_stop(
    closes: dict,
    asset_list: list,
    mom_days: int = 120,
    ma_len: int = 60,
    top_k: int = 1,
) -> pd.DataFrame:
    """
    월말 모멘텀 종목 선정 + 일간 추세 이탈 즉시 방어(KOFR 대피):
    - 매월 말: mom_days > 0 AND close > MA(ma_len) 인 자산 중 상위 top_k 선정
    - 월중: 보유 자산이 MA(ma_len)을 3영업일 연속 또는 -1% 이상 하향 이탈하면 즉시 KOFR 대피 (월말까지 대기하지 않아 급락장 MDD 급감!)
    """
    dates = closes[asset_list[0]].index
    for a in asset_list[1:]:
        dates = dates.intersection(closes[a].index)
    dates = dates.sort_values()
    n = len(dates)

    moms = {a: closes[a].pct_change(mom_days).loc[dates] for a in asset_list}
    ma_sigs = {a: build_ma_signal(closes[a], ma_len=ma_len, confirm=3, brk=0.01).loc[dates] for a in asset_list}

    is_me = np.zeros(n, dtype=bool)
    is_me[0] = True
    for i in range(n - 1):
        if dates[i].month != dates[i + 1].month:
            is_me[i] = True

    W = np.zeros((n, len(asset_list)), dtype=float)
    cur_w = np.zeros(len(asset_list), dtype=float)

    for i in range(n):
        if is_me[i]:
            scored = []
            for j, a in enumerate(asset_list):
                mv = moms[a].iloc[i]
                sig = ma_sigs[a].iloc[i]
                if not np.isnan(mv) and mv > 0 and sig == 1:
                    scored.append((mv, j))
            scored.sort(reverse=True)
            cur_w = np.zeros(len(asset_list), dtype=float)
            for _, j in scored[:top_k]:
                cur_w[j] = 1.0 / top_k
        else:
            # 월중 일간 추세 이탈 시 해당 슬롯만 KOFR로 대피
            for j, a in enumerate(asset_list):
                if cur_w[j] > 0 and ma_sigs[a].iloc[i] == 0:
                    cur_w[j] = 0.0
        W[i] = cur_w
    return pd.DataFrame(W, index=dates, columns=asset_list)


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

    df = df_base.loc[common_idx].copy()
    kofr_ret = df["kofr_ret"].copy()
    kofr_ret.iloc[0] = 0.0
    kofr_at = kofr_ret * (1.0 - 0.154)

    full_closes = {k: loaded[k]["adj_close"] for k in loaded}
    price_rets = {k: loaded[k].loc[common_idx, "price_ret"] for k in loaded}
    total_rets = {k: loaded[k].loc[common_idx, "total_ret"] for k in loaded}
    div_rets = {k: loaded[k].loc[common_idx, "div_ret"] for k in loaded}
    is_taxable = {k: etf_files[k][1] for k in loaded}

    def run_eval(W):
        eq, ret, turn, tax_p, _ = simulate_general_account_portfolio(
            common_idx, W.loc[common_idx], price_rets, total_rets, div_rets, is_taxable, kofr_ret
        )
        m = calc_metrics(eq, ret, kofr_at)
        # Pre-2025 check (2016-06 ~ 2024-12, 8.5년 — 2025년 코스피 폭등 제외!)
        r_pre25 = ret.loc[:"2024-12-31"]
        m_pre25 = calc_metrics((1 + r_pre25).cumprod(), r_pre25, kofr_at.loc[:"2024-12-31"])
        # Half1 vs Half2
        r1 = ret.loc[:"2021-12-31"]
        r2 = ret.loc["2022-01-01":]
        m1 = calc_metrics((1 + r1).cumprod(), r1, kofr_at.loc[:"2021-12-31"])
        m2 = calc_metrics((1 + r2).cumprod(), r2, kofr_at.loc["2022-01-01":])
        dd = eq / eq.cummax() - 1.0
        mdd_date = dd.idxmin().strftime("%Y-%m-%d")
        yrs = (common_idx[-1] - common_idx[0]).days / 365.25
        return m, m_pre25, m1, m2, mdd_date, tax_p / 1e4, turn.sum() / yrs, eq, ret

    # ==========================================================================
    # 1. 국내 2지수(KODEX 200 + 코스닥150) & 한미 3지수: 월간 vs 월간+일간MA방어 비교
    # ==========================================================================
    print("=" * 135)
    print(" [1] 일반계좌 1x 지수 전용 — 월간 모멘텀 vs '월간모멘텀 + 일간MA 방어' 비교 (2025 코스피 폭등 제외 구간 2016~2024 포함!)")
    print("=" * 135)
    cands = [
        ("A1. [단독] KODEX 200 MA60 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 60).loc[common_idx]}, index=common_idx)),
        ("A2. [단독] KODEX 200 MA100 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 100).loc[common_idx]}, index=common_idx)),
        ("A3. [단독] KODEX 200 MA120 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 120).loc[common_idx]}, index=common_idx)),
        ("B1. [국내2지수] 월간 6M 모멘텀 Top-1 + MA120 (비과세)", build_dual_momentum_weights(full_closes, ["k200", "kq"], mom_days=120, ma_len=120, top_k=1, rebal_freq="M")),
        ("B2. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA60 방어 (비과세)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], mom_days=120, ma_len=60, top_k=1)),
        ("B3. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA80 방어 (비과세)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], mom_days=120, ma_len=80, top_k=1)),
        ("B4. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA100 방어 (비과세)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], mom_days=120, ma_len=100, top_k=1)),
        ("B5. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA120 방어 (비과세)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq"], mom_days=120, ma_len=120, top_k=1)),
        ("C1. [한·미 3지수] 월간 6M 모멘텀 Top-2 + MA120 (세후)", build_dual_momentum_weights(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=120, top_k=2, rebal_freq="M")),
        ("C2. [한·미 3지수] 월간 6M 모멘텀 Top-2 + 일간MA60 방어 (세후)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=60, top_k=2)),
        ("C3. [한·미 3지수] 월간 6M 모멘텀 Top-2 + 일간MA80 방어 (세후)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=80, top_k=2)),
        ("C4. [한·미 3지수] 월간 6M 모멘텀 Top-2 + 일간MA100 방어 (세후)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=100, top_k=2)),
        ("C5. [한·미 3지수] 월간 6M 모멘텀 Top-1 + 일간MA60 방어 (세후)", build_monthly_mom_with_daily_ma_stop(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=60, top_k=1)),
    ]

    rows = []
    saved_rets = {}
    for name, W in cands:
        m, m_p25, m1, m2, mdd_dt, tax_m, turn_y, eq, ret = run_eval(W)
        rows.append({
            "전략": name,
            "전체CAGR": f"{m['cagr_pct']:+.2f}%",
            "MDD": f"{m['mdd_pct']:.1f}%",
            "MDD발생일": mdd_dt,
            "칼마": f"{m['calmar']:.2f}",
            "초과샤프": f"{m['sharpe_excess']:.2f}",
            "16~24(8.5년)": f"{m_p25['cagr_pct']:+.1f}%/{m_p25['mdd_pct']:.0f}%",
            "전반기(16~21)": f"{m1['cagr_pct']:+.1f}%",
            "후반기(22~26)": f"{m2['cagr_pct']:+.1f}%",
            "세금(만)": f"{tax_m:,.0f}",
            "연회전": f"{turn_y:.1f}",
        })
        saved_rets[name] = ret
    print(pd.DataFrame(rows).to_string(index=False))

    # ==========================================================================
    # 2. 파라미터 강건성 그리드: B(국내2지수 비과세) & C(한·미 3지수 Top-2 세후)
    # ==========================================================================
    print("\n" + "=" * 115)
    print(" [2] 파라미터 강건성 그리드 (세후 CAGR / MDD): mom_days × 일간 MA 방어선")
    print("=" * 115)
    for title, assets, top_k in [
        ("국내 2지수 (KODEX 200 + 코스닥150, Top-1, 매매차익 비과세)", ["k200", "kq"], 1),
        ("한·미 3지수 (KODEX 200 + 코스닥150 + 나스닥100, Top-2, 일반계좌 세후)", ["k200", "kq", "nq"], 2),
    ]:
        print(f"\n--- {title} ---")
        tbl = pd.DataFrame(index=[f"mom={d}일" for d in [60, 90, 120, 180]], columns=[f"MA{m}" for m in [50, 60, 80, 100, 120]])
        for d in [60, 90, 120, 180]:
            for ml in [50, 60, 80, 100, 120]:
                W = build_monthly_mom_with_daily_ma_stop(full_closes, assets, mom_days=d, ma_len=ml, top_k=top_k)
                m, _, _, _, _, _, _, _, _ = run_eval(W)
                tbl.loc[f"mom={d}일", f"MA{ml}"] = f"{m['cagr_pct']:+.1f}% / {m['mdd_pct']:.0f}%"
        print(tbl.to_string())

    # ==========================================================================
    # 3. 연도별 세후 수익률 비교 (핵심 4종)
    # ==========================================================================
    print("\n" + "=" * 115)
    print(" [3] 핵심 4종 연도별 일반계좌 세후 수익률 (2016~2026)")
    print("=" * 115)
    pick_keys = [
        ("KODEX200_MA60", "A1. [단독] KODEX 200 MA60 (비과세)"),
        ("국내2지수_일MA60", "B2. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA60 방어 (비과세)"),
        ("국내2지수_일MA80", "B3. [국내2지수] 월간 6M 모멘텀 Top-1 + 일간MA80 방어 (비과세)"),
        ("한미3지수Top2_일MA80", "C3. [한·미 3지수] 월간 6M 모멘텀 Top-2 + 일간MA80 방어 (세후)"),
    ]
    yr_df = pd.DataFrame({short: saved_rets[full] for short, full in pick_keys})
    yr_rows = []
    for yr, grp in yr_df.groupby(yr_df.index.year):
        d = {"연도": yr}
        for col in grp.columns:
            d[col] = f"{((1 + grp[col]).prod() - 1) * 100:+.1f}%"
        yr_rows.append(d)
    print(pd.DataFrame(yr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
