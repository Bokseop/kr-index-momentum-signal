import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, calc_metrics
from fsm_engine_v2 import load_aligned_etf, run_fsm_generalized

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_single_ma_trend(close: pd.Series, ret: pd.Series, kofr_ret: pd.Series, ma_len: int = 120, fee: float = 0.0008, alt_safe_ret: pd.Series = None):
    """단일 자산 MA120 추세추종 (당일 종가 신호 -> 익일 수익률, 미래참조 없음)."""
    ma = close.rolling(ma_len).mean()
    disp = close / ma - 1.0
    above = (close > ma).astype(int)
    below = (close < ma).astype(int)
    ab_s = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    bl_s = below.groupby((below != below.shift()).cumsum()).cumsum() * below

    n = len(close)
    w = np.zeros(n)
    cur_w = 1.0 if close.iloc[0] > ma.iloc[0] else 0.0
    strat_ret = np.zeros(n)
    turns = np.zeros(n)

    for i in range(n):
        if i > 0:
            safe_r = alt_safe_ret.iloc[i] if alt_safe_ret is not None else kofr_ret.iloc[i]
            gross = cur_w * ret.iloc[i] + (1.0 - cur_w) * safe_r
        else:
            gross = 0.0

        cond_up = (ab_s.iloc[i] >= 3) or (disp.iloc[i] >= 0.01)
        cond_dn = (bl_s.iloc[i] >= 3) or (disp.iloc[i] <= -0.01)

        nxt_w = cur_w
        if cur_w == 1.0 and cond_dn:
            nxt_w = 0.0
        elif cur_w == 0.0 and cond_up:
            nxt_w = 1.0

        t = abs(nxt_w - cur_w) if i > 0 else 0.0
        strat_ret[i] = gross - t * fee
        w[i] = nxt_w
        turns[i] = t
        cur_w = nxt_w

    return pd.Series(strat_ret, index=close.index), pd.Series(w, index=close.index), pd.Series(turns, index=close.index)


def run_global_dual_momentum(df: pd.DataFrame, fee: float = 0.0008, top_k: int = 1, mom_days: int = 120):
    """
    글로벌 듀얼 모멘텀 회전 (노레버리지 1x, 월말 리밸런싱 + MA120 절대모멘텀 필터):
    후보 자산: KODEX 코스닥150, KODEX 200, TIGER 미국나스닥100, KODEX 골드선물(H)
    규칙: 매월 마지막 영업일 종가 기준으로 (1) 6개월(120일) 수익률 > 0 AND 종가 > MA120 인 자산만 통과,
         (2) 통과 자산 중 모멘텀 상위 top_k개에 균등 배분, 부족분은 KOFR/CD91 현금 대피.
         익영업일부터 수익 실현 (미래 참조 없음).
    """
    assets = ["etf", "k200", "nq", "gold"]
    closes = {a: df[f"{a}_close"] for a in assets}
    rets = {a: df[f"{a}_ret"] for a in assets}
    moms = {a: closes[a].pct_change(mom_days) for a in assets}
    mas = {a: closes[a].rolling(120).mean() for a in assets}

    dates = df.index
    n = len(df)
    # 월말 판정: 다음 영업일의 month가 현재 month와 다르면 월말
    is_month_end = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if dates[i].month != dates[i + 1].month:
            is_month_end[i] = True

    prev_w = np.zeros(len(assets))
    strat_ret = np.zeros(n)
    turns = np.zeros(n)

    for i in range(n):
        if i > 0:
            w_cash = max(0.0, 1.0 - prev_w.sum())
            gross = sum(prev_w[j] * rets[assets[j]].iloc[i] for j in range(len(assets))) + w_cash * df["kofr_ret"].iloc[i]
        else:
            gross = 0.0

        new_w = prev_w.copy()
        if i == 0 or is_month_end[i]:
            scored = []
            for j, a in enumerate(assets):
                m_val = moms[a].iloc[i]
                c_val = closes[a].iloc[i]
                ma_val = mas[a].iloc[i]
                if not np.isnan(m_val) and not np.isnan(ma_val) and m_val > 0 and c_val > ma_val:
                    scored.append((m_val, j))
            scored.sort(reverse=True)
            new_w = np.zeros(len(assets))
            for _, j in scored[:top_k]:
                new_w[j] = 1.0 / top_k

        prev_cash = max(0.0, 1.0 - prev_w.sum())
        new_cash = max(0.0, 1.0 - new_w.sum())
        t = 0.5 * (np.abs(new_w - prev_w).sum() + abs(new_cash - prev_cash)) if i > 0 else 0.0
        strat_ret[i] = gross - t * fee
        turns[i] = t
        prev_w = new_w

    return pd.Series(strat_ret, index=dates), pd.Series(turns, index=dates)


def summarize(name: str, ret_s: pd.Series, kofr_s: pd.Series, turns_s: pd.Series = None):
    eq = 10_000_000 * (1.0 + ret_s).cumprod()
    m_all = calc_metrics(eq, ret_s, kofr_s)
    # Half 1 (2016~2021) vs Half 2 (2022~2026)
    r1 = ret_s.loc[:"2021-12-31"]
    r2 = ret_s.loc["2022-01-01":]
    m1 = calc_metrics((1.0 + r1).cumprod(), r1, kofr_s.loc[:"2021-12-31"])
    m2 = calc_metrics((1.0 + r2).cumprod(), r2, kofr_s.loc["2022-01-01":])
    years = (ret_s.index[-1] - ret_s.index[0]).days / 365.25
    ann_turn = turns_s.sum() / years if turns_s is not None else 0.0
    return {
        "전략": name,
        "최종자산(만)": f"{m_all['final_equity']/1e4:,.0f}",
        "CAGR(%)": f"{m_all['cagr_pct']:+.2f}",
        "Vol(%)": f"{m_all['ann_vol_pct']:.1f}",
        "초과샤프": f"{m_all['sharpe_excess']:.2f}",
        "MDD(%)": f"{m_all['mdd_pct']:.1f}",
        "칼마": f"{m_all['calmar']:.2f}",
        "전반기(16~21)": f"{m1['cagr_pct']:+.1f}%/{m1['mdd_pct']:.0f}%",
        "후반기(22~26)": f"{m2['cagr_pct']:+.1f}%/{m2['mdd_pct']:.0f}%",
        "연회전": f"{ann_turn:.1f}",
    }, eq


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}
    df0 = load_market_data(base_cfg)

    # Load all extra real ETFs
    lev2x = load_aligned_etf("kodex_kosdaq150_lev2x_nav.csv", "lev2x", "2016-01-01", "2026-11-24")
    inv1x = load_aligned_etf("kodex_kosdaq150_inv1x_nav.csv", "inv1x", "2016-01-01", "2026-11-24")
    nq = load_aligned_etf("tiger_nasdaq100_nav.csv", "nq", "2015-01-01", "2026-11-24")
    k200 = load_aligned_etf("kodex_200_nav.csv", "k200", "2015-01-01", "2026-11-24")
    gold = load_aligned_etf("kodex_gold_nav.csv", "gold", "2015-01-01", "2026-11-24")

    # Pre-compute MA120 trend flags on full history before slicing to common start
    nq["nq_ma120"] = nq["nq_close"].rolling(120).mean()
    nq["nq_trend_on"] = nq["nq_close"] > nq["nq_ma120"]
    gold["gold_ma120"] = gold["gold_close"].rolling(120).mean()
    gold["gold_trend_on"] = gold["gold_close"] > gold["gold_ma120"]

    df = df0.join([lev2x, inv1x, nq, k200, gold], how="inner").copy()
    # Clean Day 0 returns
    for c in ["etf_ret", "lev2x_ret", "inv1x_ret", "nq_ret", "k200_ret", "gold_ret", "kofr_ret"]:
        df[c] = df[c].fillna(0.0)
    kofr_s = df["kofr_ret"].copy()
    kofr_s.iloc[0] = 0.0

    print("=" * 125)
    print(f" [연 15% CAGR 달성 경로 실데이터 검증] 공통 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}영업일, ISA, 현실 CD91->KOFR, 편도 8bp)")
    print("=" * 125)

    rows = []
    equities = {}

    # --- Group 0: Baselines ---
    for lbl, col in [
        ("0A. 코스닥150 1x 단순보유 (229200)", "etf_ret"),
        ("0B. 코스닥150 2x 레버리지 단순보유 (233740)", "lev2x_ret"),
        ("0C. TIGER 미국나스닥100 단순보유 (133690)", "nq_ret"),
    ]:
        r = df[col].copy()
        r.iloc[0] = 0.0
        row, eq = summarize(lbl, r, kofr_s)
        rows.append(row)
        equities[lbl] = eq

    # Original & Calibrated 1x FSM
    cfg_orig = {**base_cfg, "cr_overheat_pct": 2.0, "cr_calm_pct": 1.8}
    res_orig = run_fsm_generalized(df, cfg_orig, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
    row, eq = summarize("0D. 원본 FSM 1x (과열 2.0%/1.8%)", res_orig["strat_ret"], kofr_s, res_orig["turnover"])
    rows.append(row); equities["orig_fsm"] = eq

    res_cal = run_fsm_generalized(df, base_cfg, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
    row, eq = summarize("0E. 보정 FSM 1x (과열 2.4%/2.1%)", res_cal["strat_ret"], kofr_s, res_cal["turnover"])
    rows.append(row); equities["cal_fsm_1x"] = eq

    # --- Group 1: Path 1 — Leveraged KOSDAQ150 & Vol-Targeting ---
    p1_specs = [
        ("1A. [경로1] 보정 FSM + S1 1.5x (S2 0.5x, S4 0.3x)", {1: (0.5, 0.5, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)}, None, 2.0),
        ("1B. [경로1] 보정 FSM + S1 2.0x (S2 0.5x, S4 0.3x)", {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)}, None, 2.0),
        ("1C. [경로1] 보정 FSM + S1 2.0x / S2 1.0x / S4 0.6x", {1: (0, 1.0, 0, 0), 2: (1.0, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0, 0.3, 0, 0)}, None, 2.0),
        ("1D. [경로1] 보정 FSM + 변동성타겟 25% (최대 2.0x)", {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)}, 0.25, 2.0),
        ("1E. [경로1] 보정 FSM + 변동성타겟 35% (최대 2.0x)", {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0, 0.3, 0, 0)}, 0.35, 2.0),
    ]
    for name, allocs, vt, cap in p1_specs:
        r = run_fsm_generalized(df, base_cfg, allocs, vol_target_ann=vt, max_lev_cap=cap)
        row, eq = summarize(name, r["strat_ret"], kofr_s, r["turnover"])
        rows.append(row)
        equities[name] = eq

    # --- Group 2: Path 2 — Adding Inverse (251340) in State 3 ---
    for name, allocs in [
        ("2A. [경로2] 보정 FSM 1x + S3 인버스 50% (251340)", {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0.5, 0), 4: (0.3, 0, 0, 0)}),
        ("2B. [경로2] 보정 FSM S1 2.0x + S3 인버스 50%", {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0.5, 0), 4: (0.3, 0, 0, 0)}),
    ]:
        r = run_fsm_generalized(df, base_cfg, allocs)
        row, eq = summarize(name, r["strat_ret"], kofr_s, r["turnover"])
        rows.append(row)
        equities[name] = eq

    # --- Group 3: Path 3 — Multi-Asset Rotation (코스닥150 FSM + 나스닥100/KOSPI200/골드) ---
    df_alt = df.copy()
    df_alt["alt_ret"] = df_alt["nq_ret"]
    df_alt["alt_trend_on"] = df_alt["nq_trend_on"]

    # 3A: 1x only — when KOSDAQ is in State 3 (하락대피), invest in Nasdaq100 if Nasdaq100 > MA120, else KOFR!
    r3a = run_fsm_generalized(df_alt, base_cfg, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0.5), 3: (0, 0, 0, 1.0), 4: (0.3, 0, 0, 0.7)})
    row, eq = summarize("3A. [경로3] 노레버리지 1x: 코스닥 FSM + 여윳돈 나스닥100(MA120)", r3a["strat_ret"], kofr_s, r3a["turnover"])
    rows.append(row); equities["3A"] = eq

    # 3B: S1=1.5x + idle cash into Nasdaq100(MA120)
    r3b = run_fsm_generalized(df_alt, base_cfg, {1: (0.5, 0.5, 0, 0), 2: (0.5, 0, 0, 0.5), 3: (0, 0, 0, 1.0), 4: (0.3, 0, 0, 0.7)})
    row, eq = summarize("3B. [경로3] 저레버리지(S1 1.5x): 코스닥 FSM + 나스닥100(MA120)", r3b["strat_ret"], kofr_s, r3b["turnover"])
    rows.append(row); equities["3B"] = eq

    # 3C: 50:50 Dual Engine (50% 코스닥150 보정 FSM + 50% 나스닥100 MA120)
    nq_ma_ret, nq_ma_w, nq_ma_t = run_single_ma_trend(df["nq_close"], df["nq_ret"], df["kofr_ret"], ma_len=120, fee=0.0008)
    r3c = 0.5 * res_cal["strat_ret"] + 0.5 * nq_ma_ret
    t3c = 0.5 * res_cal["turnover"] + 0.5 * nq_ma_t
    row, eq = summarize("3C. [경로3] 노레버리지 50:50 듀얼엔진 (코스닥FSM + 나스닥MA120)", r3c, kofr_s, t3c)
    rows.append(row); equities["3C"] = eq

    # 3D: 50:50 Dual Engine with Gold MA120 safe sleeve (50% Gold if Gold>MA120 else KOFR)
    gold_ma_ret, _, _ = run_single_ma_trend(df["gold_close"], df["gold_ret"], df["kofr_ret"], ma_len=120, fee=0.0008)
    safe_blend = 0.5 * gold_ma_ret + 0.5 * df["kofr_ret"]
    nq_ma_gold_ret, _, nq_t2 = run_single_ma_trend(df["nq_close"], df["nq_ret"], df["kofr_ret"], ma_len=120, fee=0.0008, alt_safe_ret=safe_blend)
    # For 코스닥150 FSM cash portion, blend 50% gold_ma_ret
    r3d_kq = res_cal["strat_ret"] + (1.0 - res_cal["eff_exposure"].clip(0, 1)) * (safe_blend - df["kofr_ret"])
    r3d = 0.5 * r3d_kq + 0.5 * nq_ma_gold_ret
    row, eq = summarize("3D. [경로3] 50:50 듀얼엔진 + 하락장 골드/KOFR 방어(노레버리지)", r3d, kofr_s, t3c)
    rows.append(row); equities["3D"] = eq

    # 3E & 3F: Global Dual Momentum Rotation (KOSDAQ150 / KOSPI200 / Nasdaq100 / Gold, monthly)
    r3e, t3e = run_global_dual_momentum(df, fee=0.0008, top_k=1, mom_days=120)
    row, eq = summarize("3E. [경로3] 글로벌 듀얼모멘텀 Top-1 (4자산 월간회전, 1x)", r3e, kofr_s, t3e)
    rows.append(row); equities["3E"] = eq

    r3f, t3f = run_global_dual_momentum(df, fee=0.0008, top_k=2, mom_days=120)
    row, eq = summarize("3F. [경로3] 글로벌 듀얼모멘텀 Top-2 (4자산 월간회전, 1x)", r3f, kofr_s, t3f)
    rows.append(row); equities["3F"] = eq

    # 3G: 50% 코스닥 FSM(S1=2x) + 50% 나스닥100 MA120
    r_1b = run_fsm_generalized(df, base_cfg, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
    r3g = 0.5 * r_1b["strat_ret"] + 0.5 * nq_ma_ret
    t3g = 0.5 * r_1b["turnover"] + 0.5 * nq_ma_t
    row, eq = summarize("3G. [경로3] 50% 코스닥FSM(S1 2x) + 50% 나스닥MA120(1x)", r3g, kofr_s, t3g)
    rows.append(row); equities["3G"] = eq

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    out_df.to_csv(os.path.join(BASE_DIR, "paths_to_15pct_summary.csv"), index=False)

    # Save daily equities of top contenders for charting
    eq_df = pd.DataFrame({
        "코스닥150_1x_BH": equities["0A. 코스닥150 1x 단순보유 (229200)"],
        "코스닥150_2x_BH": equities["0B. 코스닥150 2x 레버리지 단순보유 (233740)"],
        "나스닥100_1x_BH": equities["0C. TIGER 미국나스닥100 단순보유 (133690)"],
        "보정FSM_1x": equities["cal_fsm_1x"],
        "경로1_FSM_S1_2x": equities["1B. [경로1] 보정 FSM + S1 2.0x (S2 0.5x, S4 0.3x)"],
        "경로1_FSM_VolTarget35": equities["1E. [경로1] 보정 FSM + 변동성타겟 35% (최대 2.0x)"],
        "경로3_FSM_여윳돈나스닥": equities["3A"],
        "경로3_50_50_S1_2x_나스닥": equities["3G"],
        "경로3_글로벌모멘텀Top1": equities["3E"],
    })
    eq_df.to_csv(os.path.join(BASE_DIR, "paths_to_15pct_equities.csv"))

    # Print year-by-year returns for Top 4 candidates hitting ~14~18%
    print("\n" + "=" * 110)
    print(" [핵심 후보 4종 연도별 수익률 비교 (2016~2026)]")
    print("=" * 110)
    yr_df = pd.DataFrame({
        "코스닥150_BH": df["etf_ret"],
        "나스닥100_BH": df["nq_ret"],
        "1B_코스닥S1_2x": r_1b["strat_ret"],
        "3A_노레버_여윳돈나스닥": r3a["strat_ret"],
        "3G_50코스닥2x_50나스닥": r3g,
        "3E_글로벌모멘텀Top1": r3e,
    })
    yr_rows = []
    for yr, grp in yr_df.groupby(yr_df.index.year):
        d = {"연도": yr}
        for col in grp.columns:
            val = ((1.0 + grp[col]).prod() - 1.0) * 100.0
            d[col] = f"{val:+.1f}%"
        yr_rows.append(d)
    print(pd.DataFrame(yr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
