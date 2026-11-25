import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, run_backtest, calc_metrics

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def eval_variant(cfg: dict, base_df: pd.DataFrame = None, fixed_bull_weight: float = None):
    df = load_market_data(cfg) if base_df is None else base_df
    if fixed_bull_weight is not None:
        # Null hypothesis: Pure MA120 with fixed bull weight (e.g., 1.0 or 0.5), NO credit data
        c = {**cfg, "enable_state2": False, "enable_state4": False}
        res, s4 = run_backtest(df, c)
        # Re-simulate with fixed_bull_weight on State 1
        w = res["weight"].values * fixed_bull_weight
        rets = np.zeros(len(res))
        turns = np.zeros(len(res))
        kofr_tax = 0.154 if c["account_type"].upper() == "GENERAL" else 0.0
        for i in range(len(res)):
            if i > 0:
                eq_r = res["etf_ret"].iloc[i]
                kr = res["kofr_ret"].iloc[i]
                cash_r = kr * (1.0 - kofr_tax if kr > 0 else 1.0)
                gross = w[i - 1] * eq_r + (1.0 - w[i - 1]) * cash_r
                t = abs(w[i] - w[i - 1])
            else:
                gross = 0.0
                t = 0.0
            rets[i] = gross - t * c["fee_slippage_oneway"]
            turns[i] = t
        eq = c["initial_capital"] * pd.Series(1.0 + rets, index=res.index).cumprod()
        m = calc_metrics(eq, pd.Series(rets, index=res.index), res["kofr_after_tax_ret"])
        years = (res.index[-1] - res.index[0]).days / 365.25
        m["ann_turnover"] = turns.sum() / years
        m["avg_weight"] = w.mean() * 100.0
        m["s1_pct"] = (res["state"] == 1).mean() * 100.0
        m["s2_pct"] = 0.0
        m["s3_pct"] = (res["state"] == 3).mean() * 100.0
        m["s4_pct"] = 0.0
        m["s4_cnt"] = 0
        res["strat_ret"] = rets
        res["strat_equity"] = eq
        res["weight"] = w
        return m, res, s4

    res, s4 = run_backtest(df, cfg)
    m = calc_metrics(res["strat_equity"], res["strat_ret"], res["kofr_after_tax_ret"])
    years = (res.index[-1] - res.index[0]).days / 365.25
    m["ann_turnover"] = res["turnover"].sum() / years
    m["avg_weight"] = res["weight"].mean() * 100.0
    m["s1_pct"] = (res["state"] == 1).mean() * 100.0
    m["s2_pct"] = (res["state"] == 2).mean() * 100.0
    m["s3_pct"] = (res["state"] == 3).mean() * 100.0
    m["s4_pct"] = (res["state"] == 4).mean() * 100.0
    m["s4_cnt"] = len(s4)
    return m, res, s4


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True}
    base_df = load_market_data(base_cfg)

    # ==========================================================================
    # 1. ABLATION STUDY (요인 분해 및 귀무가설 검정)
    # ==========================================================================
    print("=" * 105)
    print(" [1] 전략 요인 분해 (Ablation Study) — 실제 데이터 + 현실 CD91/KOFR 금리 (ISA)")
    print("=" * 105)
    variants = [
        ("A. 코스닥150 단순보유 (B&H 100%)", None, None),
        ("B. 단순 MA120 (100% / 0%, 신용 미사용)", {**base_cfg}, 1.0),
        ("C. 단순 MA120 (50% / 0%, 신용 미사용 — 귀무가설)", {**base_cfg}, 0.5),
        ("D. MA120 + State2만 (과열2.0%/진정1.8%, S4 OFF)", {**base_cfg, "enable_state2": True, "enable_state4": False}, None),
        ("E. MA120 + State4만 (투매역발상, S2 OFF)", {**base_cfg, "enable_state2": False, "enable_state4": True}, None),
        ("F. 원본 4단계 FSM (과열2.0%/진정1.8%)", {**base_cfg}, None),
        ("G. 현실화 4단계 FSM (과열2.4%/진정2.1%)", {**base_cfg, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}, None),
        ("H. 현실화 4단계 FSM (과열2.5%/진정2.2%)", {**base_cfg, "cr_overheat_pct": 2.5, "cr_calm_pct": 2.2}, None),
        ("I. 동적 4단계 FSM (롤링 1년 상위20% & 20일+10%)", {**base_cfg, "use_dynamic_cr": True}, None),
    ]

    ab_rows = []
    for name, cfg, fw in variants:
        if cfg is None:
            m = calc_metrics(base_df["etf_close"] / base_df["etf_close"].iloc[0] * 10_000_000,
                             base_df["etf_ret"].where(np.arange(len(base_df)) > 0, 0.0),
                             base_df["kofr_ret"].where(np.arange(len(base_df)) > 0, 0.0))
            ab_rows.append({
                "전략": name,
                "CAGR(%)": f"{m['cagr_pct']:+.2f}",
                "Vol(%)": f"{m['ann_vol_pct']:.2f}",
                "Sharpe": f"{m['sharpe_rf0']:.2f}",
                "ExSharpe": f"{m['sharpe_excess']:.2f}",
                "MDD(%)": f"{m['mdd_pct']:.2f}",
                "Calmar": f"{m['calmar']:.2f}",
                "평균비중": "100.0%",
                "연회전": "0.00",
                "S1/S2/S3/S4(%)": "100/0/0/0",
            })
        else:
            m, _, _ = eval_variant(cfg, base_df=base_df, fixed_bull_weight=fw)
            ab_rows.append({
                "전략": name,
                "CAGR(%)": f"{m['cagr_pct']:+.2f}",
                "Vol(%)": f"{m['ann_vol_pct']:.2f}",
                "Sharpe": f"{m['sharpe_rf0']:.2f}",
                "ExSharpe": f"{m['sharpe_excess']:.2f}",
                "MDD(%)": f"{m['mdd_pct']:.2f}",
                "Calmar": f"{m['calmar']:.2f}",
                "평균비중": f"{m['avg_weight']:.1f}%",
                "연회전": f"{m['ann_turnover']:.2f}",
                "S1/S2/S3/S4(%)": f"{m['s1_pct']:.0f}/{m['s2_pct']:.0f}/{m['s3_pct']:.0f}/{m['s4_pct']:.0f}",
            })
    ab_df = pd.DataFrame(ab_rows)
    print(ab_df.to_string(index=False))

    # ==========================================================================
    # 2. PARAMETER SENSITIVITY GRID: ma_long x cr_overheat_pct
    # ==========================================================================
    print("\n" + "=" * 105)
    print(" [2] 파라미터 민감도 분석 (과최적화 점검): ma_long × cr_overheat_pct (cr_calm = overheat - 0.2%p)")
    print("=" * 105)
    ma_list = [60, 80, 100, 120, 150, 180, 200]
    cr_list = [1.8, 2.0, 2.2, 2.4, 2.6, 99.0]  # 99.0 = cr_ratio 기준 사실상 비활성 (20d +10%만 작동)

    grid_cagr = pd.DataFrame(index=[f"MA{m}" for m in ma_list], columns=[f"{c:.1f}%" if c < 50 else "20d급증만" for c in cr_list])
    grid_mdd = grid_cagr.copy()
    grid_calmar = grid_cagr.copy()
    grid_sharpe = grid_cagr.copy()

    # 동일 평가 기간(MA200 기준 시작일)으로 공정 비교!
    df_ma200 = load_market_data({**base_cfg, "ma_long": 200})
    common_start = df_ma200.index[0]
    print(f" (공정 비교를 위해 모든 MA 조합을 동일 구간 {common_start.strftime('%Y-%m-%d')} ~ {df_ma200.index[-1].strftime('%Y-%m-%d')}에서 평가)")

    grid_records = []
    for ma in ma_list:
        df_ma = load_market_data({**base_cfg, "ma_long": ma})
        df_ma = df_ma.loc[common_start:].copy()
        for cr in cr_list:
            col = f"{cr:.1f}%" if cr < 50 else "20d급증만"
            cfg = {**base_cfg, "ma_long": ma, "cr_overheat_pct": cr, "cr_calm_pct": max(cr - 0.2, 1.5)}
            m, _, _ = eval_variant(cfg, base_df=df_ma)
            grid_cagr.loc[f"MA{ma}", col] = round(m["cagr_pct"], 2)
            grid_mdd.loc[f"MA{ma}", col] = round(m["mdd_pct"], 2)
            grid_calmar.loc[f"MA{ma}", col] = round(m["calmar"], 2)
            grid_sharpe.loc[f"MA{ma}", col] = round(m["sharpe_excess"], 2)
            grid_records.append({
                "ma_long": ma,
                "cr_overheat": col,
                "cagr": m["cagr_pct"],
                "mdd": m["mdd_pct"],
                "calmar": m["calmar"],
                "sharpe_ex": m["sharpe_excess"],
            })

    print("\n--- CAGR (%) ---")
    print(grid_cagr.to_string())
    print("\n--- MDD (%) ---")
    print(grid_mdd.to_string())
    print("\n--- Calmar Ratio (CAGR / |MDD|) ---")
    print(grid_calmar.to_string())
    print("\n--- Excess Sharpe (vs KOFR/CD91) ---")
    print(grid_sharpe.to_string())

    pd.DataFrame(grid_records).to_csv(os.path.join(BASE_DIR, "sensitivity_grid.csv"), index=False)

    # ==========================================================================
    # 3. SUB-PERIOD & WALK-FORWARD / OUT-OF-SAMPLE CHECK
    # ==========================================================================
    print("\n" + "=" * 105)
    print(" [3] 국면별·구간별 성과 검증 (전반기 vs 후반기 / KOFR 상장 이후 실전 구간)")
    print("=" * 105)
    _, res_orig, _ = eval_variant(base_cfg, base_df=base_df)
    _, res_ma100, _ = eval_variant(base_cfg, base_df=base_df, fixed_bull_weight=1.0)
    _, res_ma50, _ = eval_variant(base_cfg, base_df=base_df, fixed_bull_weight=0.5)
    _, res_cal24, _ = eval_variant({**base_cfg, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}, base_df=base_df)

    periods = [
        ("전반기 (2016-06 ~ 2021-12)", "2016-06-29", "2021-12-31"),
        ("후반기 (2022-01 ~ 2026-11)", "2022-01-01", "2026-11-24"),
        ("KOFR 실상장 이후 (2022-05 ~ 2026-11)", "2022-05-01", "2026-11-24"),
        ("코로나 급락/급등장 (2020-01 ~ 2021-12)", "2020-01-01", "2021-12-31"),
        ("고금리·하락/박스장 (2022-01 ~ 2024-12)", "2022-01-01", "2024-12-31"),
        ("최근 2년 (2025-01 ~ 2026-11)", "2025-01-01", "2026-11-24"),
    ]
    sub_rows = []
    for label, s_dt, e_dt in periods:
        row = {"구간": label}
        for tag, r_df, col_r in [
            ("원본FSM(2.0%)", res_orig, "strat_ret"),
            ("보정FSM(2.4%)", res_cal24, "strat_ret"),
            ("단순MA120(100%)", res_ma100, "strat_ret"),
            ("단순MA120(50%)", res_ma50, "strat_ret"),
            ("코스닥150 B&H", res_orig, "bh_ret"),
        ]:
            sub = r_df.loc[s_dt:e_dt]
            eq = (1.0 + sub[col_r]).cumprod()
            m = calc_metrics(eq, sub[col_r], sub["kofr_after_tax_ret"])
            row[tag] = f"{m['cagr_pct']:+.1f}% / {m['mdd_pct']:.1f}%"
        sub_rows.append(row)
    print(" (표기: CAGR / MDD)")
    print(pd.DataFrame(sub_rows).to_string(index=False))

    # ==========================================================================
    # 4. COST & CREDIT LAG STRESS TEST
    # ==========================================================================
    print("\n" + "=" * 105)
    print(" [4] 거래비용(수수료+슬리피지) 및 신용잔고 시차(Lag) 스트레스 테스트")
    print("=" * 105)
    stress_rows = []
    for fee_bp in [0, 8, 15, 30]:
        cfg = {**base_cfg, "fee_slippage_oneway": fee_bp / 10000.0}
        m, _, _ = eval_variant(cfg, base_df=base_df)
        stress_rows.append({
            "테스트": f"편도 비용 {fee_bp:2d}bp ({fee_bp/100:.2f}%)",
            "CAGR(%)": f"{m['cagr_pct']:+.2f}%",
            "MDD(%)": f"{m['mdd_pct']:.2f}%",
            "초과샤프": f"{m['sharpe_excess']:.2f}",
            "칼마비율": f"{m['calmar']:.2f}",
        })
    for lag in [1, 2, 3, 5]:
        cfg = {**base_cfg, "credit_lag_days": lag}
        m, _, _ = eval_variant(cfg)
        stress_rows.append({
            "테스트": f"신용잔고 시차 shift({lag})",
            "CAGR(%)": f"{m['cagr_pct']:+.2f}%",
            "MDD(%)": f"{m['mdd_pct']:.2f}%",
            "초과샤프": f"{m['sharpe_excess']:.2f}",
            "칼마비율": f"{m['calmar']:.2f}",
        })
    print(pd.DataFrame(stress_rows).to_string(index=False))


if __name__ == "__main__":
    main()
