import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, load_market_data, calc_metrics
from fsm_engine_v2 import load_aligned_etf, run_fsm_generalized
from test_15pct_strategies import run_single_ma_trend, run_global_dual_momentum

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}
    df0 = load_market_data(base_cfg)
    lev2x = load_aligned_etf("kodex_kosdaq150_lev2x_nav.csv", "lev2x", "2016-01-01", "2026-11-24")
    nq = load_aligned_etf("tiger_nasdaq100_nav.csv", "nq", "2015-01-01", "2026-11-24")
    k200 = load_aligned_etf("kodex_200_nav.csv", "k200", "2015-01-01", "2026-11-24")
    gold = load_aligned_etf("kodex_gold_nav.csv", "gold", "2015-01-01", "2026-11-24")

    df = df0.join([lev2x, nq, k200, gold], how="inner").copy()
    kofr_s = df["kofr_ret"].copy()
    kofr_s.iloc[0] = 0.0

    # 1. Raw Data Integrity Check on all 4 assets (max daily return, annual returns)
    print("=== [1] 자산별 원본 데이터 무결성 점검 (일일 최대/최소 수익률 & 연도별 수익률) ===")
    for col, lbl in [("etf_ret", "코스닥150"), ("lev2x_ret", "코스닥150_2x"), ("k200_ret", "KODEX200"), ("nq_ret", "나스닥100"), ("gold_ret", "골드선물(H)")]:
        s = df[col]
        print(f"  {lbl:10s}: min_1d={s.min()*100:+6.2f}% ({s.idxmin().date()}), max_1d={s.max()*100:+6.2f}% ({s.idxmax().date()}), "
              f"10y_CAGR={((1+s.iloc[1:]).prod()**(365.25/(df.index[-1]-df.index[0]).days)-1)*100:+6.2f}%")

    yr_a = []
    for yr, grp in df.groupby(df.index.year):
        yr_a.append({
            "연도": yr,
            "코스닥150": f"{((1+grp['etf_ret']).prod()-1)*100:+.1f}%",
            "KODEX200": f"{((1+grp['k200_ret']).prod()-1)*100:+.1f}%",
            "나스닥100": f"{((1+grp['nq_ret']).prod()-1)*100:+.1f}%",
            "골드(H)": f"{((1+grp['gold_ret']).prod()-1)*100:+.1f}%",
        })
    print(pd.DataFrame(yr_a).to_string(index=False))

    # 2. Check 3F (Global Dual Momentum Top-2) robustness across mom_days & fee & top_k
    print("\n=== [2] 3F(글로벌 듀얼모멘텀 Top-2, 노레버리지 1x) 모멘텀 기간 민감도 ===")
    for md in [60, 90, 120, 180, 240]:
        r, t = run_global_dual_momentum(df, fee=0.0008, top_k=2, mom_days=md)
        eq = 10_000_000 * (1 + r).cumprod()
        m = calc_metrics(eq, r, kofr_s)
        r1, r2 = r.loc[:"2021-12-31"], r.loc["2022-01-01":]
        m1 = calc_metrics((1 + r1).cumprod(), r1, kofr_s.loc[:"2021-12-31"])
        m2 = calc_metrics((1 + r2).cumprod(), r2, kofr_s.loc["2022-01-01":])
        print(f"  mom={md:3d}일 | CAGR={m['cagr_pct']:+6.2f}% | MDD={m['mdd_pct']:6.2f}% | ExSharpe={m['sharpe_excess']:.2f} | Calmar={m['calmar']:.2f} | 전반기={m1['cagr_pct']:+5.1f}% | 후반기={m2['cagr_pct']:+5.1f}%")

    # 3. Check 3G (50% 코스닥 FSM S1=2x + 50% 나스닥100 MA120) sensitivity across MA & credit threshold & General Tax
    print("\n=== [3] 3G(50% 코스닥FSM S1 2x + 50% 나스닥MA) 파라미터 민감도 & 계좌 세금 비교 ===")
    for ma in [100, 120, 150]:
        for cr_ov in [2.2, 2.4, 2.6]:
            cfg_t = {**base_cfg, "ma_long": ma, "cr_overheat_pct": cr_ov, "cr_calm_pct": round(cr_ov - 0.3, 2)}
            df_t = load_market_data(cfg_t).join([lev2x, nq], how="inner")
            r_kq = run_fsm_generalized(df_t, cfg_t, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
            r_nq, _, _ = run_single_ma_trend(df_t["nq_close"], df_t["nq_ret"], df_t["kofr_ret"], ma_len=ma, fee=0.0008)
            r_tot = 0.5 * r_kq["strat_ret"] + 0.5 * r_nq
            ks = df_t["kofr_ret"].copy(); ks.iloc[0] = 0.0
            m = calc_metrics(10_000_000 * (1 + r_tot).cumprod(), r_tot, ks)
            print(f"  MA{ma:3d}, cr={cr_ov:.1f}% | CAGR={m['cagr_pct']:+6.2f}% | MDD={m['mdd_pct']:6.2f}% | ExSharpe={m['sharpe_excess']:.2f} | Calmar={m['calmar']:.2f}")

    # 4. General Account (15.4% tax on 2x/Nasdaq/Gold/KOFR) vs ISA comparison for top 4
    print("\n=== [4] 절세 계좌(ISA/연금저축) vs 일반계좌(15.4% 과세 근사) 비교 ===")
    # 3G in ISA vs General
    cfg_gen = {**base_cfg, "account_type": "GENERAL"}
    r_1b_gen = run_fsm_generalized(df, cfg_gen, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)}, tax_derivatives=True)
    # Approximate 15.4% annual tax drag on taxable ETFs (conservatively applying 15.4% on positive annual gains)
    def after_tax_annual(ret_s: pd.Series, tax_rate: float = 0.154):
        out = ret_s.copy()
        for yr, grp in ret_s.groupby(ret_s.index.year):
            yr_ret = (1.0 + grp).prod() - 1.0
            if yr_ret > 0:
                # deduct tax at year end
                tax_drag = yr_ret * tax_rate / (1.0 + yr_ret)
                out.loc[grp.index[-1]] = (1.0 + out.loc[grp.index[-1]]) * (1.0 - tax_drag) - 1.0
        return out

    r_3f, _ = run_global_dual_momentum(df, fee=0.0008, top_k=2, mom_days=120)
    r_3f_tax = after_tax_annual(r_3f, 0.154)
    r_nq_ma, _, _ = run_single_ma_trend(df["nq_close"], df["nq_ret"], df["kofr_ret"], ma_len=120, fee=0.0008)
    r_1b_isa = run_fsm_generalized(df, base_cfg, {1: (0, 1.0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
    r_3g_isa = 0.5 * r_1b_isa["strat_ret"] + 0.5 * r_nq_ma
    r_3g_tax = after_tax_annual(r_3g_isa, 0.154)

    for lbl, r_isa, r_tx in [
        ("1B. 코스닥150 FSM (S1 2x)", r_1b_isa["strat_ret"], after_tax_annual(r_1b_isa["strat_ret"], 0.154)),
        ("3G. 50%코스닥FSM(S1 2x)+50%나스닥MA120", r_3g_isa, r_3g_tax),
        ("3F. 글로벌 듀얼모멘텀 Top-2 (1x)", r_3f, r_3f_tax),
    ]:
        m_i = calc_metrics(10_000_000 * (1 + r_isa).cumprod(), r_isa, kofr_s)
        m_t = calc_metrics(10_000_000 * (1 + r_tx).cumprod(), r_tx, kofr_s)
        print(f"  {lbl:38s} | 절세계좌 CAGR={m_i['cagr_pct']:+6.2f}% (MDD {m_i['mdd_pct']:5.1f}%) -> 일반과세 CAGR={m_t['cagr_pct']:+6.2f}% (세금차이 {m_t['cagr_pct']-m_i['cagr_pct']:+.2f}%p/yr)")

    # 5. Also check 3F with hybrid 50% (코스닥 FSM 1x) + 50% (글로벌 듀얼모멘텀 Top-2) -> 100% 노레버리지 15% 조합!
    r_cal = run_fsm_generalized(df, base_cfg, {1: (1, 0, 0, 0), 2: (0.5, 0, 0, 0), 3: (0, 0, 0, 0), 4: (0.3, 0, 0, 0)})
    r_hyb = 0.4 * r_cal["strat_ret"] + 0.6 * r_3f
    m_h = calc_metrics(10_000_000 * (1 + r_hyb).cumprod(), r_hyb, kofr_s)
    r1, r2 = r_hyb.loc[:"2021-12-31"], r_hyb.loc["2022-01-01":]
    m1 = calc_metrics((1 + r1).cumprod(), r1, kofr_s.loc[:"2021-12-31"])
    m2 = calc_metrics((1 + r2).cumprod(), r2, kofr_s.loc["2022-01-01":])
    print(f"\n=== [5] 노레버리지 1x 15% 하이브리드 (40% 코스닥FSM 1x + 60% 글로벌모멘텀 Top-2) ===")
    print(f"  CAGR={m_h['cagr_pct']:+6.2f}% | Vol={m_h['ann_vol_pct']:.1f}% | MDD={m_h['mdd_pct']:6.2f}% | ExSharpe={m_h['sharpe_excess']:.2f} | Calmar={m_h['calmar']:.2f} | 전반기={m1['cagr_pct']:+5.1f}%/{m1['mdd_pct']:.0f}% | 후반기={m2['cagr_pct']:+5.1f}%/{m2['mdd_pct']:.0f}%")


if __name__ == "__main__":
    main()
