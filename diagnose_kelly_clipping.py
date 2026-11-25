import os
import warnings
import numpy as np
import pandas as pd
from signal_domestic_2index import ETF_META, STRATEGY_CONFIG, reconstruct_total_return_nav, compute_ma_state

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    return {k: reconstruct_total_return_nav(pd.read_csv(os.path.join(DATA_DIR, m["csv"]), parse_dates=["date"])) for k, m in ETF_META.items()}


def run_diagnostic():
    data = load_data()
    idx = data["k200"].index.intersection(data["kq"].index).sort_values()
    k200_c = data["k200"]["adj_close"].loc[idx]
    kq_c = data["kq"]["adj_close"].loc[idx]
    r_k2 = data["k200"]["total_ret"].loc[idx]
    r_kq = data["kq"]["total_ret"].loc[idx]
    kofr_ret = data["kofr"]["total_ret"].reindex(idx).fillna(
        (1.0 + data["kq"]["cd91"].reindex(idx).ffill().bfill() / 100.0) ** (1.0 / 252.0) - 1.0
    )

    # 1. Diagnose current naive 120d-mu Kelly saturation
    m_k2 = k200_c.pct_change(120)
    m_kq = kq_c.pct_change(120)
    st_k2 = compute_ma_state(k200_c, 100, 3, 0.01)
    st_kq = compute_ma_state(kq_c, 100, 3, 0.01)

    mu120_k2 = (r_k2 - kofr_ret).rolling(120).mean() * 252.0
    vol60_k2 = r_k2.rolling(60).std() * np.sqrt(252.0)
    f_naive_k2 = (mu120_k2 / (vol60_k2 ** 2)).fillna(0.0)

    mu120_kq = (r_kq - kofr_ret).rolling(120).mean() * 252.0
    vol60_kq = r_kq.rolling(60).std() * np.sqrt(252.0)
    f_naive_kq = (mu120_kq / (vol60_kq ** 2)).fillna(0.0)

    bt_idx = idx[idx >= pd.Timestamp("2016-06-29")]
    is_me = np.zeros(len(bt_idx), dtype=bool); is_me[0] = True
    for i in range(len(bt_idx) - 1):
        if bt_idx[i].month != bt_idx[i + 1].month:
            is_me[i] = True

    # Collect raw f* on all equity-holding days of MA100 base strategy
    cur = "kofr"
    f_on_hold = []
    hold_log = []
    for i, d in enumerate(bt_idx):
        mk2, mkq = m_k2.loc[d], m_kq.loc[d]
        ok2, okq = int(st_k2.loc[d, "trend_on"]), int(st_kq.loc[d, "trend_on"])
        if is_me[i]:
            cands = []
            if mk2 > 0 and ok2 == 1: cands.append((mk2, "k200"))
            if mkq > 0 and okq == 1: cands.append((mkq, "kq"))
            cands.sort(reverse=True)
            cur = cands[0][1] if cands else "kofr"
        else:
            if cur == "k200" and ok2 == 0: cur = "kofr"
            elif cur == "kq" and okq == 0: cur = "kofr"
        if cur != "kofr":
            fv = float(f_naive_k2.loc[d] if cur == "k200" else f_naive_kq.loc[d])
            mv = float(mu120_k2.loc[d] if cur == "k200" else mu120_kq.loc[d])
            vv = float(vol60_k2.loc[d] if cur == "k200" else vol60_kq.loc[d])
            f_on_hold.append(fv)
            hold_log.append((d, cur, mv * 100, vv * 100, fv))

    f_s = pd.Series(f_on_hold)
    print("=" * 95)
    print(" [원인 진단] 기존 단기 120일 μ 기반 원시 켈리 f* 분포 (주식 보유일 전체 1,344일 중)")
    print("=" * 95)
    print(f"  중앙값 f* = {f_s.median():.2f} | 평균 f* = {f_s.mean():.2f} | 하위 25% = {f_s.quantile(0.25):.2f} | 상위 75% = {f_s.quantile(0.75):.2f}")
    print(f"  풀 켈리(1.0x)가 100% 상한에 걸린 비율 (f* >= 1.0): {(f_s >= 1.0).mean()*100:.1f}% ({(f_s >= 1.0).sum()}/{len(f_s)}일)")
    print(f"  하프 켈리(0.5x)가 100% 상한에 걸린 비율 (f* >= 2.0): {(f_s >= 2.0).mean()*100:.1f}% ({(f_s >= 2.0).sum()}/{len(f_s)}일)")
    print(f"  쿼터 켈리(0.25x)가 100% 상한에 걸린 비율 (f* >= 4.0): {(f_s >= 4.0).mean()*100:.1f}% ({(f_s >= 4.0).sum()}/{len(f_s)}일)")

    # Show 2026-05 ~ 2026-07 MDD episode!
    h_df = pd.DataFrame(hold_log, columns=["date", "asset", "mu120_pct", "vol60_pct", "raw_f"]).set_index("date")
    print("\n [2026-05 ~ 2026-07 MDD(-30.1%) 발생 구간 원시 f* 추이]")
    print(h_df.loc["2026-05-15":"2026-07-25"].iloc[::5].to_string())

    # ==========================================================================
    # 2. Test Proper Non-Saturated Kelly Formulations for 1x No-Leverage Account!
    #    Model A: Shrinkage Kelly (보수적 장기 기대초과수익률 μ_shrunk / σ_60^2)
    #             μ_shrunk = 0.5 * expanding_mu_of_asset + 0.5 * ERP_prior (또는 보수적 기대초과수익률 8%/yr 상한)
    #             Wait: Let's test BOTH:
    #             (A) Prior-Anchored / Expanding Kelly: mu_est = clip(expanding_excess_mu, 0.04, 0.12) / vol_60^2
    #                 (즉, 단기 거품 수익률 +150%를 미래 기대수익으로 착각하지 않고, 장기 현실적 기대초과수익률을 사용!)
    #             (B) Pure Expanding μ & Expanding/60d σ Kelly: mu_exp (전체 과거 상승장 평균초과수익) / vol_60^2
    #             (C) Fractional Scaled Kelly (f_norm = clip(f_exp, 0, 1) * kelly_frac) -> Half = 최대 50%? NO, let's check both!
    # ==========================================================================
    print("\n" + "=" * 115)
    print(" [개선 켈리 모델 비교] '장기 현실 기대수익률(축소추정 μ_shrink) / 최근 60일 분산(σ_60²)' 기반 연속 켈리 (MA100, 일반계좌 세후)")
    print("=" * 115)

    fee = STRATEGY_CONFIG["fee_oneway"]
    tax = STRATEGY_CONFIG["tax_rate"]
    r_k2_at = data["k200"].loc[bt_idx, "total_ret"] - tax * data["k200"].loc[bt_idx, "div_ret"]
    r_kq_at = data["kq"].loc[bt_idx, "total_ret"] - tax * data["kq"].loc[bt_idx, "div_ret"]
    r_kofr_at = kofr_ret.loc[bt_idx] * (1.0 - tax)

    # Expanding historical excess return (strictly causal, min 252 days) shrunk with 6% ERP prior
    # μ_est = 0.5 * clip(rolling_750d_excess_mu, 0.0, 0.15) + 0.5 * 0.07  -> realistically 3.5% ~ 11.0% (no +150% bubble illusion!)
    def make_shrunk_mu(r_s, rf_s, prior=0.07, cap=0.14):
        long_mu = (r_s - rf_s).expanding(min_periods=120).mean() * 252.0
        # Also blend 30% of clipped 120d trend edge (capped at cap) so momentum strength still matters mildly
        short_mu = ((r_s - rf_s).rolling(120).mean() * 252.0).clip(0.0, cap)
        return (0.5 * prior + 0.25 * long_mu.clip(0.0, cap) + 0.25 * short_mu).fillna(prior)

    mu_s_k2 = make_shrunk_mu(r_k2, kofr_ret)
    mu_s_kq = make_shrunk_mu(r_kq, kofr_ret)
    f_shrunk_k2 = (mu_s_k2 / (vol60_k2 ** 2)).fillna(0.0)
    f_shrunk_kq = (mu_s_kq / (vol60_kq ** 2)).fillna(0.0)

    # Check distribution of shrunk f* on equity-holding days!
    fs_hold = pd.Series([float(f_shrunk_k2.loc[d] if a == "k200" else f_shrunk_kq.loc[d]) for d, a, _, _, _ in hold_log])
    print(f"  [축소추정 켈리 f* 분포] 중앙값={fs_hold.median():.2f} | 하위25%={fs_hold.quantile(0.25):.2f} | 상위75%={fs_hold.quantile(0.75):.2f}")
    print(f"  -> 풀켈리(1.0x) 100%포화율: {(fs_hold >= 1.0).mean()*100:.1f}% | 하프켈리(0.5x) 100%포화율: {(fs_hold >= 2.0).mean()*100:.1f}% | 쿼터켈리(0.25x) 100%포화율: {(fs_hold >= 4.0).mean()*100:.1f}%\n")

    def eval_kelly_model(label, f_k2_s, f_kq_s, k_frac, ma_len=100, use_short_vol=False):
        st2 = compute_ma_state(k200_c, ma_len, 3, 0.01)
        stq = compute_ma_state(kq_c, ma_len, 3, 0.01)
        cur_a, cur_w = "kofr", 0.0
        eq = np.zeros(len(bt_idx)); eq[0] = 10_000_000.0
        rets = np.zeros(len(bt_idx))
        ws = np.zeros(len(bt_idx))
        turns = 0.0
        for i, d in enumerate(bt_idx):
            if i > 0:
                if cur_a == "k200": gross = cur_w * float(r_k2_at.iloc[i]) + (1 - cur_w) * float(r_kofr_at.iloc[i])
                elif cur_a == "kq": gross = cur_w * float(r_kq_at.iloc[i]) + (1 - cur_w) * float(r_kofr_at.iloc[i])
                else: gross = float(r_kofr_at.iloc[i])
            else:
                gross = 0.0
            mk2, mkq = m_k2.loc[d], m_kq.loc[d]
            ok2, okq = int(st2.loc[d, "trend_on"]), int(stq.loc[d, "trend_on"])
            pa, pw = cur_a, cur_w
            if is_me[i]:
                cands = []
                if mk2 > 0 and ok2 == 1: cands.append((mk2, "k200"))
                if mkq > 0 and okq == 1: cands.append((mkq, "kq"))
                cands.sort(reverse=True)
                na = cands[0][1] if cands else "kofr"
            else:
                na = cur_a
                if cur_a == "k200" and ok2 == 0: na = "kofr"
                elif cur_a == "kq" and okq == 0: na = "kofr"

            if na == "kofr":
                nw = 0.0
            elif k_frac is None:
                nw = 1.0
            else:
                fv = float(f_k2_s.loc[d] if na == "k200" else f_kq_s.loc[d])
                tw = float(np.clip(fv * k_frac, 0.0, 1.0))
                if na == pa and not is_me[i] and abs(tw - pw) < 0.10:
                    nw = pw
                else:
                    nw = round(tw * 20) / 20.0
            t = 0.0 if i == 0 else (abs(nw - pw) if na == pa else 0.5 * (pw + nw + abs((1 - nw) - (1 - pw))))
            turns += t
            rets[i] = gross - t * fee
            if i > 0: eq[i] = eq[i - 1] * (1 + rets[i])
            ws[i] = nw
            cur_a, cur_w = na, nw

        eq_s = pd.Series(eq, index=bt_idx)
        ret_s = pd.Series(rets, index=bt_idx)
        yrs = (bt_idx[-1] - bt_idx[0]).days / 365.25
        cagr = (eq_s.iloc[-1] / eq_s.iloc[0]) ** (1 / yrs) - 1
        mdd = (eq_s / eq_s.cummax() - 1).min()
        vol = ret_s.iloc[1:].std() * np.sqrt(252)
        ex = ret_s.iloc[1:] - r_kofr_at.iloc[1:]
        sh = (ex.mean() * 252) / (ex.std() * np.sqrt(252))
        cal = cagr / abs(mdd)
        r_p25 = ret_s.loc[:"2024-12-31"]
        ep25 = (1 + r_p25).cumprod()
        cp25 = ep25.iloc[-1] ** (1 / ((r_p25.index[-1] - r_p25.index[0]).days / 365.25)) - 1
        mp25 = (ep25 / ep25.cummax() - 1).min()
        w_active = ws[ws > 0]
        return {
            "모델": label,
            "세후CAGR": f"{cagr*100:+.2f}%",
            "MDD": f"{mdd*100:.1f}%",
            "Vol": f"{vol*100:.1f}%",
            "초과샤프": f"{sh:.2f}",
            "칼마": f"{cal:.2f}",
            "16~24(8.5년)": f"{cp25*100:+.1f}%/{mp25*100:.0f}%",
            "보유시평균비중": f"{w_active.mean()*100:.0f}%" if len(w_active) else "0%",
            "전체평균비중": f"{ws.mean()*100:.1f}%",
            "연회전": f"{turns/yrs:.1f}",
        }

    # Also test 20-day fast vol (reacts immediately when 2026 bubble vol spikes!)
    vol20_k2 = r_k2.rolling(20).std() * np.sqrt(252.0)
    vol20_kq = r_kq.rolling(20).std() * np.sqrt(252.0)
    # Blend 20d & 60d max vol (conservative variance = max(var20, var60))
    vcons_k2 = np.maximum(vol20_k2, vol60_k2)
    vcons_kq = np.maximum(vol20_kq, vol60_kq)
    f_cons_k2 = (mu_s_k2 / (vcons_k2 ** 2)).fillna(0.0)
    f_cons_kq = (mu_s_kq / (vcons_kq ** 2)).fillna(0.0)

    res_rows = [
        eval_kelly_model("MA100 고정 100%", f_shrunk_k2, f_shrunk_kq, None, 100),
        eval_kelly_model("MA100 축소켈리 풀(1.0x, σ60)", f_shrunk_k2, f_shrunk_kq, 1.0, 100),
        eval_kelly_model("MA100 축소켈리 하프(0.5x, σ60)", f_shrunk_k2, f_shrunk_kq, 0.5, 100),
        eval_kelly_model("MA100 축소켈리 쿼터(0.25x, σ60)", f_shrunk_k2, f_shrunk_kq, 0.25, 100),
        eval_kelly_model("MA100 축소켈리 풀(1.0x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 1.0, 100),
        eval_kelly_model("MA100 축소켈리 하프(0.5x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 0.5, 100),
        eval_kelly_model("MA100 축소켈리 쿼터(0.25x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 0.25, 100),
        eval_kelly_model("MA80  축소켈리 풀(1.0x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 1.0, 80),
        eval_kelly_model("MA80  축소켈리 하프(0.5x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 0.5, 80),
        eval_kelly_model("MA60  축소켈리 풀(1.0x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 1.0, 60),
        eval_kelly_model("MA60  축소켈리 하프(0.5x, max(σ20,σ60))", f_cons_k2, f_cons_kq, 0.5, 60),
    ]
    print(pd.DataFrame(res_rows).to_string(index=False))


if __name__ == "__main__":
    run_diagnostic()
