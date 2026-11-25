import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, _reconstruct_total_return_nav, load_market_data, calc_metrics

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_etf_tax_aware(csv_name: str, is_domestic_equity: bool) -> pd.DataFrame:
    """
    KOFIA DIS 공식 데이터에서 가격수익(nav_close)과 총수익(adj_close)을 분리 복원.
    국내주식형 1x ETF는 매매차익 비과세 / 분배금만 15.4% 과세.
    """
    raw = pd.read_csv(os.path.join(BASE_DIR, csv_name), parse_dates=["date"])
    al = _reconstruct_total_return_nav(raw).set_index("date")
    al["price_ret"] = al["nav_close"].pct_change().fillna(0.0)
    al["total_ret"] = al["adj_close"].pct_change().fillna(0.0)
    # 분배금 수익률 (배당락일 양수)
    al["div_ret"] = np.maximum(0.0, al["total_ret"] - al["price_ret"])
    if is_domestic_equity:
        # 일반계좌: 매매차익 0% 비과세, 분배금만 15.4% 원천징수
        al["dom_after_tax_ret"] = al["total_ret"] - 0.154 * al["div_ret"]
    else:
        al["dom_after_tax_ret"] = al["total_ret"]
    return al


def simulate_general_account_portfolio(
    dates: pd.DatetimeIndex,
    target_weights: pd.DataFrame,
    price_rets: dict,
    total_rets: dict,
    div_rets: dict,
    is_taxable_capgain: dict,
    kofr_ret: pd.Series,
    initial_capital: float = 10_000_000.0,
    fee_oneway: float = 0.0008,
    tax_rate: float = 0.154,
):
    """
    대한민국 일반계좌 실제 세법 반영 시뮬레이터:
    - 국내주식형 1x ETF (is_taxable_capgain=False): 매매차익 비과세(0%), 분배금만 15.4% 과세
    - 해외/금리/원자재 ETF (is_taxable_capgain=True): 분배금 15.4% + 매도 시 실현차익(평균단가 기준) 15.4% 과세
    - 백테스트 종료 시점 미실현 과세차익도 세후 청산가치(liquidation_equity)로 엄격하게 반영
    - 미래 참조 방지: 전일(t-1) 종가 기준 확정된 target_weights로 당일(t) 수익 실현
    """
    assets = list(target_weights.columns)
    n = len(dates)

    # 보유 평가액 및 과세 취득원가(평균단가법)
    vals = {a: 0.0 for a in assets}
    costs = {a: 0.0 for a in assets}
    cash_val = initial_capital
    cash_cost = initial_capital  # KOFR 보유 취득원가

    eq_curve = np.zeros(n)
    eq_after_liq = np.zeros(n)
    daily_rets = np.zeros(n)
    turnovers = np.zeros(n)
    total_tax_paid = 0.0
    total_fee_paid = 0.0

    for i in range(n):
        # 1. 전일 보유 포지션에 오늘(i) 수익률 반영
        if i > 0:
            for a in assets:
                if vals[a] > 0:
                    pr = price_rets[a].iloc[i]
                    tr = total_rets[a].iloc[i]
                    dr = div_rets[a].iloc[i]
                    # 분배금 세후 재투자 + 가격 변동
                    if not is_taxable_capgain[a]:
                        # 국내주식형: 매매차익 비과세, 분배금만 15.4%
                        tax_d = vals[a] * dr * tax_rate
                        total_tax_paid += tax_d
                        vals[a] = vals[a] * (1.0 + tr) - tax_d
                    else:
                        # 과세 ETF: 분배금 15.4% 과세 (세후 재투자분은 취득원가에 가산)
                        tax_d = vals[a] * dr * tax_rate
                        net_div = vals[a] * dr * (1.0 - tax_rate)
                        total_tax_paid += tax_d
                        vals[a] = vals[a] * (1.0 + pr) + net_div
                        costs[a] += net_div

            # KOFR 현금 수익 (보유기간 과세: 일별 이자 발생, 매도 시 정산 근사 -> 일복리 평가)
            kr = kofr_ret.iloc[i]
            cash_val = cash_val * (1.0 + kr)

        total_val = sum(vals.values()) + cash_val

        # 2. 오늘 종가 기준 목표 비중으로 리밸런싱 (세금 & 거래비용 차감)
        tw = target_weights.iloc[i]
        tw_cash = max(0.0, 1.0 - float(tw.sum()))

        # 먼저 매도해야 할 자산 처리 (세금 발생)
        tax_today = 0.0
        traded_notional = 0.0

        # 목표 금액 계산 (세전 총자산 기준 1차 계산 후 세금/수수료는 현금에서 차감)
        desired = {a: total_val * float(tw[a]) for a in assets}
        desired_cash = total_val * tw_cash

        for a in assets:
            diff_amt = desired[a] - vals[a]
            if diff_amt < -1.0:  # 매도
                sell_amt = -diff_amt
                frac = min(1.0, sell_amt / vals[a]) if vals[a] > 0 else 0.0
                if is_taxable_capgain[a]:
                    realized_gain = max(0.0, (vals[a] - costs[a]) * frac)
                    tax_today += realized_gain * tax_rate
                costs[a] *= (1.0 - frac)
                vals[a] -= sell_amt
                traded_notional += sell_amt

        # KOFR 매도 시 실현이자 15.4% 과세
        if desired_cash < cash_val - 1.0:
            sell_c = cash_val - desired_cash
            frac_c = min(1.0, sell_c / cash_val) if cash_val > 0 else 0.0
            realized_c_gain = max(0.0, (cash_val - cash_cost) * frac_c)
            tax_today += realized_c_gain * tax_rate
            cash_cost *= (1.0 - frac_c)
            cash_val -= sell_c
            traded_notional += sell_c

        # 매수 실행
        for a in assets:
            diff_amt = desired[a] - vals[a]
            if diff_amt > 1.0:
                buy_amt = diff_amt
                vals[a] += buy_amt
                costs[a] += buy_amt
                traded_notional += buy_amt

        if desired_cash > cash_val + 1.0:
            buy_c = desired_cash - cash_val
            cash_val += buy_c
            cash_cost += buy_c
            traded_notional += buy_c

        # 편도 거래량 = traded_notional / 2 (매도+매수 짝 기준, 기존 8bp와 동일 정의)
        oneway_turn = (traded_notional * 0.5) / total_val if (i > 0 and total_val > 0) else 0.0
        fee_today = (traded_notional * 0.5) * fee_oneway if i > 0 else 0.0
        total_fee_paid += fee_today
        total_tax_paid += tax_today

        # 세금과 수수료를 전체 자산에서 비례 차감 (현금 부족 시에도 정확히 순자산 반영)
        drag = tax_today + fee_today
        if drag > 0:
            cur_tot = sum(vals.values()) + cash_val
            scale = max(0.0, (cur_tot - drag) / cur_tot) if cur_tot > 0 else 0.0
            for a in assets:
                vals[a] *= scale
            cash_val *= scale

        end_val = sum(vals.values()) + cash_val
        # 미실현 과세차익 청산 가정 순자산
        unrealized_tax = sum(max(0.0, vals[a] - costs[a]) * tax_rate for a in assets if is_taxable_capgain[a])
        unrealized_tax += max(0.0, cash_val - cash_cost) * tax_rate

        eq_curve[i] = end_val
        eq_after_liq[i] = end_val - unrealized_tax
        daily_rets[i] = (eq_after_liq[i] / eq_after_liq[i - 1] - 1.0) if i > 0 else 0.0
        turnovers[i] = oneway_turn

    eq_s = pd.Series(eq_after_liq, index=dates)
    ret_s = pd.Series(daily_rets, index=dates)
    turn_s = pd.Series(turnovers, index=dates)
    return eq_s, ret_s, turn_s, total_tax_paid, total_fee_paid


def build_ma_signal(close: pd.Series, ma_len: int = 120, confirm: int = 3, brk: float = 0.01) -> pd.Series:
    ma = close.rolling(ma_len).mean()
    disp = close / ma - 1.0
    above = (close > ma).astype(int)
    below = (close < ma).astype(int)
    ab_s = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    bl_s = below.groupby((below != below.shift()).cumsum()).cumsum() * below

    sig = np.zeros(len(close), dtype=int)
    cur = 1 if close.iloc[0] > ma.iloc[0] else 0
    for i in range(len(close)):
        up = (ab_s.iloc[i] >= confirm) or (disp.iloc[i] >= brk)
        dn = (bl_s.iloc[i] >= confirm) or (disp.iloc[i] <= -brk)
        if cur == 1 and dn:
            cur = 0
        elif cur == 0 and up:
            cur = 1
        sig[i] = cur
    return pd.Series(sig, index=close.index)


def build_dual_momentum_weights(
    closes: dict,
    asset_list: list,
    mom_days: int = 120,
    ma_len: int = 120,
    top_k: int = 1,
    rebal_freq: str = "M",  # "M"=월말, "W"=주말, "2W"=격주
    use_ma_filter: bool = True,
    abs_mom_zero: bool = True,
    blend_mom_days: list = None,
) -> pd.DataFrame:
    dates = closes[asset_list[0]].index
    for a in asset_list[1:]:
        dates = dates.intersection(closes[a].index)
    dates = dates.sort_values()
    n = len(dates)
    if blend_mom_days:
        moms = {a: (sum(closes[a].pct_change(d) for d in blend_mom_days) / len(blend_mom_days)).loc[dates] for a in asset_list}
    else:
        moms = {a: closes[a].pct_change(mom_days).loc[dates] for a in asset_list}
    mas = {a: closes[a].rolling(ma_len).mean().loc[dates] for a in asset_list}
    aligned_closes = {a: closes[a].loc[dates] for a in asset_list}

    rebal = np.zeros(n, dtype=bool)
    rebal[0] = True
    if rebal_freq == "M":
        for i in range(n - 1):
            if dates[i].month != dates[i + 1].month:
                rebal[i] = True
    elif rebal_freq == "W":
        for i in range(n - 1):
            if dates[i].isocalendar().week != dates[i + 1].isocalendar().week:
                rebal[i] = True
    elif rebal_freq == "2W":
        wk_cnt = 0
        for i in range(n - 1):
            if dates[i].isocalendar().week != dates[i + 1].isocalendar().week:
                wk_cnt += 1
                if wk_cnt % 2 == 0:
                    rebal[i] = True
    elif rebal_freq == "D":
        rebal[:] = True

    W = np.zeros((n, len(asset_list)), dtype=float)
    cur_w = np.zeros(len(asset_list), dtype=float)
    for i in range(n):
        if rebal[i]:
            scored = []
            for j, a in enumerate(asset_list):
                mv = moms[a].iloc[i]
                cv = aligned_closes[a].iloc[i]
                mav = mas[a].iloc[i]
                if np.isnan(mv) or np.isnan(mav):
                    continue
                ok_abs = (mv > 0) if abs_mom_zero else True
                ok_ma = (cv > mav) if use_ma_filter else True
                if ok_abs and ok_ma:
                    scored.append((mv, j))
            scored.sort(reverse=True)
            cur_w = np.zeros(len(asset_list), dtype=float)
            for _, j in scored[:top_k]:
                cur_w[j] = 1.0 / top_k
        W[i] = cur_w
    return pd.DataFrame(W, index=dates, columns=asset_list)


def build_kospi_kosdaq_hybrid_fsm_weights(
    df: pd.DataFrame,
    k200_close: pd.Series,
    kq_close: pd.Series,
    mom_days: int = 60,
    ma_len: int = 120,
    cr_overheat: float = 2.4,
    cr_calm: float = 2.1,
    rebal_switch_buffer: float = 0.02,
    use_s4: bool = True,
) -> pd.DataFrame:
    """
    순수 국내 2대 지수(KODEX 200 + KODEX 코스닥150, 100% 매매차익 비과세!) 결합 FSM:
    - 각 지수의 추세(MA120) 판정: k200_up, kq_up
    - 둘 다 상승: 모멘텀 승자 100% 보유 (단, 코스닥150 승자여도 신용과열 State 2이면 50% KODEX200 + 50% 코스닥150으로 분산!)
    - 한쪽만 상승: 상승 중인 지수 100% 보유 (코스닥 하락 시 KOFR 대신 상승 중인 KOSPI200으로 이동!)
    - 둘 다 하락: KOFR 대피 (단, 코스닥 투매역발상 State 4 조건 충족 시 코스닥150 30% 단기 반등 포착)
    """
    dates = df.index
    n = len(dates)
    k200_sig = build_ma_signal(k200_close, ma_len=ma_len)
    kq_sig = build_ma_signal(kq_close, ma_len=ma_len)

    m_k200 = k200_close.pct_change(mom_days)
    m_kq = kq_close.pct_change(mom_days)

    ma120_kq = kq_close.rolling(ma_len).mean()
    ma5_kq = kq_close.rolling(5).mean()
    disp_kq = kq_close / ma120_kq - 1.0

    W = np.zeros((n, 2), dtype=float)  # [k200, kq]
    cur_leader = "k200"
    kq_overheated = False
    s4_active = False
    s4_entry = None
    s4_cool = 0

    for i in range(n):
        up_k200 = bool(k200_sig.iloc[i] == 1)
        up_kq = bool(kq_sig.iloc[i] == 1)
        cr_r = df["cr_ratio"].iloc[i]
        cr_10 = df["cr_chg_10d"].iloc[i]
        cr_20 = df["cr_chg_20d"].iloc[i]

        # 히스테리시스 과열 판정
        if not kq_overheated and ((cr_r >= cr_overheat) or (cr_20 >= 0.10)):
            kq_overheated = True
        elif kq_overheated and ((cr_r <= cr_calm) and (cr_20 < 0.05)):
            kq_overheated = False

        # 리더 판정 (휩소 방지를 위해 버퍼 적용)
        mk2 = m_k200.iloc[i]
        mkq = m_kq.iloc[i]
        if not np.isnan(mk2) and not np.isnan(mkq):
            if cur_leader == "k200" and (mkq > mk2 + rebal_switch_buffer):
                cur_leader = "kq"
            elif cur_leader == "kq" and (mk2 > mkq + rebal_switch_buffer):
                cur_leader = "k200"

        # State 4 관리
        if s4_active:
            tr = kq_close.iloc[i] / s4_entry - 1.0
            if tr >= 0.10 or kq_close.iloc[i] >= ma120_kq.iloc[i] or tr <= -0.07 or up_k200 or up_kq:
                s4_active = False
                s4_entry = None
                s4_cool = 10
        else:
            if s4_cool > 0:
                s4_cool -= 1

        # 비중 결정
        if up_k200 and up_kq:
            if cur_leader == "k200":
                W[i] = [1.0, 0.0]
            else:
                # 코스닥이 리더지만 과열이면 코스피200으로 절반 분산 (현금 대기 대신 비과세 상승지수 활용!)
                W[i] = [0.5, 0.5] if kq_overheated else [0.0, 1.0]
        elif up_k200 and not up_kq:
            W[i] = [1.0, 0.0]
        elif not up_k200 and up_kq:
            W[i] = [0.0, 0.5] if kq_overheated else [0.0, 1.0]
        else:
            if use_s4 and not s4_active and s4_cool == 0:
                if disp_kq.iloc[i] <= -0.10 and cr_10 <= -0.07 and kq_close.iloc[i] > ma5_kq.iloc[i]:
                    s4_active = True
                    s4_entry = kq_close.iloc[i]
            W[i] = [0.0, 0.3] if s4_active else [0.0, 0.0]

    return pd.DataFrame(W, index=dates, columns=["k200", "kq"])


def main():
    base_cfg = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True, "cr_overheat_pct": 2.4, "cr_calm_pct": 2.1}
    df_base = load_market_data(base_cfg)

    # Load all 1x spot ETFs with exact tax flags
    etf_files = {
        "kq": ("kodex_kosdaq150_nav.csv", False),      # 국내주식형 1x (매매차익 비과세)
        "k200": ("kodex_200_nav.csv", False),          # 국내주식형 1x (매매차익 비과세)
        "nq": ("tiger_nasdaq100_nav.csv", True),       # 해외주식형 1x (15.4% 과세)
        "gold": ("kodex_gold_nav.csv", True),          # 파생/원자재 1x (15.4% 과세)
    }
    # Optional domestic sector 1x ETFs (매매차익 비과세!)
    for key in ["semi", "bank", "auto", "steel", "chem"]:
        p = os.path.join(BASE_DIR, f"kodex_{key}_nav.csv")
        if os.path.exists(p):
            etf_files[key] = (f"kodex_{key}_nav.csv", False)

    loaded = {}
    for k, (fn, is_tax) in etf_files.items():
        loaded[k] = load_etf_tax_aware(fn, is_domestic_equity=(not is_tax))

    # Align common index starting from df_base.index[0] (2016-06-29 ~ 2026-11-24)
    common_idx = df_base.index
    for k in loaded:
        common_idx = common_idx.intersection(loaded[k].index)
    common_idx = common_idx.sort_values()

    df = df_base.loc[common_idx].copy()
    kofr_ret = df["kofr_ret"].copy()
    kofr_ret.iloc[0] = 0.0
    kofr_after_tax = kofr_ret * (1.0 - 0.154)

    # Full-history closes for indicator warmup (before slicing to common_idx)
    full_closes = {k: loaded[k]["adj_close"] for k in loaded}
    price_rets = {k: loaded[k].loc[common_idx, "price_ret"] for k in loaded}
    total_rets = {k: loaded[k].loc[common_idx, "total_ret"] for k in loaded}
    div_rets = {k: loaded[k].loc[common_idx, "div_ret"] for k in loaded}
    is_taxable = {k: etf_files[k][1] for k in loaded}

    print("=" * 130)
    print(f" [일반계좌 · 1x 노레버리지 · 지수 전용 백테스트 (실제 세법: 국내주식형 매매차익 비과세 / 해외·KOFR 15.4% 실현과세)]")
    print(f" 검증 기간: {common_idx[0].strftime('%Y-%m-%d')} ~ {common_idx[-1].strftime('%Y-%m-%d')} ({len(common_idx)}영업일 / 10.4년, 편도 8bp)")
    print("=" * 130)

    def eval_weights(name: str, W: pd.DataFrame):
        W_aligned = W.loc[common_idx].copy()
        eq, ret, turn, tax_p, fee_p = simulate_general_account_portfolio(
            common_idx, W_aligned, price_rets, total_rets, div_rets, is_taxable, kofr_ret
        )
        m = calc_metrics(eq, ret, kofr_after_tax)
        r1 = ret.loc[:"2021-12-31"]
        r2 = ret.loc["2022-01-01":]
        m1 = calc_metrics((1 + r1).cumprod(), r1, kofr_after_tax.loc[:"2021-12-31"])
        m2 = calc_metrics((1 + r2).cumprod(), r2, kofr_after_tax.loc["2022-01-01":])
        yrs = (common_idx[-1] - common_idx[0]).days / 365.25
        return {
            "전략 (일반계좌 세후 · 1x 노레버리지)": name,
            "세후자산(만)": f"{eq.iloc[-1]/1e4:,.0f}",
            "세후CAGR": f"{m['cagr_pct']:+.2f}%",
            "Vol": f"{m['ann_vol_pct']:.1f}%",
            "초과샤프": f"{m['sharpe_excess']:.2f}",
            "MDD": f"{m['mdd_pct']:.1f}%",
            "칼마": f"{m['calmar']:.2f}",
            "전반기(16~21)": f"{m1['cagr_pct']:+.1f}%/{m1['mdd_pct']:.0f}%",
            "후반기(22~26)": f"{m2['cagr_pct']:+.1f}%/{m2['mdd_pct']:.0f}%",
            "총세금(만)": f"{tax_p/1e4:,.0f}",
            "연회전": f"{turn.sum()/yrs:.1f}",
        }, eq, ret

    rows = []
    eq_map = {}

    # === Group 1: 국내 대표지수 단독 (KODEX 200 / 코스닥150) ===
    for name, W in [
        ("1A. [단독] KODEX 코스닥150 단순보유 (비과세)", pd.DataFrame({"kq": 1.0}, index=common_idx)),
        ("1B. [단독] KODEX 200 단순보유 (비과세)", pd.DataFrame({"k200": 1.0}, index=common_idx)),
        ("1C. [단독] KODEX 200 MA120 추세추종 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 120).loc[common_idx]}, index=common_idx)),
        ("1D. [단독] KODEX 200 MA60 추세추종 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 60).loc[common_idx]}, index=common_idx)),
        ("1E. [단독] KODEX 200 MA150 추세추종 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 150).loc[common_idx]}, index=common_idx)),
        ("1F. [단독] KODEX 200 MA200 추세추종 (비과세)", pd.DataFrame({"k200": build_ma_signal(full_closes["k200"], 200).loc[common_idx]}, index=common_idx)),
    ]:
        r, eq, _ = eval_weights(name, W); rows.append(r); eq_map[name] = eq

    # === Group 2: 순수 국내 2대 지수만 (KODEX 200 + 코스닥150, 100% 매매차익 비과세!) ===
    W_2a_m120 = build_dual_momentum_weights(full_closes, ["k200", "kq"], mom_days=120, ma_len=120, top_k=1, rebal_freq="M")
    W_2a_m60 = build_dual_momentum_weights(full_closes, ["k200", "kq"], mom_days=60, ma_len=120, top_k=1, rebal_freq="M")
    W_2a_bl = build_dual_momentum_weights(full_closes, ["k200", "kq"], blend_mom_days=[20, 60, 120], ma_len=120, top_k=1, rebal_freq="M")
    W_2a_w = build_dual_momentum_weights(full_closes, ["k200", "kq"], mom_days=60, ma_len=120, top_k=1, rebal_freq="W")
    W_2b_hyb = build_kospi_kosdaq_hybrid_fsm_weights(df, full_closes["k200"].loc[common_idx], full_closes["kq"].loc[common_idx], mom_days=60, ma_len=120, cr_overheat=2.4, cr_calm=2.1, use_s4=True)
    W_2b_hyb120 = build_kospi_kosdaq_hybrid_fsm_weights(df, full_closes["k200"].loc[common_idx], full_closes["kq"].loc[common_idx], mom_days=120, ma_len=120, cr_overheat=2.4, cr_calm=2.1, use_s4=True)
    W_2c_5050 = pd.DataFrame({
        "k200": 0.5 * build_ma_signal(full_closes["k200"], 120).loc[common_idx],
        "kq": 0.5 * build_ma_signal(full_closes["kq"], 120).loc[common_idx],
    }, index=common_idx)

    for name, W in [
        ("2A. [국내2지수] 50:50 각각 MA120 추세추종 (비과세)", W_2c_5050),
        ("2B. [국내2지수] 월간 듀얼모멘텀(6M) Top-1 (비과세)", W_2a_m120),
        ("2C. [국내2지수] 월간 듀얼모멘텀(3M) Top-1 (비과세)", W_2a_m60),
        ("2D. [국내2지수] 월간 복합모멘텀(1·3·6M) Top-1 (비과세)", W_2a_bl),
        ("2E. [국내2지수] 주간 듀얼모멘텀(3M) Top-1 (비과세)", W_2a_w),
        ("2F. [국내2지수] 코스피200+코스닥150 하이브리드 FSM (3M, 비과세)", W_2b_hyb),
        ("2G. [국내2지수] 코스피200+코스닥150 하이브리드 FSM (6M, 비과세)", W_2b_hyb120),
    ]:
        r, eq, _ = eval_weights(name, W); rows.append(r); eq_map[name] = eq

    # === Group 3: 국내2지수 + 미국나스닥100 (3대 대표지수 1x, 일반계좌 15.4% 세후!) ===
    W_3a = build_dual_momentum_weights(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=120, top_k=1, rebal_freq="M")
    W_3b = build_dual_momentum_weights(full_closes, ["k200", "kq", "nq"], mom_days=120, ma_len=120, top_k=2, rebal_freq="M")
    W_3c = build_dual_momentum_weights(full_closes, ["k200", "kq", "nq"], blend_mom_days=[60, 120, 180], ma_len=120, top_k=2, rebal_freq="M")
    # 3D: 50% 국내2지수 하이브리드 FSM (비과세) + 50% 나스닥100 MA120 (과세)
    W_3d = pd.DataFrame({
        "k200": 0.5 * W_2b_hyb["k200"],
        "kq": 0.5 * W_2b_hyb["kq"],
        "nq": 0.5 * build_ma_signal(full_closes["nq"], 120).loc[common_idx],
    }, index=common_idx)

    for name, W in [
        ("3A. [한·미 3지수] 월간 듀얼모멘텀(6M) Top-1 (세후)", W_3a),
        ("3B. [한·미 3지수] 월간 듀얼모멘텀(6M) Top-2 (세후)", W_3b),
        ("3C. [한·미 3지수] 월간 복합모멘텀(3·6·9M) Top-2 (세후)", W_3c),
        ("3D. [한·미 3지수] 50% 국내2지수FSM(비과세) + 50% 나스닥MA120(세후)", W_3d),
    ]:
        r, eq, _ = eval_weights(name, W); rows.append(r); eq_map[name] = eq

    # === Group 4: 100% 국내 지수만 (대표지수 2종 + KRX 주요 섹터지수, 전액 매매차익 비과세!) ===
    dom_sectors = [k for k in ["k200", "kq", "semi", "bank", "auto", "steel", "chem"] if k in loaded]
    if len(dom_sectors) >= 4:
        W_4a = build_dual_momentum_weights(full_closes, dom_sectors, mom_days=120, ma_len=120, top_k=2, rebal_freq="M")
        W_4b = build_dual_momentum_weights(full_closes, dom_sectors, blend_mom_days=[20, 60, 120], ma_len=120, top_k=2, rebal_freq="M")
        W_4c = build_dual_momentum_weights(full_closes, dom_sectors, mom_days=120, ma_len=120, top_k=3, rebal_freq="M")
        for name, W in [
            (f"4A. [100% 국내지수 {len(dom_sectors)}종] 월간 모멘텀(6M) Top-2 (비과세)", W_4a),
            (f"4B. [100% 국내지수 {len(dom_sectors)}종] 월간 복합모멘텀 Top-2 (비과세)", W_4b),
            (f"4C. [100% 국내지수 {len(dom_sectors)}종] 월간 모멘텀(6M) Top-3 (비과세)", W_4c),
        ]:
            r, eq, _ = eval_weights(name, W); rows.append(r); eq_map[name] = eq

    res_tbl = pd.DataFrame(rows)
    print(res_tbl.to_string(index=False))
    res_tbl.to_csv(os.path.join(BASE_DIR, "general_account_1x_summary.csv"), index=False)


if __name__ == "__main__":
    main()
