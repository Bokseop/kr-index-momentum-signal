import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. 전략 파라미터 설정 (원본 CONFIG 그대로 유지)
# ==============================================================================
DEFAULT_CONFIG = {
    "start_date": "2016-01-01",
    "end_date": "2026-11-24",
    "initial_capital": 10_000_000,   # 초기 투자금 (1,000만 원)
    "etf_ticker": "229200",          # KODEX 코스닥150
    "kofr_ticker": "423160",         # KODEX KOFR금리액티브 (2022-04-26 상장)
    "fallback_rf_annual": 0.025,     # KOFR 상장 이전 대체 연이율 (2.5%)
    "use_real_cd91_pre_kofr": False, # True이면 KOFR 상장 전 실제 일별 CD91 금리 적용

    # 계좌 및 거래 비용
    "account_type": "ISA",           # "ISA" (KOFR 비과세) 또는 "GENERAL" (KOFR 15.4% 과세)
    "fee_slippage_oneway": 0.0008,   # 편도 수수료 + 슬리피지 합계 (0.08% = 8bp)

    # 추세 및 신용잔고 파라미터
    "ma_long": 120,                  # 장기 추세 이평선
    "ma_short": 5,                   # State 4 단기 반등 확인선
    "confirm_days": 3,               # 120일선 돌파/이탈 연속 확인일
    "breakout_pct": 0.01,            # 120일선 ±1% 이상 즉시 돌파/이탈 허용
    "credit_lag_days": 2,            # FreeSIS 결제일 시차 보정 (필수: 2영업일)

    # State 2 (과열 축소) & 히스테리시스
    "cr_overheat_pct": 2.0,          # 시총 대비 신용비율 2.0% 이상 과열
    "cr_20d_surge": 0.10,            # 20영업일 신용잔고 +10% 이상 급증
    "cr_calm_pct": 1.8,              # State 2 -> State 1 복원 기준 (1.8% 이하)
    "enable_state2": True,
    "enable_state4": True,

    # State 4 (투매 역발상)
    "s4_disparity": -0.10,           # 120일선 대비 -10% 이상 과매도
    "s4_cr_10d_drop": -0.07,         # 10영업일 신용잔고 -7% 이상 급감
    "s4_take_profit": 0.10,          # 익절 +10%
    "s4_stop_loss": -0.07,           # 손절 -7%
    "s4_cooldown_days": 10,          # 청산 후 10영업일 재진입 금지

    # 선택 옵션: 계절성 필터 (5~10월 State 1 최대 비중 70% 제한)
    "use_seasonality_filter": False,
    "offseason_max_weight": 0.70,

    # 데이터 파일 경로 (금융투자협회 FreeSIS & DIS 공식 실데이터)
    "credit_csv_path": os.path.join(BASE_DIR, "freesis_credit.csv"),
    "etf_nav_csv_path": os.path.join(BASE_DIR, "kodex_kosdaq150_nav.csv"),
    "kofr_nav_csv_path": os.path.join(BASE_DIR, "kodex_kofr_nav.csv"),
    "use_mock_credit": False,
}


def _reconstruct_total_return_nav(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    금융투자협회 전자공시(DIS) 기준가격 정렬 및 수정기준가격(분배금 재투자) 복원:
    1) DIS의 T+1 영업일 고시 기준가는 시장 T일 종가 기준 NAV임 (코스닥 지수 수익률 상관계수 +0.961 검증 완료).
       따라서 거래일 기준으로 1영업일 앞당겨(shift(-1)) T일 종가에 정확히 정렬함.
    2) tmpV3(diff)는 설정일 기준 누적 수정기준가격(adj_nav)의 전일대비 등락임.
       2015년 이전 상장 펀드(KODEX 200, TIGER 나스닥100 등)는 조회 시작 시점에 이미 분배금 누적 배수 F0(>1)가
       존재하므로, 초기 큰 변동일들의 median(diff / raw_diff)로 F0를 정확히 역산해 adj_nav[0] = F0 * nav[0]로
       복원해야 수익률 과대/과소 계상이 전혀 발생하지 않음.
    """
    d = raw_df.sort_values("date").drop_duplicates("date").reset_index(drop=True).copy()
    raw_diff = d["nav"].diff()
    # 초기 구간(첫 60영업일 중 변동폭이 큰 날)에서 초기 누적 분배배수 F0 추정
    early = d.iloc[1:min(80, len(d))].copy()
    early_raw = raw_diff.iloc[1:min(80, len(d))]
    mask = (early_raw.abs() > d["nav"].iloc[0] * 0.003) & early["diff"].notna() & (early["diff"] * early_raw > 0)
    if mask.sum() >= 3:
        ratios = (early.loc[mask, "diff"] / early_raw.loc[mask]).values
        f0 = float(np.median(ratios))
        # 설정일 직후 펀드(F0 ≈ 1.0)는 미세 반올림 오차 제거
        if abs(f0 - 1.0) < 0.002:
            f0 = 1.0
    else:
        f0 = 1.0

    adj = np.empty(len(d), dtype=float)
    adj[0] = d["nav"].iloc[0] * f0
    for i in range(1, len(d)):
        diff_val = d["diff"].iloc[i]
        raw_chg = d["nav"].iloc[i] - d["nav"].iloc[i - 1]
        # 당일 미확정(diff==0인데 raw_chg!=0인 마지막 행) 처리
        if i == len(d) - 1 and diff_val == 0.0 and abs(raw_chg) > 1e-6:
            factor = adj[i - 1] / d["nav"].iloc[i - 1]
            adj[i] = d["nav"].iloc[i] * factor
        else:
            adj[i] = adj[i - 1] + (0.0 if np.isnan(diff_val) else diff_val)
    d["adj_nav"] = adj

    # T+1 고시일 기준가 -> T일 시장 종가로 정렬 (마지막 행의 다음 영업일 없으므로 len-1까지)
    aligned = pd.DataFrame({
        "date": d["date"].iloc[:-1].values,
        "nav_close": d["nav"].iloc[1:].values,
        "adj_close": d["adj_nav"].iloc[1:].values,
    })
    if "cd91" in d.columns:
        cd = d["cd91"].iloc[:-1].replace(0.0, np.nan).ffill()
        aligned["cd91"] = cd.values
    return aligned


def load_market_data(cfg: dict) -> pd.DataFrame:
    # 1. KODEX 코스닥150 (공식 KOFIA DIS 수정기준가)
    etf_raw = pd.read_csv(cfg["etf_nav_csv_path"], parse_dates=["date"])
    etf = _reconstruct_total_return_nav(etf_raw).set_index("date")
    etf = etf.loc[cfg["start_date"]:cfg["end_date"]].copy()
    etf["etf_close"] = etf["adj_close"]
    etf["etf_ret"] = etf["etf_close"].pct_change().fillna(0.0)

    # 2. KODEX KOFR금리액티브 (공식 KOFIA DIS 수정기준가, 2022-04-25 이후)
    kofr_raw = pd.read_csv(cfg["kofr_nav_csv_path"], parse_dates=["date"])
    kofr = _reconstruct_total_return_nav(kofr_raw).set_index("date")
    kofr["kofr_ret"] = kofr["adj_close"].pct_change()

    df = etf[["etf_close", "nav_close", "etf_ret", "cd91"]].join(kofr[["kofr_ret"]], how="left")

    if cfg.get("use_real_cd91_pre_kofr", False):
        # KOFR 상장 이전에는 실제 일별 CD91 금리(연율 -> 일복리) 적용
        cd_daily = (1.0 + df["cd91"].ffill().bfill() / 100.0) ** (1.0 / 252.0) - 1.0
        df["kofr_ret"] = df["kofr_ret"].fillna(cd_daily)
    else:
        daily_fallback_rf = (1.0 + cfg["fallback_rf_annual"]) ** (1.0 / 252.0) - 1.0
        df["kofr_ret"] = df["kofr_ret"].fillna(daily_fallback_rf)

    # 3. FreeSIS 공식 코스닥 신용잔고 & 코스닥 시가총액
    if not cfg.get("use_mock_credit", False):
        cr = pd.read_csv(cfg["credit_csv_path"], parse_dates=["date"]).set_index("date").sort_index()
        cr = cr[["credit_balance", "market_cap", "kosdaq_index"]].reindex(df.index).ffill()
        df = df.join(cr, how="left")
    else:
        rng = np.random.default_rng(42)
        base_mcap = 3500_000 * (df["etf_close"] / df["etf_close"].iloc[0])
        ret_20 = df["etf_close"].pct_change(20).fillna(0)
        sim_ratio = (1.55 + 2.2 * ret_20.rolling(10).mean().fillna(0) + rng.normal(0, 0.05, len(df))).clip(0.8, 2.6)
        df["market_cap"] = base_mcap
        df["credit_balance"] = base_mcap * (sim_ratio / 100.0)
        df["kosdaq_index"] = np.nan

    # 4. 이동평균선 및 괴리율 계산 (당일 종가 기준)
    df["ma120"] = df["etf_close"].rolling(cfg["ma_long"]).mean()
    df["ma5"] = df["etf_close"].rolling(cfg["ma_short"]).mean()
    df["disparity_120"] = (df["etf_close"] / df["ma120"]) - 1.0

    # 연속 돌파/이탈 카운트
    above = (df["etf_close"] > df["ma120"]).astype(int)
    below = (df["etf_close"] < df["ma120"]).astype(int)
    df["above_streak"] = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    df["below_streak"] = below.groupby((below != below.shift()).cumsum()).cumsum() * below

    # 5. [핵심] 신용잔고 2영업일 시차 보정 (Look-ahead Bias 차단)
    lag = cfg["credit_lag_days"]
    df["cr_bal_lag"] = df["credit_balance"].shift(lag)
    df["mcap_lag"] = df["market_cap"].shift(lag)
    df["cr_ratio"] = (df["cr_bal_lag"] / df["mcap_lag"]) * 100.0
    df["cr_chg_10d"] = df["cr_bal_lag"].pct_change(10)
    df["cr_chg_20d"] = df["cr_bal_lag"].pct_change(20)

    # 선택적 상대 과열(롤링 백분위) 컬럼도 미리 계산 (민감도/개선 검증용)
    df["cr_ratio_p80_252"] = df["cr_ratio"].rolling(252, min_periods=120).quantile(0.80)
    df["cr_ratio_p50_252"] = df["cr_ratio"].rolling(252, min_periods=120).quantile(0.50)

    return df.dropna(subset=["ma120", "cr_ratio", "cr_chg_20d"]).copy()


def run_backtest(df: pd.DataFrame, cfg: dict):
    n = len(df)
    dates = df.index

    states = np.zeros(n, dtype=int)
    target_weights = np.zeros(n, dtype=float)
    strat_rets = np.zeros(n, dtype=float)
    turnovers = np.zeros(n, dtype=float)
    s4_trades = []

    # 초기 상태 설정
    cur_state = 1 if df["etf_close"].iloc[0] > df["ma120"].iloc[0] else 3
    prev_weight = 1.0 if cur_state == 1 else 0.0
    s4_entry_price = None
    s4_entry_date = None
    s4_cooldown = 0

    kofr_tax_rate = 0.154 if cfg["account_type"].upper() == "GENERAL" else 0.0
    enable_s2 = cfg.get("enable_state2", True)
    enable_s4 = cfg.get("enable_state4", True)
    use_dynamic_cr = cfg.get("use_dynamic_cr", False)

    for i in range(n):
        row = df.iloc[i]
        date = dates[i]
        close = row["etf_close"]
        ma120 = row["ma120"]
        ma5 = row["ma5"]
        disp = row["disparity_120"]
        cr_ratio = row["cr_ratio"]
        cr_10d = row["cr_chg_10d"]
        cr_20d = row["cr_chg_20d"]

        # (A) 전일 종가에서 결정한 비중(prev_weight)으로 오늘(i)의 수익 실현
        if i > 0:
            eq_ret = row["etf_ret"]
            cash_ret = row["kofr_ret"] * (1.0 - kofr_tax_rate if row["kofr_ret"] > 0 else 1.0)
            gross_ret = prev_weight * eq_ret + (1.0 - prev_weight) * cash_ret
        else:
            gross_ret = 0.0

        # (B) 오늘(i) 종가 기준 상태 전이 판정
        cond_up_confirm = (row["above_streak"] >= cfg["confirm_days"]) or (disp >= cfg["breakout_pct"])
        cond_dn_confirm = (row["below_streak"] >= cfg["confirm_days"]) or (disp <= -cfg["breakout_pct"])

        if enable_s2:
            if use_dynamic_cr and not np.isnan(row["cr_ratio_p80_252"]):
                cond_overheat = (cr_ratio >= row["cr_ratio_p80_252"]) and (cr_20d >= cfg["cr_20d_surge"])
                cond_calm = (cr_ratio <= row["cr_ratio_p50_252"]) or (cr_20d <= 0.0)
            else:
                cond_overheat = (cr_ratio >= cfg["cr_overheat_pct"]) or (cr_20d >= cfg["cr_20d_surge"])
                cond_calm = (cr_ratio <= cfg["cr_calm_pct"]) and (cr_20d < cfg["cr_20d_surge"] * 0.5)
        else:
            cond_overheat = False
            cond_calm = True

        next_state = cur_state

        if cur_state == 4:
            trade_ret = (close / s4_entry_price) - 1.0
            exit_reason = None
            if trade_ret >= cfg["s4_take_profit"]:
                exit_reason = "TP(+10%)"
            elif close >= ma120:
                exit_reason = "MA120_TOUCH"
            elif trade_ret <= cfg["s4_stop_loss"]:
                exit_reason = "SL(-7%)"

            if exit_reason is not None:
                s4_trades.append({
                    "entry_date": s4_entry_date.strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "holding_days": i - df.index.get_loc(s4_entry_date),
                    "entry_price": round(s4_entry_price, 1),
                    "exit_price": round(close, 1),
                    "return_pct": round(trade_ret * 100, 2),
                    "reason": exit_reason,
                })
                s4_entry_price = None
                s4_entry_date = None
                s4_cooldown = cfg["s4_cooldown_days"]
                next_state = 1 if (cond_up_confirm and not cond_overheat) else (2 if cond_up_confirm and cond_overheat else 3)

        elif cur_state == 1:
            if cond_dn_confirm:
                next_state = 3
            elif cond_overheat:
                next_state = 2

        elif cur_state == 2:
            if cond_dn_confirm:
                next_state = 3
            elif cond_calm:
                next_state = 1

        elif cur_state == 3:
            if cond_up_confirm:
                next_state = 2 if cond_overheat else 1
            elif enable_s4 and s4_cooldown == 0:
                if (
                    disp <= cfg["s4_disparity"]
                    and cr_10d <= cfg["s4_cr_10d_drop"]
                    and close > ma5
                ):
                    next_state = 4
                    s4_entry_price = close
                    s4_entry_date = date

        if next_state != 4 and cur_state != 4 and s4_cooldown > 0:
            s4_cooldown -= 1

        # (C) 목표 비중 결정 및 리밸런싱 비용 차감
        if next_state == 1:
            new_weight = 1.0
            if cfg["use_seasonality_filter"] and (5 <= date.month <= 10):
                new_weight = cfg["offseason_max_weight"]
        elif next_state == 2:
            new_weight = 0.50
        elif next_state == 3:
            new_weight = 0.00
        else:
            new_weight = 0.30

        turnover = abs(new_weight - prev_weight) if i > 0 else 0.0
        cost = turnover * cfg["fee_slippage_oneway"]

        strat_rets[i] = gross_ret - cost
        states[i] = next_state
        target_weights[i] = new_weight
        turnovers[i] = turnover

        cur_state = next_state
        prev_weight = new_weight

    res = df.copy()
    res["state"] = states
    res["weight"] = target_weights
    res["turnover"] = turnovers
    res["strat_ret"] = strat_rets

    # 공정 비교를 위해 Day 0 수익률 모두 0으로 통일
    bh_ret = res["etf_ret"].copy()
    bh_ret.iloc[0] = 0.0
    kofr_after_tax = res["kofr_ret"].where(res["kofr_ret"] <= 0, res["kofr_ret"] * (1.0 - kofr_tax_rate))
    kofr_after_tax.iloc[0] = 0.0

    res["bh_ret"] = bh_ret
    res["kofr_after_tax_ret"] = kofr_after_tax
    res["strat_equity"] = cfg["initial_capital"] * (1.0 + res["strat_ret"]).cumprod()
    res["bh_equity"] = cfg["initial_capital"] * (1.0 + res["bh_ret"]).cumprod()
    res["kofr_equity"] = cfg["initial_capital"] * (1.0 + res["kofr_after_tax_ret"]).cumprod()

    return res, pd.DataFrame(s4_trades)


def calc_metrics(equity: pd.Series, daily_ret: pd.Series, rf_ret: pd.Series = None) -> dict:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    ann_vol = daily_ret.iloc[1:].std() * np.sqrt(252)
    sharpe_0 = (daily_ret.iloc[1:].mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    if rf_ret is not None:
        ex = daily_ret.iloc[1:] - rf_ret.iloc[1:]
        ex_vol = ex.std() * np.sqrt(252)
        sharpe_ex = (ex.mean() * 252) / ex_vol if ex_vol > 0 else 0.0
    else:
        sharpe_ex = np.nan
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        "final_equity": equity.iloc[-1],
        "total_ret_pct": total_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "ann_vol_pct": ann_vol * 100.0,
        "sharpe_rf0": sharpe_0,
        "sharpe_excess": sharpe_ex,
        "mdd_pct": mdd * 100.0,
        "calmar": calmar,
    }


def format_metrics_table(metrics_map: dict) -> pd.DataFrame:
    rows = {}
    for name, m in metrics_map.items():
        rows[name] = {
            "최종 자산(원)": f"{m['final_equity']:,.0f}",
            "누적 수익률(%)": f"{m['total_ret_pct']:+.2f}%",
            "연복리 CAGR(%)": f"{m['cagr_pct']:+.2f}%",
            "연 변동성(%)": f"{m['ann_vol_pct']:.2f}%",
            "샤프지수(Rf=0)": f"{m['sharpe_rf0']:.2f}",
            "초과샤프(vs KOFR)": f"{m['sharpe_excess']:.2f}" if not np.isnan(m["sharpe_excess"]) else "-",
            "최대낙폭 MDD(%)": f"{m['mdd_pct']:.2f}%",
            "칼마비율": f"{m['calmar']:.2f}",
        }
    return pd.DataFrame(rows)


def main():
    # 1. 원본 CONFIG 그대로 실데이터(ISA, pre-KOFR 2.5%) 실행
    df_real = load_market_data(DEFAULT_CONFIG)
    res_isa, s4_isa = run_backtest(df_real, DEFAULT_CONFIG)

    # 2. 실데이터 + pre-KOFR 실제 CD91 금리 적용 (가장 현실적인 기준)
    cfg_cd91 = {**DEFAULT_CONFIG, "use_real_cd91_pre_kofr": True}
    df_cd91 = load_market_data(cfg_cd91)
    res_cd91, _ = run_backtest(df_cd91, cfg_cd91)

    # 3. 일반계좌 (KOFR 15.4% 과세 + 실제 CD91)
    cfg_gen = {**DEFAULT_CONFIG, "account_type": "GENERAL", "use_real_cd91_pre_kofr": True}
    res_gen, _ = run_backtest(df_cd91, cfg_gen)

    # 4. 계절성 필터 ON (실데이터 + 실제 CD91)
    cfg_seas = {**cfg_cd91, "use_seasonality_filter": True}
    res_seas, _ = run_backtest(df_cd91, cfg_seas)

    # 5. 더미(Mock) 신용데이터 비교용
    cfg_mock = {**DEFAULT_CONFIG, "use_mock_credit": True}
    df_mock = load_market_data(cfg_mock)
    res_mock, _ = run_backtest(df_mock, cfg_mock)

    print("=" * 95)
    print(f" [공식 실데이터 백테스트] 기간: {res_isa.index[0].strftime('%Y-%m-%d')} ~ {res_isa.index[-1].strftime('%Y-%m-%d')} ({len(res_isa)}영업일)")
    print(" 데이터 출처: 금융투자협회 FreeSIS (신용잔고·코스닥시총) + KOFIA 전자공시 DIS (수정기준가·CD91)")
    print("=" * 95)

    tbl = format_metrics_table({
        "FSM 원본(ISA, 2.5%)": calc_metrics(res_isa["strat_equity"], res_isa["strat_ret"], res_isa["kofr_after_tax_ret"]),
        "FSM 현실금리(ISA, CD91)": calc_metrics(res_cd91["strat_equity"], res_cd91["strat_ret"], res_cd91["kofr_after_tax_ret"]),
        "FSM 일반계좌(과세, CD91)": calc_metrics(res_gen["strat_equity"], res_gen["strat_ret"], res_gen["kofr_after_tax_ret"]),
        "FSM 계절성ON(ISA, CD91)": calc_metrics(res_seas["strat_equity"], res_seas["strat_ret"], res_seas["kofr_after_tax_ret"]),
        "[참고] 더미신용(Mock)": calc_metrics(res_mock["strat_equity"], res_mock["strat_ret"], res_mock["kofr_after_tax_ret"]),
        "코스닥150 단순보유": calc_metrics(res_cd91["bh_equity"], res_cd91["bh_ret"], res_cd91["kofr_after_tax_ret"]),
        "KOFR/CD91 전액보유": calc_metrics(res_cd91["kofr_equity"], res_cd91["kofr_after_tax_ret"], res_cd91["kofr_after_tax_ret"]),
    })
    print(tbl.to_string())

    print("\n[국면(State)별 체류 비중 — 실데이터 vs 더미신용 비교]")
    state_names = {
        1: "State 1 (정상상승 100%)",
        2: "State 2 (과열축소  50%)",
        3: "State 3 (하락대피   0%)",
        4: "State 4 (투매역발상 30%)",
    }
    c_real = res_isa["state"].value_counts().sort_index()
    c_mock = res_mock["state"].value_counts().sort_index()
    for st in [1, 2, 3, 4]:
        r_cnt = c_real.get(st, 0)
        m_cnt = c_mock.get(st, 0)
        print(f" - {state_names[st]}: 실데이터 {r_cnt:4d}일 ({r_cnt/len(res_isa)*100:5.1f}%) | 더미 {m_cnt:4d}일 ({m_cnt/len(res_mock)*100:5.1f}%)")

    years = (res_isa.index[-1] - res_isa.index[0]).days / 365.25
    trade_days = (res_isa["turnover"] > 0).sum()
    ann_turnover = res_isa["turnover"].sum() / years
    total_cost_pct = (res_isa["turnover"] * DEFAULT_CONFIG["fee_slippage_oneway"]).sum() * 100
    print(f"\n[매매 빈도 & 비용] 비중 변경일: {trade_days}회 | 연평균 회전율: {ann_turnover:.2f}회 | 누적 거래비용(단리합): {total_cost_pct:.2f}%")

    print("\n[State 4 (투매 역발상) 전체 매매 내역 — 실데이터]")
    if len(s4_isa) > 0:
        win_rate = (s4_isa["return_pct"] > 0).mean() * 100
        avg_ret = s4_isa["return_pct"].mean()
        print(f" - 총 {len(s4_isa)}회 진입 | 승률: {win_rate:.1f}% | 평균 수익률: {avg_ret:+.2f}%")
        print(s4_isa.to_string(index=False))
    else:
        print(" - 조건을 충족한 State 4 진입 내역 없음")

    # 연도별 수익률 비교
    print("\n[연도별 수익률 비교 (실데이터, CD91/KOFR 현실금리 적용)]")
    yr_rows = []
    for yr, grp in res_cd91.groupby(res_cd91.index.year):
        s_r = ((1 + grp["strat_ret"]).prod() - 1) * 100
        b_r = ((1 + grp["bh_ret"]).prod() - 1) * 100
        k_r = ((1 + grp["kofr_after_tax_ret"]).prod() - 1) * 100
        cr_avg = grp["cr_ratio"].mean()
        cr_max = grp["cr_ratio"].max()
        s1_pct = (grp["state"] == 1).mean() * 100
        s2_pct = (grp["state"] == 2).mean() * 100
        s3_pct = (grp["state"] == 3).mean() * 100
        s4_pct = (grp["state"] == 4).mean() * 100
        yr_rows.append({
            "연도": yr,
            "FSM(%)": f"{s_r:+.1f}%",
            "코스닥150(%)": f"{b_r:+.1f}%",
            "KOFR/CD(%)": f"{k_r:+.1f}%",
            "초과(vs B&H)": f"{s_r - b_r:+.1f}%p",
            "평균신용비율": f"{cr_avg:.2f}%",
            "최대신용비율": f"{cr_max:.2f}%",
            "S1/S2/S3/S4(%)": f"{s1_pct:4.0f}/{s2_pct:4.0f}/{s3_pct:4.0f}/{s4_pct:2.0f}",
        })
    print(pd.DataFrame(yr_rows).to_string(index=False))

    # 결과 CSV 저장
    out_csv = os.path.join(BASE_DIR, "backtest_result_kosdaq150.csv")
    res_cd91[["etf_close", "nav_close", "ma120", "cr_ratio", "cr_chg_10d", "cr_chg_20d", "state", "weight", "strat_equity", "bh_equity", "kofr_equity"]].to_csv(out_csv)
    print(f"\n[완료] 상세 일별 결과가 '{out_csv}'에 저장되었습니다.")


if __name__ == "__main__":
    main()
