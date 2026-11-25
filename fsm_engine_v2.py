import os
import warnings
import numpy as np
import pandas as pd
from backtest_kosdaq150_fsm import DEFAULT_CONFIG, _reconstruct_total_return_nav, load_market_data, calc_metrics

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_aligned_etf(csv_name: str, col_name: str, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(BASE_DIR, csv_name), parse_dates=["date"])
    df = _reconstruct_total_return_nav(raw).set_index("date").loc[start:end].copy()
    df[f"{col_name}_close"] = df["adj_close"]
    df[f"{col_name}_ret"] = df["adj_close"].pct_change().fillna(0.0)
    return df[[f"{col_name}_close", f"{col_name}_ret"]]


def run_fsm_generalized(
    df: pd.DataFrame,
    cfg: dict,
    state_allocs: dict,
    vol_target_ann: float = None,
    vol_window: int = 20,
    max_lev_cap: float = 2.0,
    tax_derivatives: bool = False,
):
    """
    일반화된 4단계 FSM 시뮬레이터:
    - state_allocs[state] = (w_1x, w_2x, w_inv, w_alt)
      나머지 비중(1 - sum)은 KOFR/CD91 현금으로 자동 배분.
    - vol_target_ann이 설정되면 State 1/2의 주식 익스포저를 (vol_target / realized_vol_20d)로 스케일링하되
      [0, max_lev_cap] 범위로 클립하고, 1.0 초과는 2x ETF(233740), 1.0 이하는 1x ETF(229200)로 구현.
    - 미래 참조 방지: 전일(t-1) 종가 기준 변동성과 신호로 오늘(t) 수익률 실현.
    """
    n = len(df)
    dates = df.index

    # 전일 종가 기준 20일 실현변동성 (연율화)
    rvol = df["etf_ret"].rolling(vol_window).std() * np.sqrt(252)

    states = np.zeros(n, dtype=int)
    strat_rets = np.zeros(n, dtype=float)
    turnovers = np.zeros(n, dtype=float)
    eff_exposures = np.zeros(n, dtype=float)

    cur_state = 1 if df["etf_close"].iloc[0] > df["ma120"].iloc[0] else 3
    prev_alloc = np.array([1.0, 0.0, 0.0, 0.0]) if cur_state == 1 else np.array([0.0, 0.0, 0.0, 0.0])
    s4_entry_price = None
    s4_cooldown = 0

    kofr_tax = 0.154 if cfg["account_type"].upper() == "GENERAL" else 0.0
    deriv_tax = 0.154 if (cfg["account_type"].upper() == "GENERAL" and tax_derivatives) else 0.0

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

        # (A) 전일 확정 비중으로 오늘 수익 실현
        if i > 0:
            r_1x = row["etf_ret"]
            r_2x = row.get("lev2x_ret", 0.0)
            r_inv = row.get("inv1x_ret", 0.0)
            r_alt = row.get("alt_ret", 0.0)
            kr = row["kofr_ret"]
            r_cash = kr * (1.0 - kofr_tax if kr > 0 else 1.0)

            w_cash = max(0.0, 1.0 - prev_alloc.sum())
            # 일반계좌에서 파생형(2x/인버스/해외ETF) 양의 수익 근사 세금 반영 옵션
            gross = (
                prev_alloc[0] * r_1x
                + prev_alloc[1] * (r_2x * (1.0 - deriv_tax) if r_2x > 0 else r_2x)
                + prev_alloc[2] * (r_inv * (1.0 - deriv_tax) if r_inv > 0 else r_inv)
                + prev_alloc[3] * (r_alt * (1.0 - deriv_tax) if r_alt > 0 else r_alt)
                + w_cash * r_cash
            )
        else:
            gross = 0.0

        # (B) 오늘 종가 기준 상태 전이
        cond_up = (row["above_streak"] >= cfg["confirm_days"]) or (disp >= cfg["breakout_pct"])
        cond_dn = (row["below_streak"] >= cfg["confirm_days"]) or (disp <= -cfg["breakout_pct"])
        cond_overheat = (cr_ratio >= cfg["cr_overheat_pct"]) or (cr_20d >= cfg["cr_20d_surge"])
        cond_calm = (cr_ratio <= cfg["cr_calm_pct"]) and (cr_20d < cfg["cr_20d_surge"] * 0.5)

        next_state = cur_state
        if cur_state == 4:
            trade_ret = (close / s4_entry_price) - 1.0
            if (trade_ret >= cfg["s4_take_profit"]) or (close >= ma120) or (trade_ret <= cfg["s4_stop_loss"]):
                s4_entry_price = None
                s4_cooldown = cfg["s4_cooldown_days"]
                next_state = 1 if (cond_up and not cond_overheat) else (2 if cond_up and cond_overheat else 3)
        elif cur_state == 1:
            if cond_dn:
                next_state = 3
            elif cond_overheat:
                next_state = 2
        elif cur_state == 2:
            if cond_dn:
                next_state = 3
            elif cond_calm:
                next_state = 1
        elif cur_state == 3:
            if cond_up:
                next_state = 2 if cond_overheat else 1
            elif s4_cooldown == 0:
                if disp <= cfg["s4_disparity"] and cr_10d <= cfg["s4_cr_10d_drop"] and close > ma5:
                    next_state = 4
                    s4_entry_price = close

        if next_state != 4 and cur_state != 4 and s4_cooldown > 0:
            s4_cooldown -= 1

        # (C) 목표 배분 결정
        if vol_target_ann is not None and next_state in (1, 2):
            rv = rvol.iloc[i]
            if np.isnan(rv) or rv <= 0.05:
                target_beta = 1.0 if next_state == 1 else 0.5
            else:
                cap = max_lev_cap if next_state == 1 else min(1.0, max_lev_cap * 0.5)
                target_beta = float(np.clip(vol_target_ann / rv, 0.25, cap))
            if target_beta <= 1.0:
                new_alloc = np.array([target_beta, 0.0, 0.0, 0.0])
            else:
                # 1x + 2x 조합으로 정확히 target_beta 구현: w1 + 2*w2 = beta, w1 + w2 = 1 -> w2 = beta - 1
                w2 = target_beta - 1.0
                w1 = 1.0 - w2
                new_alloc = np.array([w1, w2, 0.0, 0.0])
        else:
            new_alloc = np.array(state_allocs[next_state], dtype=float)

        # State 3 대체자산(예: 나스닥100)에 자체 추세필터(alt_trend_on) 적용
        if new_alloc[3] > 0 and "alt_trend_on" in df.columns:
            if not bool(row["alt_trend_on"]):
                new_alloc[3] = 0.0

        # 회전율: 4개 ETF + 현금 변동 합 / 2 (또는 편도 매수+매도 절대합의 절반 -> 기존 단일자산 기준과 통일)
        prev_cash = max(0.0, 1.0 - prev_alloc.sum())
        new_cash = max(0.0, 1.0 - new_alloc.sum())
        # 총 편도 거래량 = (sum(|delta_etf|) + |delta_cash|) / 2
        if i > 0:
            turnover = 0.5 * (np.abs(new_alloc - prev_alloc).sum() + abs(new_cash - prev_cash))
        else:
            turnover = 0.0
        # 실제 ETF 전환 시 매도+매수 양쪽 발생 -> 2*turnover가 편도 비용 적용 대상 (단일자산 1x<->KOFR은 prev->new 각 1번씩)
        # 기존 backtest_kosdaq150_fsm.py는 abs(new_w - prev_w) * 0.0008 (ETF+KOFR 전환 합쳐 8bp) 적용 -> 동일 기준 유지
        cost = turnover * cfg["fee_slippage_oneway"]

        strat_rets[i] = gross - cost
        states[i] = next_state
        turnovers[i] = turnover
        eff_exposures[i] = new_alloc[0] + 2.0 * new_alloc[1] - new_alloc[2] + new_alloc[3]

        cur_state = next_state
        prev_alloc = new_alloc

    res = df.copy()
    res["state"] = states
    res["strat_ret"] = strat_rets
    res["turnover"] = turnovers
    res["eff_exposure"] = eff_exposures
    res["strat_equity"] = cfg["initial_capital"] * (1.0 + res["strat_ret"]).cumprod()
    kofr_at = res["kofr_ret"].where(res["kofr_ret"] <= 0, res["kofr_ret"] * (1.0 - kofr_tax))
    kofr_at.iloc[0] = 0.0
    res["kofr_after_tax_ret"] = kofr_at
    return res
