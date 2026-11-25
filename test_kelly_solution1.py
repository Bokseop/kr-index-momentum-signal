import os
import warnings
import numpy as np
import pandas as pd
from signal_domestic_2index import ETF_META, STRATEGY_CONFIG, reconstruct_total_return_nav, compute_ma_state

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data():
    out = {}
    for k, meta in ETF_META.items():
        p = os.path.join(DATA_DIR, meta["csv"])
        raw = pd.read_csv(p, parse_dates=["date"])
        out[k] = reconstruct_total_return_nav(raw)
    return out


def sim_solution1_kelly(
    data: dict,
    ma_len: int = 100,
    mom_days: int = 120,
    kelly_frac: float = None,   # None = 고정 100%, 1.0 = 풀켈리, 0.5 = 하프켈리, 0.25 = 쿼터켈리
    mu_win: int = 120,
    vol_win: int = 60,
    rebal_band: float = 0.10,   # ±10%p 이상 변동 시만 비중 조정 (과매매 방지)
    min_weight: float = 0.0,
    max_weight: float = 1.0,    # 노레버리지 1x 상한
):
    idx = data["k200"].index.intersection(data["kq"].index).sort_values()
    k200_c = data["k200"]["adj_close"].loc[idx]
    kq_c = data["kq"]["adj_close"].loc[idx]
    r_k2 = data["k200"]["total_ret"].loc[idx]
    r_kq = data["kq"]["total_ret"].loc[idx]

    m_k200 = k200_c.pct_change(mom_days)
    m_kq = kq_c.pct_change(mom_days)
    st_k200 = compute_ma_state(k200_c, ma_len, 3, 0.01)
    st_kq = compute_ma_state(kq_c, ma_len, 3, 0.01)

    kofr_ret = data["kofr"]["total_ret"].reindex(idx)
    cd91_daily = (1.0 + data["kq"]["cd91"].reindex(idx).ffill().bfill() / 100.0) ** (1.0 / 252.0) - 1.0
    kofr_ret = kofr_ret.fillna(cd91_daily)

    # 연속 켈리 공식: f* = (연율화 초과수익률) / (연율화 분산) — 엄격히 과거(t까지) 데이터만 사용
    # 연율화 초과수익률: (r_asset - r_kofr).rolling(mu_win).mean() * 252
    # 연율화 분산: r_asset.rolling(vol_win).var() * 252
    ex_k2 = (r_k2 - kofr_ret).rolling(mu_win).mean() * 252.0
    var_k2 = r_k2.rolling(vol_win).var() * 252.0
    raw_kelly_k2 = (ex_k2 / var_k2.replace(0, np.nan)).fillna(0.0)

    ex_kq = (r_kq - kofr_ret).rolling(mu_win).mean() * 252.0
    var_kq = r_kq.rolling(vol_win).var() * 252.0
    raw_kelly_kq = (ex_kq / var_kq.replace(0, np.nan)).fillna(0.0)

    bt_idx = idx[idx >= pd.Timestamp(STRATEGY_CONFIG["bt_start"])]
    n = len(bt_idx)

    is_me = np.zeros(n, dtype=bool)
    is_me[0] = True
    for i in range(n - 1):
        if bt_idx[i].month != bt_idx[i + 1].month:
            is_me[i] = True

    fee = STRATEGY_CONFIG["fee_oneway"]
    tax = STRATEGY_CONFIG["tax_rate"]
    r_k2_at = data["k200"].loc[bt_idx, "total_ret"] - tax * data["k200"].loc[bt_idx, "div_ret"]
    r_kq_at = data["kq"].loc[bt_idx, "total_ret"] - tax * data["kq"].loc[bt_idx, "div_ret"]
    r_kofr_at = kofr_ret.loc[bt_idx] * (1.0 - tax)

    cur_asset = "kofr"
    cur_w = 0.0  # equity weight in [0, 1]
    rets = np.zeros(n)
    eq = np.zeros(n)
    eq[0] = 10_000_000.0
    weights_hist = np.zeros(n)
    turns = 0.0

    for i in range(n):
        d = bt_idx[i]
        # 1. 전일 확정 비중으로 오늘 수익 실현
        if i > 0:
            if cur_asset == "k200":
                gross = cur_w * float(r_k2_at.iloc[i]) + (1.0 - cur_w) * float(r_kofr_at.iloc[i])
            elif cur_asset == "kq":
                gross = cur_w * float(r_kq_at.iloc[i]) + (1.0 - cur_w) * float(r_kofr_at.iloc[i])
            else:
                gross = float(r_kofr_at.iloc[i])
        else:
            gross = 0.0

        # 2. 오늘 종가 기준 종목 선정 & 일간 MA 방어
        mk2, mkq = m_k200.loc[d], m_kq.loc[d]
        on_k2, on_kq = int(st_k200.loc[d, "trend_on"]), int(st_kq.loc[d, "trend_on"])

        prev_asset, prev_w = cur_asset, cur_w
        if is_me[i]:
            cands = []
            if not np.isnan(mk2) and mk2 > 0 and on_k2 == 1:
                cands.append((mk2, "k200"))
            if not np.isnan(mkq) and mkq > 0 and on_kq == 1:
                cands.append((mkq, "kq"))
            cands.sort(reverse=True)
            new_asset = cands[0][1] if cands else "kofr"
        else:
            new_asset = cur_asset
            if cur_asset == "k200" and on_k2 == 0:
                new_asset = "kofr"
            elif cur_asset == "kq" and on_kq == 0:
                new_asset = "kofr"

        # 3. 켈리 포지션 사이징 계산
        if new_asset == "kofr":
            new_w = 0.0
        elif kelly_frac is None:
            new_w = 1.0
        else:
            rk = float(raw_kelly_k2.loc[d]) if new_asset == "k200" else float(raw_kelly_kq.loc[d])
            target_w = float(np.clip(rk * kelly_frac, min_weight, max_weight))
            # 같은 자산 보유 중이면 리밸런싱 밴드(±10%p) 적용해 불필요한 소액 매매 방지
            if new_asset == prev_asset and not is_me[i] and abs(target_w - prev_w) < rebal_band:
                new_w = prev_w
            else:
                # 실전 편의상 5% 단위 반올림
                new_w = round(target_w * 20) / 20.0

        # 회전율 계산
        if i > 0:
            if new_asset == prev_asset:
                t = abs(new_w - prev_w)
            else:
                # 기존 주식 전량 매도 + 신규 주식 매수 -> 편도 회전율 = max(prev_w, new_w) + ... 정확히는 (prev_w + new_w + |cash_diff|)/2
                t = 0.5 * (prev_w + new_w + abs((1.0 - new_w) - (1.0 - prev_w)))
        else:
            t = 0.0

        turns += t
        rets[i] = gross - t * fee
        if i > 0:
            eq[i] = eq[i - 1] * (1.0 + rets[i])
        weights_hist[i] = new_w
        cur_asset, cur_w = new_asset, new_w

    eq_s = pd.Series(eq, index=bt_idx)
    ret_s = pd.Series(rets, index=bt_idx)
    years = (bt_idx[-1] - bt_idx[0]).days / 365.25
    cagr = (eq_s.iloc[-1] / eq_s.iloc[0]) ** (1.0 / years) - 1.0
    mdd = (eq_s / eq_s.cummax() - 1.0).min()
    vol = ret_s.iloc[1:].std() * np.sqrt(252)
    ex = ret_s.iloc[1:] - r_kofr_at.iloc[1:]
    ex_sharpe = (ex.mean() * 252) / (ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    # Pre-2025 (2016~2024)
    r_p25 = ret_s.loc[:"2024-12-31"]
    eq_p25 = (1 + r_p25).cumprod()
    y_p25 = (r_p25.index[-1] - r_p25.index[0]).days / 365.25
    cagr_p25 = eq_p25.iloc[-1] ** (1.0 / y_p25) - 1.0
    mdd_p25 = (eq_p25 / eq_p25.cummax() - 1.0).min()

    return {
        "cagr": cagr * 100,
        "mdd": mdd * 100,
        "vol": vol * 100,
        "ex_sharpe": ex_sharpe,
        "calmar": calmar,
        "p25_cagr": cagr_p25 * 100,
        "p25_mdd": mdd_p25 * 100,
        "avg_w": weights_hist.mean() * 100,
        "ann_turn": turns / years,
        "final_man": eq_s.iloc[-1] / 1e4,
    }


def main():
    data = load_data()
    print("=" * 125)
    print(" [해법 1 + 켈리 베팅(연속 켈리 f* = (μ-rf)/σ², 노레버리지 0~100%) 실데이터 백테스트 비교 (2016-06 ~ 2026-11, 일반계좌 세후)]")
    print("=" * 125)
    rows = []
    specs = [
        ("1. 기본 해법1 (MA100, 고정 100%)", 100, None, 120, 60),
        ("2. MA100 + 풀켈리 (1.0x f*, μ120/σ60)", 100, 1.0, 120, 60),
        ("3. MA100 + 하프켈리 (0.5x f*, μ120/σ60)", 100, 0.5, 120, 60),
        ("4. MA100 + 하프켈리 (0.5x f*, μ120/σ120)", 100, 0.5, 120, 120),
        ("5. MA100 + 하프켈리 (0.5x f*, μ252/σ60)", 100, 0.5, 252, 60),
        ("6. MA100 + 쿼터켈리 (0.25x f*, μ120/σ60)", 100, 0.25, 120, 60),
        ("7. 기본 해법1 (MA80, 고정 100%)", 80, None, 120, 60),
        ("8. MA80 + 하프켈리 (0.5x f*, μ120/σ60)", 80, 0.5, 120, 60),
        ("9. MA80 + 풀켈리 (1.0x f*, μ120/σ60)", 80, 1.0, 120, 60),
        ("10. 기본 해법1 (MA60, 고정 100%)", 60, None, 120, 60),
        ("11. MA60 + 하프켈리 (0.5x f*, μ120/σ60)", 60, 0.5, 120, 60),
        ("12. MA60 + 풀켈리 (1.0x f*, μ120/σ60)", 60, 1.0, 120, 60),
    ]
    for name, ma, kf, mw, vw in specs:
        m = sim_solution1_kelly(data, ma_len=ma, kelly_frac=kf, mu_win=mw, vol_win=vw)
        rows.append({
            "전략": name,
            "세후자산(만)": f"{m['final_man']:,.0f}",
            "세후CAGR": f"{m['cagr']:+.2f}%",
            "MDD": f"{m['mdd']:.1f}%",
            "Vol": f"{m['vol']:.1f}%",
            "초과샤프": f"{m['ex_sharpe']:.2f}",
            "칼마": f"{m['calmar']:.2f}",
            "16~24(8.5년)": f"{m['p25_cagr']:+.1f}%/{m['p25_mdd']:.0f}%",
            "평균비중": f"{m['avg_w']:.1f}%",
            "연회전": f"{m['ann_turn']:.1f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
