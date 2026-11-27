import os
import ssl
import json
import time
import datetime as dt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter

urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ==============================================================================
# 옵션 A (장기 풀사이클): 일반계좌 · 노레버리지(1x) · 국내 2대 지수 추세 + 코스닥150 단타 스윙
# - 검증 1: 2011 ~ 2024 (14.0년 · 박스피 6년 포함 · 25~26 폭등 제외)
# - 검증 2: 2011 ~ 2026 (15.9년 · 박스피 + 25~26 풀사이클 전체)
# ==============================================================================
ETF_META = {
    "k200": {
        "name": "KODEX 200",
        "ticker": "069500",
        "kofia_cd": "KR5105352888",
        "csv": "kodex_200_nav.csv",
        "start_year": 2009,
        "taxable_capgain": False,
    },
    "kq": {
        "name": "KODEX 코스닥150",
        "ticker": "229200",
        "kofia_cd": "K55105B74603",
        "csv": "kodex_kosdaq150_nav.csv",
        "start_year": 2015,
        "taxable_capgain": False,
    },
    "kofr": {
        "name": "KODEX KOFR금리액티브",
        "ticker": "423160",
        "kofia_cd": "K55105DR9373",
        "csv": "kodex_kofr_nav.csv",
        "start_year": 2022,
        "taxable_capgain": True,
    },
}

STRATEGY_CONFIG = {
    "mom_days": 120,           # 6개월(120영업일) 모멘텀
    "ma_options": [100, 80, 60],
    "kelly_options": [
        {"id": "full",    "label": "풀 켈리 (1.0x)",    "frac": 1.00, "desc": "축소추정 켈리 f* 100% 적용 (고수익·고샤프 권장)"},
        {"id": "half",    "label": "하프 켈리 (0.5x)",  "frac": 0.50, "desc": "축소추정 켈리 f* 50% 적용 (안정형 · MDD 대폭 완화)"},
        {"id": "quarter", "label": "쿼터 켈리 (0.25x)", "frac": 0.25, "desc": "축소추정 켈리 f* 25% 적용 (초방어형 · MDD 10%대)"},
        {"id": "fixed",   "label": "고정 100%",        "frac": None, "desc": "선정 지수에 100% 고정 투자 (최대 공격형)"},
    ],
    "swing_options": [
        {
            "id": "overlay25",
            "label": "유휴현금 25% 스윙 (공격 권장)",
            "short_label": "유휴25% 스윙",
            "cap": 0.25,
            "carve_out": False,
            "desc": "추세 엔진 100% 유지 + 남는 KOFR 현금에서 최대 25%만 코스닥150 RSI(5)<20 1주일 분할 단타 (1차 12.5% → RSI<15 시 25%)",
        },
        {
            "id": "split75",
            "label": "계좌 분리 75:25 (박스피 방어 권장)",
            "short_label": "분리 75:25",
            "cap": 0.25,
            "carve_out": True,
            "desc": "추세 엔진 75% + 코스닥150 1주일 분할 단타 25% 분리 운용 (2011~16 박스피 방어 · MDD -10%p 축소)",
        },
        {
            "id": "pair60",
            "label": "차별화 페어 롱숏 60/40 (CAGR·MDD 최강)",
            "short_label": "페어 롱숏 60/40",
            "cap": 0.25,
            "carve_out": False,
            "pair_ls": True,
            "pair_long_cap": 0.60,
            "pair_short_cap": 0.40,
            "desc": "차별화 장세(한쪽 추세ON + 반대쪽 MA OFF & 6M<0%)일 때만 강한 지수 60% 롱 + 반대 인버스(114800/251340) 40% 숏 동시 보유 (하락장 맨몸 숏 금지 · 인버스 차익 15.4% 세후 반영)",
        },
        {
            "id": "none",
            "label": "스윙 미사용 (순수 추세)",
            "short_label": "순수 추세",
            "cap": 0.0,
            "carve_out": False,
            "desc": "기존 국내 2지수 MA 추세 + 켈리 전략만 단독 실행 (비교 기준)",
        },
    ],
    "inv_drag_ann": 0.0080,    # 인버스 1x ETF 연 보수·선물 롤오버 보수적 차감 (0.80%/yr)
    "swing_rsi_len": 5,
    "swing_entry_1": 20.0,
    "swing_entry_2": 15.0,
    "swing_exit_rsi": 50.0,
    "swing_max_hold": 5,
    "kelly_mu_win": 120,
    "kelly_erp_prior": 0.07,
    "kelly_mu_cap": 0.14,
    "kelly_vol_fast": 20,
    "kelly_vol_win": 60,
    "kelly_rebal_band": 0.10,
    "confirm_days": 3,
    "breakout_pct": 0.01,
    "fee_oneway": 0.0005,      # 실전 국내 ETF 편도 5bp (거래세 0% + 수수료·호가1틱)
    "tax_rate": 0.154,         # 일반계좌 배당/이자소득세 15.4%
    "bt_start": "2011-01-03",  # 2011년 첫 거래일 시작 (2009~2010 2년 워밍업)
    "bt_mid_end": "2024-12-31",
    "default_strategy_id": "ma100_full_overlay25",
}


class LegacySSLAdapter(HTTPAdapter):
    """금융투자협회(KOFIA) 레거시 DH 키 호환용 SSL 어댑터."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        try:
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        except ssl.SSLError:
            pass
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def get_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", LegacySSLAdapter())
    return s


def fetch_dis_nav_range(s: requests.Session, standard_cd: str, start_year: int, end_year: int) -> pd.DataFrame:
    url = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/"
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://dis.kofia.or.kr/",
    }
    rows = []
    y = start_year
    while y <= end_year:
        y_end = min(y + 1, end_year)
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<message><proframeHeader>"
            "<pfmAppName>FS-DIS2</pfmAppName>"
            "<pfmSvcName>DISFundStdPrcStutSO</pfmSvcName>"
            "<pfmFnName>select</pfmFnName>"
            "</proframeHeader><systemHeader></systemHeader>"
            "<DISCondFuncDTO>"
            f"<tmpV30>{y}0101</tmpV30>"
            f"<tmpV31>{y_end}1231</tmpV31>"
            "<tmpV10>0</tmpV10>"
            f"<tmpV12>{standard_cd}</tmpV12>"
            "</DISCondFuncDTO></message>"
        )
        r = s.post(url, data=xml.encode("utf-8"), headers=headers, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for it in root.findall(".//selectMeta"):
            d = {c.tag: c.text for c in it}
            if d.get("tmpV1") and d.get("tmpV2"):
                rows.append({
                    "date": d["tmpV1"],
                    "nav": float(d["tmpV2"]),
                    "diff": float(d["tmpV3"]) if d.get("tmpV3") not in (None, "", "-") else np.nan,
                    "tax_base": float(d["tmpV4"]) if d.get("tmpV4") not in (None, "", "-") else np.nan,
                    "cd91": float(d["tmpV10"]) if d.get("tmpV10") not in (None, "", "-") else np.nan,
                })
        y = y_end + 1
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def update_csv_data(s: requests.Session, meta: dict) -> pd.DataFrame:
    data_path = os.path.join(DATA_DIR, meta["csv"])
    root_path = os.path.join(BASE_DIR, meta["csv"])
    cur_year = dt.datetime.now().year

    existing = None
    if os.path.exists(data_path):
        existing = pd.read_csv(data_path, parse_dates=["date"])
    elif os.path.exists(root_path):
        existing = pd.read_csv(root_path, parse_dates=["date"])

    try:
        if existing is not None and len(existing) > 100:
            fetch_from = max(meta["start_year"], cur_year - 1)
            fresh = fetch_dis_nav_range(s, meta["kofia_cd"], fetch_from, cur_year)
            if len(fresh) > 0:
                cols = [c for c in ["date", "nav", "diff", "tax_base", "cd91"] if c in existing.columns]
                merged = pd.concat([existing[cols], fresh[cols]], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                merged.to_csv(data_path, index=False)
                return merged
            existing.to_csv(data_path, index=False)
            return existing
        else:
            full = fetch_dis_nav_range(s, meta["kofia_cd"], meta["start_year"], cur_year)
            full.to_csv(data_path, index=False)
            return full
    except Exception as e:
        print(f"[경고] {meta['name']} 온라인 갱신 실패({e}) -> 로컬 캐시 사용")
        if existing is not None:
            return existing
        raise


def reconstruct_total_return_nav(raw_df: pd.DataFrame) -> pd.DataFrame:
    d = raw_df.sort_values("date").drop_duplicates("date").reset_index(drop=True).copy()
    raw_diff = d["nav"].diff()

    early = d.iloc[1:min(80, len(d))]
    early_raw = raw_diff.iloc[1:min(80, len(d))]
    mask = (early_raw.abs() > d["nav"].iloc[0] * 0.003) & early["diff"].notna() & (early["diff"] * early_raw > 0)
    if mask.sum() >= 3:
        f0 = float(np.median((early.loc[mask, "diff"] / early_raw.loc[mask]).values))
        if abs(f0 - 1.0) < 0.002:
            f0 = 1.0
    else:
        f0 = 1.0

    adj = np.empty(len(d), dtype=float)
    adj[0] = d["nav"].iloc[0] * f0
    for i in range(1, len(d)):
        diff_val = d["diff"].iloc[i]
        raw_chg = d["nav"].iloc[i] - d["nav"].iloc[i - 1]
        if i == len(d) - 1 and diff_val == 0.0 and abs(raw_chg) > 1e-6:
            adj[i] = d["nav"].iloc[i] * (adj[i - 1] / d["nav"].iloc[i - 1])
        else:
            adj[i] = adj[i - 1] + (0.0 if np.isnan(diff_val) else diff_val)
    d["adj_nav"] = adj

    aligned = pd.DataFrame({
        "date": d["date"].iloc[:-1].values,
        "nav_close": d["nav"].iloc[1:].values,
        "adj_close": d["adj_nav"].iloc[1:].values,
    })
    if "cd91" in d.columns:
        aligned["cd91"] = d["cd91"].iloc[:-1].replace(0.0, np.nan).ffill().values
    aligned = aligned.set_index("date")
    aligned["price_ret"] = aligned["nav_close"].pct_change().fillna(0.0)
    aligned["total_ret"] = aligned["adj_close"].pct_change().fillna(0.0)
    aligned["div_ret"] = np.maximum(0.0, aligned["total_ret"] - aligned["price_ret"])
    return aligned


def splice_kosdaq_2009(kq_etf_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    2009-01 ~ 2015-09-29: 금융투자협회 FreeSIS 공식 코스닥지수 대용 (분배금 0 가정)
    2015-09-30 ~ 최신일 : 공식 KODEX 코스닥150 ETF 수정기준가 실데이터
    접합일 가격 연속 스케일링으로 MA·모멘텀·RSI 단절 제거.
    """
    pre_path = os.path.join(DATA_DIR, "freesis_kosdaq_2009.csv")
    if not os.path.exists(pre_path):
        return kq_etf_aligned

    kq_old = pd.read_csv(pre_path, parse_dates=["date"]).sort_values("date").drop_duplicates("date")
    first_etf_dt = kq_etf_aligned.index[0]
    first_etf_adj = float(kq_etf_aligned["adj_close"].iloc[0])
    first_etf_nav = float(kq_etf_aligned["nav_close"].iloc[0])

    pre = kq_old[kq_old["date"] < first_etf_dt].set_index("date").sort_index().copy()
    anchor_idx = float(kq_old[kq_old["date"] <= first_etf_dt]["kosdaq_index"].iloc[-1])
    scale_adj = first_etf_adj / anchor_idx
    scale_nav = first_etf_nav / anchor_idx

    pre_df = pd.DataFrame({
        "nav_close": pre["kosdaq_index"] * scale_nav,
        "adj_close": pre["kosdaq_index"] * scale_adj,
    }, index=pre.index)

    combined = pd.concat([pre_df, kq_etf_aligned[["nav_close", "adj_close"]]])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined["price_ret"] = combined["nav_close"].pct_change().fillna(0.0)
    combined["total_ret"] = combined["adj_close"].pct_change().fillna(0.0)
    combined["div_ret"] = np.maximum(0.0, combined["total_ret"] - combined["price_ret"])
    return combined


def compute_ma_state(close: pd.Series, ma_len: int, confirm_days: int = 3, brk: float = 0.01) -> pd.DataFrame:
    ma = close.rolling(ma_len).mean()
    disp = close / ma - 1.0
    above = (close > ma).astype(int)
    below = (close < ma).astype(int)
    ab_s = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    bl_s = below.groupby((below != below.shift()).cumsum()).cumsum() * below

    sig = np.zeros(len(close), dtype=int)
    cur = 1 if (not np.isnan(ma.iloc[0]) and close.iloc[0] > ma.iloc[0]) else 0
    for i in range(len(close)):
        if np.isnan(ma.iloc[i]):
            sig[i] = 0
            continue
        if i > 0 and np.isnan(ma.iloc[i - 1]):
            cur = 1 if close.iloc[i] > ma.iloc[i] else 0
        up = (ab_s.iloc[i] >= confirm_days) or (disp.iloc[i] >= brk)
        dn = (bl_s.iloc[i] >= confirm_days) or (disp.iloc[i] <= -brk)
        if cur == 1 and dn:
            cur = 0
        elif cur == 0 and up:
            cur = 1
        sig[i] = cur

    return pd.DataFrame({
        "ma": ma,
        "disp": disp,
        "above_streak": ab_s,
        "below_streak": bl_s,
        "trend_on": sig,
    }, index=close.index)


def compute_rsi(close: pd.Series, period: int = 5) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def summarize_trades(tr_list: list) -> dict:
    tr_arr = np.array(tr_list) if tr_list else np.array([])
    if len(tr_arr) == 0:
        return {"trades": 0, "win_rate_pct": 0.0, "avg_ret_pct": 0.0, "profit_factor": 0.0, "kelly_f_pct": 0.0}
    wins = tr_arr[tr_arr > 0]
    losses = tr_arr[tr_arr <= 0]
    win_rate = len(wins) / len(tr_arr)
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else 9.99
    kelly_f = (win_rate - (1.0 - win_rate) / (avg_win / avg_loss)) if (avg_win > 0 and avg_loss > 0) else 0.0
    return {
        "trades": int(len(tr_arr)),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_ret_pct": round(float(tr_arr.mean()) * 100, 2),
        "profit_factor": round(pf, 2),
        "kelly_f_pct": round(kelly_f * 100, 1),
    }


def compute_kq_swing_signal(kq_close: pd.Series, kq_ret_at: pd.Series, bt_idx: pd.DatetimeIndex):
    """
    코스닥150 극단 과매도 1주일 분할 스윙 엔진 (미래참조 0, 2011~2024 및 2011~2026 통계 분리 집계).
    """
    rsi_s = compute_rsi(kq_close, STRATEGY_CONFIG["swing_rsi_len"])
    ent1 = STRATEGY_CONFIG["swing_entry_1"]
    ent2 = STRATEGY_CONFIG["swing_entry_2"]
    ext_rsi = STRATEGY_CONFIG["swing_exit_rsi"]
    max_hold = STRATEGY_CONFIG["swing_max_hold"]
    fee = STRATEGY_CONFIG["fee_oneway"]
    mid_end = pd.Timestamp(STRATEGY_CONFIG["bt_mid_end"])

    n = len(bt_idx)
    swing_frac = np.zeros(n)
    swing_stage = np.zeros(n, dtype=int)
    swing_days = np.zeros(n, dtype=int)
    swing_reason = []
    entry_px_hist = np.zeros(n)

    in_pos = False
    cur_f = 0.0
    cur_st = 0
    hold_d = 0
    ent_px = 0.0

    trades_all = []
    trades_14 = []
    tr_active = False
    tr_eq = 1.0
    prev_f = 0.0

    for i, d in enumerate(bt_idx):
        px = float(kq_close.loc[d])
        rv = float(rsi_s.loc[d])
        r_day = float(kq_ret_at.iloc[i])

        if not in_pos:
            if rv < ent1:
                in_pos = True
                ent_px = px
                hold_d = 0
                if rv < ent2:
                    cur_f, cur_st = 1.0, 2
                    rsn = f"코스닥150 극단 과매도 RSI(5)={rv:.1f}<{ent2:.0f} → 1·2차 동시 진입"
                else:
                    cur_f, cur_st = 0.5, 1
                    rsn = f"코스닥150 과매도 RSI(5)={rv:.1f}<{ent1:.0f} → 1차 분할 진입"
            else:
                cur_f, cur_st = 0.0, 0
                rsn = f"코스닥150 스윙 대기 (RSI(5)={rv:.1f})"
        else:
            hold_d += 1
            if rv >= ext_rsi:
                in_pos = False
                cur_f, cur_st = 0.0, 0
                rsn = f"코스닥150 RSI(5)={rv:.1f}≥{ext_rsi:.0f} 반등 익절 청산 ({hold_d}일 보유)"
                hold_d = 0
                ent_px = 0.0
            elif hold_d >= max_hold:
                in_pos = False
                cur_f, cur_st = 0.0, 0
                rsn = f"코스닥150 1주일({max_hold}영업일) 만기 자동 청산 (RSI={rv:.1f})"
                hold_d = 0
                ent_px = 0.0
            else:
                if cur_st == 1 and rv < ent2:
                    cur_f, cur_st = 1.0, 2
                    rsn = f"코스닥150 추가 급락 RSI(5)={rv:.1f}<{ent2:.0f} → 2차 분할 증액 ({hold_d}/{max_hold}일)"
                else:
                    rsn = f"코스닥150 스윙 {cur_st}차 보유 중 ({hold_d}/{max_hold}일, RSI={rv:.1f})"

        swing_frac[i] = cur_f
        swing_stage[i] = cur_st
        swing_days[i] = hold_d
        entry_px_hist[i] = ent_px
        swing_reason.append(rsn)

        if i > 0:
            if not tr_active and prev_f > 0:
                tr_active = True
                tr_eq = 1.0 + prev_f * r_day - prev_f * fee
            elif tr_active:
                if prev_f > 0:
                    tr_eq *= (1.0 + prev_f * r_day)
                else:
                    tr_eq *= (1.0 - fee)
                    trades_all.append(tr_eq - 1.0)
                    if d <= mid_end:
                        trades_14.append(tr_eq - 1.0)
                    tr_active = False
        prev_f = cur_f

    if tr_active:
        trades_all.append(tr_eq - 1.0)
        if bt_idx[-1] <= mid_end:
            trades_14.append(tr_eq - 1.0)

    ld = bt_idx[-1]
    cur_rsi = float(rsi_s.loc[ld])
    live_pnl = ((float(kq_close.loc[ld]) / entry_px_hist[-1]) - 1.0) * 100.0 if entry_px_hist[-1] > 0 else 0.0

    live_swing = {
        "rsi5": round(cur_rsi, 1),
        "stage": int(swing_stage[-1]),
        "frac_of_cap": float(swing_frac[-1]),
        "hold_days": int(swing_days[-1]),
        "max_hold_days": max_hold,
        "remain_days": max(0, max_hold - int(swing_days[-1])) if swing_stage[-1] > 0 else 0,
        "open_pnl_pct": round(live_pnl, 2),
        "status_text": swing_reason[-1],
        "stats_11_24": summarize_trades(trades_14),
        "stats_11_26": summarize_trades(trades_all),
    }
    return {
        "rsi5_series": rsi_s,
        "frac": swing_frac,
        "stage": swing_stage,
        "days": swing_days,
        "reasons": swing_reason,
        "live": live_swing,
    }


def fmt_pos_label(w_k2: float, w_kq: float, w_k2s: float = 0.0, w_kqs: float = 0.0) -> str:
    p2 = round(w_k2 * 100, 1)
    pq = round(w_kq * 100, 1)
    p2s = round(w_k2s * 100, 1)
    pqs = round(w_kqs * 100, 1)
    pc = round(max(0.0, 100.0 - p2 - pq - p2s - pqs), 1)
    def fp(v):
        return f"{int(v)}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"
    parts = []
    if p2 >= 0.5:
        parts.append(f"KODEX 200 {fp(p2)}")
    if pq >= 0.5:
        parts.append(f"코스닥150 {fp(pq)}")
    if p2s >= 0.5:
        parts.append(f"KODEX인버스 {fp(p2s)}")
    if pqs >= 0.5:
        parts.append(f"코스닥인버스 {fp(pqs)}")
    if pc >= 0.5 or not parts:
        parts.append(f"KOFR {fp(pc)}")
    return " + ".join(parts)


def calc_period_metrics(eq_sub: pd.Series, ret_sub: pd.Series, rf_sub: pd.Series, w_sub: pd.Series, turn_sum: float, n_events: int) -> dict:
    yrs = (eq_sub.index[-1] - eq_sub.index[0]).days / 365.25
    cagr = (eq_sub.iloc[-1] / eq_sub.iloc[0]) ** (1.0 / yrs) - 1.0
    mdd = (eq_sub / eq_sub.cummax() - 1.0).min()
    vol = ret_sub.iloc[1:].std() * np.sqrt(252.0)
    ex = ret_sub.iloc[1:] - rf_sub.iloc[1:]
    ex_sharpe = (ex.mean() * 252.0) / (ex.std() * np.sqrt(252.0)) if ex.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    return {
        "bt_start": eq_sub.index[0].strftime("%Y-%m-%d"),
        "bt_end": eq_sub.index[-1].strftime("%Y-%m-%d"),
        "years": round(yrs, 2),
        "final_equity_man": round(float(eq_sub.iloc[-1]) / 1e4, 1),
        "cagr_pct": round(cagr * 100, 2),
        "mdd_pct": round(mdd * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "ex_sharpe": round(ex_sharpe, 2),
        "calmar": round(calmar, 2),
        "avg_weight_pct": round(float(w_sub.mean()) * 100, 1),
        "ann_turnover": round(turn_sum / yrs, 2),
        "total_switches": int(n_events),
    }


def sub_cagr_mdd(ret_s: pd.Series, s_dt: str, e_dt: str):
    sub = ret_s.loc[s_dt:e_dt]
    if len(sub) < 20:
        return 0.0, 0.0
    eq = (1.0 + sub).cumprod()
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    return round(((eq.iloc[-1] ** (1.0 / yrs)) - 1.0) * 100, 1), round(((eq / eq.cummax()) - 1.0).min() * 100, 1)


def run_strategy_variant(data: dict, ma_len: int, kelly_opt: dict, swing_opt: dict, swing_bundle: dict):
    mom_days = STRATEGY_CONFIG["mom_days"]
    mu_win = STRATEGY_CONFIG["kelly_mu_win"]
    vol_win = STRATEGY_CONFIG["kelly_vol_win"]
    rebal_band = STRATEGY_CONFIG["kelly_rebal_band"]
    kelly_frac = kelly_opt["frac"]
    sw_cap = swing_opt["cap"]
    sw_carve = swing_opt["carve_out"]

    idx = data["k200"].index.intersection(data["kq"].index).sort_values()
    k200_c = data["k200"]["adj_close"].loc[idx]
    kq_c = data["kq"]["adj_close"].loc[idx]
    r_k2 = data["k200"]["total_ret"].loc[idx]
    r_kq = data["kq"]["total_ret"].loc[idx]

    m_k200 = k200_c.pct_change(mom_days)
    m_kq = kq_c.pct_change(mom_days)
    st_k200 = compute_ma_state(k200_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])
    st_kq = compute_ma_state(kq_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])

    # 2009~2022.04: KODEX 200 공식 CD91 금리 / 2022.04~현재: 공식 KOFR ETF 실수익률
    kofr_ret = data["kofr"]["total_ret"].reindex(idx)
    cd91_s = data["k200"]["cd91"].reindex(idx).replace(0.0, np.nan).ffill().bfill()
    cd91_daily = (1.0 + cd91_s / 100.0) ** (1.0 / 252.0) - 1.0
    kofr_ret = kofr_ret.fillna(cd91_daily)

    prior = STRATEGY_CONFIG["kelly_erp_prior"]
    mu_cap = STRATEGY_CONFIG["kelly_mu_cap"]
    vol_fast = STRATEGY_CONFIG["kelly_vol_fast"]

    def compute_shrinkage_kelly(r_s, rf_s):
        long_mu = ((r_s - rf_s).expanding(min_periods=120).mean() * 252.0).clip(0.0, mu_cap)
        short_mu = ((r_s - rf_s).rolling(mu_win).mean() * 252.0).clip(0.0, mu_cap)
        mu_shrunk = (0.50 * prior + 0.25 * long_mu + 0.25 * short_mu).fillna(prior)
        v_fast = r_s.rolling(vol_fast).std() * np.sqrt(252.0)
        v_slow = r_s.rolling(vol_win).std() * np.sqrt(252.0)
        v_cons = np.maximum(v_fast, v_slow)
        fstar = (mu_shrunk / (v_cons ** 2).replace(0, np.nan)).fillna(0.0)
        return mu_shrunk, v_cons, fstar

    mu_k2, vol_k2, fstar_k2 = compute_shrinkage_kelly(r_k2, kofr_ret)
    mu_kq, vol_kq, fstar_kq = compute_shrinkage_kelly(r_kq, kofr_ret)

    bt_idx = idx[idx >= pd.Timestamp(STRATEGY_CONFIG["bt_start"])]
    mid_end_ts = pd.Timestamp(STRATEGY_CONFIG["bt_mid_end"])
    n = len(bt_idx)
    idx_2024_end = int(np.where(bt_idx <= mid_end_ts)[0][-1])

    is_me = np.zeros(n, dtype=bool)
    is_me[0] = True
    for i in range(n - 1):
        if bt_idx[i].month != bt_idx[i + 1].month:
            is_me[i] = True
    last_dt = bt_idx[-1]
    if (last_dt + pd.offsets.BDay(1)).month != last_dt.month:
        is_me[-1] = True

    fee = STRATEGY_CONFIG["fee_oneway"]
    tax = STRATEGY_CONFIG["tax_rate"]
    inv_drag = STRATEGY_CONFIG["inv_drag_ann"] / 252.0
    pair_ls = bool(swing_opt.get("pair_ls", False))
    pair_long_cap = float(swing_opt.get("pair_long_cap", 0.60))
    pair_short_cap = float(swing_opt.get("pair_short_cap", 0.40))

    r_k2_at = data["k200"].loc[bt_idx, "total_ret"] - tax * data["k200"].loc[bt_idx, "div_ret"]
    r_kq_at = data["kq"].loc[bt_idx, "total_ret"] - tax * data["kq"].loc[bt_idx, "div_ret"]
    r_kofr_at = kofr_ret.loc[bt_idx] * (1.0 - tax)
    # 1x 인버스 ETF 일간 세전 수익률 (매도 시 실현차익에 15.4% 배당소득세 별도 원천징수!)
    r_k2s_gr = (-data["k200"].loc[bt_idx, "price_ret"] + 0.5 * kofr_ret.loc[bt_idx] - inv_drag).clip(lower=-0.30)
    r_kqs_gr = (-data["kq"].loc[bt_idx, "price_ret"] + 0.5 * kofr_ret.loc[bt_idx] - inv_drag).clip(lower=-0.30)

    init_cap = 10_000_000.0
    eq = np.zeros(n); eq[0] = init_cap
    bh_k200 = np.zeros(n); bh_k200[0] = init_cap
    bh_kq = np.zeros(n); bh_kq[0] = init_cap
    rets = np.zeros(n)
    w_hist = np.zeros(n)
    turn_arr = np.zeros(n)

    # 숏(인버스 ETF) 이동평균단가 추적용 가상 가격지수 & 평가원가 (실현이익 15.4% 과세 정확 계산)
    px_inv_k2, px_inv_kq = 1.0, 1.0
    avg_inv_k2, avg_inv_kq = 0.0, 0.0
    short_tax_paid = 0.0

    tr_asset = "kofr"
    tr_w = 0.0
    prev_wk2, prev_wkq, prev_wk2s, prev_wkqs = 0.0, 0.0, 0.0, 0.0
    prev_sw_w = 0.0

    events = []
    reasons = []
    alloc_hist = []

    sw_frac_arr = swing_bundle["frac"]
    sw_reasons = swing_bundle["reasons"]
    rsi5_s = swing_bundle["rsi5_series"]

    for i in range(n):
        d = bt_idx[i]
        r2s_i = float(r_k2s_gr.iloc[i])
        rqs_i = float(r_kqs_gr.iloc[i])
        if i > 0:
            px_inv_k2 *= (1.0 + r2s_i)
            px_inv_kq *= (1.0 + rqs_i)
            prev_wkofr = max(0.0, 1.0 - prev_wk2 - prev_wkq - prev_wk2s - prev_wkqs)
            gross = (
                prev_wk2 * float(r_k2_at.iloc[i])
                + prev_wkq * float(r_kq_at.iloc[i])
                + prev_wk2s * r2s_i
                + prev_wkqs * rqs_i
                + prev_wkofr * float(r_kofr_at.iloc[i])
            )
        else:
            gross = 0.0

        mk2, mkq = m_k200.loc[d], m_kq.loc[d]
        on_k2, on_kq = int(st_k200.loc[d, "trend_on"]), int(st_kq.loc[d, "trend_on"])
        p_tr_asset, p_tr_w = tr_asset, tr_w
        tr_reason = ""

        if is_me[i]:
            cands = []
            if not np.isnan(mk2) and mk2 > 0 and on_k2 == 1:
                cands.append((mk2, "k200"))
            if not np.isnan(mkq) and mkq > 0 and on_kq == 1:
                cands.append((mkq, "kq"))
            cands.sort(reverse=True)
            n_tr_asset = cands[0][1] if cands else "kofr"
            if n_tr_asset != p_tr_asset:
                tr_reason = "월말 모멘텀 선정" if n_tr_asset != "kofr" else "월말 전 종목 추세이탈 → KOFR 대피"
        else:
            n_tr_asset = tr_asset
            if tr_asset == "k200" and on_k2 == 0:
                n_tr_asset = "kofr"
                tr_reason = f"월중 KODEX 200 MA{ma_len} 이탈 → KOFR 대피"
            elif tr_asset == "kq" and on_kq == 0:
                n_tr_asset = "kofr"
                tr_reason = f"월중 코스닥150 MA{ma_len} 이탈 → KOFR 대피"

        raw_f = float(fstar_k2.loc[d]) if n_tr_asset == "k200" else (float(fstar_kq.loc[d]) if n_tr_asset == "kq" else 0.0)
        if n_tr_asset == "kofr":
            n_tr_w = 0.0
        elif kelly_frac is None:
            n_tr_w = 1.0
        else:
            target_w = float(np.clip(raw_f * kelly_frac, 0.0, 1.0))
            if n_tr_asset == p_tr_asset and not is_me[i] and abs(target_w - p_tr_w) < rebal_band:
                n_tr_w = p_tr_w
            else:
                n_tr_w = round(target_w * 20) / 20.0
                if n_tr_asset == p_tr_asset and abs(n_tr_w - p_tr_w) >= 0.01:
                    direction = "확대" if n_tr_w > p_tr_w else "축소"
                    tr_reason = f"켈리 비중 {direction} ({int(p_tr_w*100)}%→{int(n_tr_w*100)}%, f*={raw_f:.2f})"
                if n_tr_w == 0.0:
                    n_tr_asset = "kofr"
                    tr_reason = f"켈리 기대우위 소멸(f*={raw_f:.2f}) → KOFR 대피"

        new_wk2s, new_wkqs = 0.0, 0.0
        pair_reason = ""
        if pair_ls:
            # 4번째 옵션: 차별화 페어 롱숏 (60/40) + 유휴25% 코스닥150 단타 스윙
            if n_tr_asset == "k200" and on_kq == 0 and (not np.isnan(mkq)) and mkq < 0:
                long_w = min(n_tr_w, pair_long_cap)
                short_w = round(min(1.0 - long_w, pair_short_cap * min(1.0, n_tr_w / pair_long_cap)) * 20) / 20.0
                new_wk2, new_wkq, new_wk2s, new_wkqs = long_w, 0.0, 0.0, short_w
                sw_w = 0.0
                pair_reason = f"차별화 페어 롱숏 (K200 {int(round(long_w*100))}% 롱 + 코스닥인버스 {int(round(short_w*100))}% 숏)"
            elif n_tr_asset == "kq" and on_k2 == 0 and (not np.isnan(mk2)) and mk2 < 0:
                long_w = min(n_tr_w, pair_long_cap)
                short_w = round(min(1.0 - long_w, pair_short_cap * min(1.0, n_tr_w / pair_long_cap)) * 20) / 20.0
                new_wk2, new_wkq, new_wk2s, new_wkqs = 0.0, long_w, short_w, 0.0
                sw_w = 0.0
                pair_reason = f"차별화 페어 롱숏 (코스닥 {int(round(long_w*100))}% 롱 + KODEX인버스 {int(round(short_w*100))}% 숏)"
            else:
                new_wk2 = n_tr_w if n_tr_asset == "k200" else 0.0
                new_wkq_tr = n_tr_w if n_tr_asset == "kq" else 0.0
                idle_cash = max(0.0, 1.0 - new_wk2 - new_wkq_tr)
                sw_w = min(idle_cash, float(sw_frac_arr[i] * sw_cap))
                new_wkq = new_wkq_tr + sw_w
        elif sw_cap > 0:
            if sw_carve:
                tr_scale = 1.0 - sw_cap
                new_wk2 = (n_tr_w * tr_scale) if n_tr_asset == "k200" else 0.0
                new_wkq_tr = (n_tr_w * tr_scale) if n_tr_asset == "kq" else 0.0
                sw_w = float(sw_frac_arr[i] * sw_cap)
                new_wkq = new_wkq_tr + sw_w
            else:
                new_wk2 = n_tr_w if n_tr_asset == "k200" else 0.0
                new_wkq_tr = n_tr_w if n_tr_asset == "kq" else 0.0
                idle_cash = max(0.0, 1.0 - new_wk2 - new_wkq_tr)
                sw_w = min(idle_cash, float(sw_frac_arr[i] * sw_cap))
                new_wkq = new_wkq_tr + sw_w
        else:
            new_wk2 = n_tr_w if n_tr_asset == "k200" else 0.0
            new_wkq = n_tr_w if n_tr_asset == "kq" else 0.0
            sw_w = 0.0

        sw_changed = abs(sw_w - prev_sw_w) >= 0.01
        tr_changed = (n_tr_asset != p_tr_asset) or (abs(n_tr_w - p_tr_w) >= 0.01)
        pair_changed = (abs(new_wk2s - prev_wk2s) >= 0.01) or (abs(new_wkqs - prev_wkqs) >= 0.01)

        if pair_changed:
            if new_wk2s > 0 or new_wkqs > 0:
                reason = f"{tr_reason + ' · ' if tr_changed and tr_reason else ''}{pair_reason}"
            else:
                reason = f"{tr_reason or '차별화 해소'} → 페어 인버스 숏 청산"
        elif tr_changed and sw_changed:
            reason = f"{tr_reason or '추세 조정'} + {sw_reasons[i]}"
        elif tr_changed:
            reason = tr_reason or "추세 비중 조정"
        elif sw_changed:
            reason = sw_reasons[i]
        else:
            if new_wk2s > 0 or new_wkqs > 0:
                reason = f"유지 · {pair_reason}"
            else:
                reason = "기존 포지션 유지" if sw_w == 0 else f"추세 유지 · {sw_reasons[i]}"

        # 인버스 숏 매도 실현차익 15.4% 세금 원천징수 (이동평균단가 기준)
        tax_drag_pct = 0.0
        if i > 0:
            if new_wk2s < prev_wk2s and prev_wk2s > 0:
                sold_w = prev_wk2s - new_wk2s
                if avg_inv_k2 > 0 and px_inv_k2 > avg_inv_k2:
                    gain_ratio = (px_inv_k2 - avg_inv_k2) / px_inv_k2
                    tax_drag_pct += sold_w * gain_ratio * tax
            if new_wk2s > prev_wk2s:
                add_w = new_wk2s - prev_wk2s
                avg_inv_k2 = ((prev_wk2s * avg_inv_k2) + (add_w * px_inv_k2)) / new_wk2s if prev_wk2s > 0 else px_inv_k2
            elif new_wk2s == 0:
                avg_inv_k2 = 0.0

            if new_wkqs < prev_wkqs and prev_wkqs > 0:
                sold_w = prev_wkqs - new_wkqs
                if avg_inv_kq > 0 and px_inv_kq > avg_inv_kq:
                    gain_ratio = (px_inv_kq - avg_inv_kq) / px_inv_kq
                    tax_drag_pct += sold_w * gain_ratio * tax
            if new_wkqs > prev_wkqs:
                add_w = new_wkqs - prev_wkqs
                avg_inv_kq = ((prev_wkqs * avg_inv_kq) + (add_w * px_inv_kq)) / new_wkqs if prev_wkqs > 0 else px_inv_kq
            elif new_wkqs == 0:
                avg_inv_kq = 0.0

            new_wkofr = max(0.0, 1.0 - new_wk2 - new_wkq - new_wk2s - new_wkqs)
            prev_wkofr = max(0.0, 1.0 - prev_wk2 - prev_wkq - prev_wk2s - prev_wkqs)
            t = 0.5 * (
                abs(new_wk2 - prev_wk2)
                + abs(new_wkq - prev_wkq)
                + abs(new_wk2s - prev_wk2s)
                + abs(new_wkqs - prev_wkqs)
                + abs(new_wkofr - prev_wkofr)
            )
        else:
            t = 0.0

        turn_arr[i] = t
        rets[i] = gross - t * fee - tax_drag_pct
        if i > 0:
            short_tax_paid += eq[i - 1] * tax_drag_pct
            eq[i] = eq[i - 1] * (1.0 + rets[i])
            bh_k200[i] = bh_k200[i - 1] * (1.0 + float(r_k2_at.iloc[i]))
            bh_kq[i] = bh_kq[i - 1] * (1.0 + float(r_kq_at.iloc[i]))

        w_hist[i] = new_wk2 + new_wkq + new_wk2s + new_wkqs
        alloc_hist.append((new_wk2, new_wkq, new_wk2s, new_wkqs, n_tr_asset, n_tr_w, sw_w))
        reasons.append(reason)

        if i > 0 and (
            abs(new_wk2 - prev_wk2) >= 0.005
            or abs(new_wkq - prev_wkq) >= 0.005
            or abs(new_wk2s - prev_wk2s) >= 0.005
            or abs(new_wkqs - prev_wkqs) >= 0.005
        ):
            events.append({
                "date": d.strftime("%Y-%m-%d"),
                "from": fmt_pos_label(prev_wk2, prev_wkq, prev_wk2s, prev_wkqs),
                "to": fmt_pos_label(new_wk2, new_wkq, new_wk2s, new_wkqs),
                "reason": reason,
                "k200_mom6m": round(float(mk2) * 100, 1) if not np.isnan(mk2) else None,
                "kq_mom6m": round(float(mkq) * 100, 1) if not np.isnan(mkq) else None,
                "kq_rsi5": round(float(rsi5_s.loc[d]), 1),
                "kelly_f": round(raw_f, 2),
                "is_asset_switch": bool(n_tr_asset != p_tr_asset or pair_changed),
                "is_swing_event": bool(sw_changed),
            })

        tr_asset, tr_w = n_tr_asset, n_tr_w
        prev_wk2, prev_wkq, prev_wk2s, prev_wkqs, prev_sw_w = new_wk2, new_wkq, new_wk2s, new_wkqs, sw_w

    eq_s = pd.Series(eq, index=bt_idx)
    ret_s = pd.Series(rets, index=bt_idx)
    w_s = pd.Series(w_hist, index=bt_idx)

    ev_14_cnt = sum(1 for ev in events if ev["date"] <= STRATEGY_CONFIG["bt_mid_end"])
    m_11_24 = calc_period_metrics(
        eq_s.iloc[:idx_2024_end + 1],
        ret_s.iloc[:idx_2024_end + 1],
        r_kofr_at.iloc[:idx_2024_end + 1],
        w_s.iloc[:idx_2024_end + 1],
        float(turn_arr[:idx_2024_end + 1].sum()),
        ev_14_cnt,
    )
    b_cagr, b_mdd = sub_cagr_mdd(ret_s, "2011-01-03", "2016-12-31")
    p_cagr, p_mdd = sub_cagr_mdd(ret_s, "2017-01-01", "2024-12-31")
    m_11_24["boxpi_cagr_pct"] = b_cagr
    m_11_24["boxpi_mdd_pct"] = b_mdd
    m_11_24["post_cagr_pct"] = p_cagr
    m_11_24["post_mdd_pct"] = p_mdd

    m_11_26 = calc_period_metrics(
        eq_s, ret_s, r_kofr_at, w_s, float(turn_arr.sum()), len(events)
    )
    r25_cagr, r25_mdd = sub_cagr_mdd(ret_s, "2025-01-01", bt_idx[-1].strftime("%Y-%m-%d"))
    m_11_26["rally_cagr_pct"] = r25_cagr
    m_11_26["rally_mdd_pct"] = r25_mdd

    yearly = []
    for yr, grp in ret_s.groupby(ret_s.index.year):
        yearly.append({"year": int(yr), "ret_pct": round(float((1 + grp).prod() - 1) * 100, 1)})

    # 최신일 스냅샷 & 오늘 액션 판정
    ld = bt_idx[-1]
    lk2, lkq, lk2s, lkqs, ltr_a, ltr_w, lsw_w = alloc_hist[-1]
    pk2, pkq, pk2s, pkqs, ptr_a, ptr_w, psw_w = alloc_hist[-2] if n >= 2 else alloc_hist[-1]

    k2_pct = round(lk2 * 100, 1)
    kq_pct = round(lkq * 100, 1)
    k2s_pct = round(lk2s * 100, 1)
    kqs_pct = round(lkqs * 100, 1)
    kofr_pct = round(max(0.0, 100.0 - k2_pct - kq_pct - k2s_pct - kqs_pct), 1)

    pk2_pct = round(pk2 * 100, 1)
    pkq_pct = round(pkq * 100, 1)
    pk2s_pct = round(pk2s * 100, 1)
    pkqs_pct = round(pkqs * 100, 1)
    pkofr_pct = round(max(0.0, 100.0 - pk2_pct - pkq_pct - pk2s_pct - pkqs_pct), 1)

    sw_diff = round((lsw_w - psw_w) * 100, 1)
    sh_diff = round(((lk2s + lkqs) - (pk2s + pkqs)) * 100, 1)
    tr_diff_asset = (ltr_a != ptr_a)
    tr_diff_w = abs(ltr_w - ptr_w) >= 0.01

    if abs(sh_diff) >= 0.5:
        if (k2s_pct + kqs_pct) > 0 and (pk2s_pct + pkqs_pct) == 0:
            action_code = "PAIR_SHORT_ENTRY"
            action_title = f"⚖️ 차별화 페어 롱숏 진입 (인버스 {k2s_pct + kqs_pct:g}%)"
        elif (k2s_pct + kqs_pct) == 0:
            action_code = "PAIR_SHORT_EXIT"
            action_title = "⚖️ 차별화 페어 인버스 숏 청산"
        else:
            action_code = "PAIR_REBALANCE"
            action_title = f"⚖️ 페어 롱숏 비중 조정 ({sh_diff:+g}%p)"
    elif tr_diff_asset:
        if ltr_a == "kofr" and lsw_w == 0:
            action_code = "EXIT_TO_CASH"
            action_title = "즉시 전량 매도 → KOFR 대피"
        elif ltr_a == "kofr" and lsw_w > 0:
            action_code = "TREND_EXIT_SWING_ON"
            action_title = f"추세 대피 + 코스닥150 단타 {round(lsw_w*100,1):g}% 보유"
        elif ptr_a == "kofr":
            action_code = "NEW_ENTRY"
            action_title = f"신규 추세 매수 ({ETF_META[ltr_a]['name']})"
        else:
            action_code = "SWITCH"
            action_title = f"추세 종목 교체 ({ETF_META[ptr_a]['name']} → {ETF_META[ltr_a]['name']})"
    elif abs(sw_diff) >= 0.5:
        if sw_diff > 0 and psw_w == 0:
            action_code = "SWING_ENTRY"
            action_title = f"⚡ 코스닥150 단타 진입 (+{sw_diff:g}%p)"
        elif sw_diff > 0:
            action_code = "SWING_ADD"
            action_title = f"⚡ 코스닥150 단타 2차 증액 (+{sw_diff:g}%p)"
        else:
            action_code = "SWING_EXIT"
            action_title = f"⚡ 코스닥150 단타 청산 ({sw_diff:g}%p)"
    elif tr_diff_w:
        delta_p = round((k2_pct + kq_pct) - (pk2_pct + pkq_pct), 1)
        action_code = "REBALANCE_WEIGHT"
        action_title = f"켈리 비중 {'확대' if delta_p > 0 else '축소'} ({delta_p:+g}%p)"
    else:
        action_code = "HOLD"
        action_title = "오늘 매매 없음 (기존 포지션 유지)"

    month_end_bday = ld + pd.offsets.BMonthEnd(0)
    if month_end_bday <= ld:
        month_end_bday = ld + pd.offsets.BMonthEnd(1)
    bdays_to_me = int(len(pd.bdate_range(ld, month_end_bday)) - 1)

    alloc = {
        "k200_pct": k2_pct, "kq_pct": kq_pct,
        "k200_short_pct": k2s_pct, "kq_short_pct": kqs_pct,
        "kofr_pct": kofr_pct,
    }
    prev_alloc = {
        "k200_pct": pk2_pct, "kq_pct": pkq_pct,
        "k200_short_pct": pk2s_pct, "kq_short_pct": pkqs_pct,
        "kofr_pct": pkofr_pct,
    }

    if (k2s_pct + kqs_pct) > 0:
        holding_key, holding_name, holding_ticker = "pair", "차별화 페어 롱숏", "롱60/인버스40"
    elif k2_pct > 0 and kq_pct > 0:
        holding_key, holding_name, holding_ticker = "both", "KODEX 200 + 코스닥150", "069500/229200"
    elif k2_pct > 0:
        holding_key, holding_name, holding_ticker = "k200", ETF_META["k200"]["name"], ETF_META["k200"]["ticker"]
    elif kq_pct > 0:
        holding_key, holding_name, holding_ticker = "kq", ETF_META["kq"]["name"], ETF_META["kq"]["ticker"]
    else:
        holding_key, holding_name, holding_ticker = "kofr", ETF_META["kofr"]["name"], ETF_META["kofr"]["ticker"]

    def asset_live_info(key: str, close_s, mom_s, st_df, mu_s, vol_s, fstar_s):
        f_val = float(fstar_s.loc[ld])
        k_w = 100 if kelly_frac is None else int(round(float(np.clip(f_val * kelly_frac, 0.0, 1.0)) * 20) * 5)
        nav_c = float(data[key].loc[ld, "nav_close"])
        adj_c = float(close_s.loc[ld])
        ma_adj = float(st_df.loc[ld, "ma"])
        ratio_nav = nav_c / adj_c if adj_c > 0 else 1.0
        ma_nav = ma_adj * ratio_nav
        info = {
            "name": ETF_META[key]["name"],
            "ticker": ETF_META[key]["ticker"],
            "nav_close": round(nav_c, 1),
            "adj_close": round(adj_c, 1),
            "ret_1d_pct": round(float(data[key].loc[ld, "total_ret"]) * 100, 2),
            "mom_6m_pct": round(float(mom_s.loc[ld]) * 100, 2),
            "ma_nav": round(ma_nav, 1),
            "stop_nav": round(ma_nav * (1.0 - STRATEGY_CONFIG["breakout_pct"]), 1),
            "disp_pct": round(float(st_df.loc[ld, "disp"]) * 100, 2),
            "above_streak": int(st_df.loc[ld, "above_streak"]),
            "below_streak": int(st_df.loc[ld, "below_streak"]),
            "trend_on": bool(st_df.loc[ld, "trend_on"] == 1),
            "pass_all": bool((mom_s.loc[ld] > 0) and (st_df.loc[ld, "trend_on"] == 1)),
            "ann_excess_mu_pct": round(float(mu_s.loc[ld]) * 100, 2),
            "ann_vol_60d_pct": round(float(vol_s.loc[ld]) * 100, 2),
            "raw_kelly_f": round(f_val, 2),
            "kelly_target_pct": k_w,
        }
        if key == "kq":
            info["rsi5"] = round(float(rsi5_s.loc[ld]), 1)
        return info

    asset_days = {"k200": 0, "kq": 0, "kofr": 0, "swing": 0, "pair_short": 0}
    for wk2_i, wkq_i, wk2s_i, wkqs_i, _, _, sw_i in alloc_hist:
        if wk2_i >= 0.005: asset_days["k200"] += 1
        if wkq_i >= 0.005: asset_days["kq"] += 1
        if (wk2s_i + wkqs_i) >= 0.005: asset_days["pair_short"] += 1
        if (1.0 - wk2_i - wkq_i - wk2s_i - wkqs_i) >= 0.005: asset_days["kofr"] += 1
        if sw_i >= 0.005: asset_days["swing"] += 1

    switch_only_events = [ev for ev in events if ev.get("is_asset_switch")]
    swing_only_events = [ev for ev in events if ev.get("is_swing_event")]

    dd_s = (eq_s / eq_s.cummax() - 1.0) * 100.0
    # 5영업일 샘플링 + 2024년 말일 + 최종 거래일 정확 포함
    sample_set = set(range(0, n, 5))
    sample_set.add(idx_2024_end)
    sample_set.add(n - 1)
    sample_idx = sorted(sample_set)
    end_2024_sample_pos = sample_idx.index(idx_2024_end)

    strategy_id = f"ma{ma_len}_{kelly_opt['id']}_{swing_opt['id']}"
    result = {
        "id": strategy_id,
        "ma_len": ma_len,
        "kelly_id": kelly_opt["id"],
        "kelly_label": kelly_opt["label"],
        "kelly_frac": kelly_frac,
        "kelly_desc": kelly_opt["desc"],
        "swing_id": swing_opt["id"],
        "swing_label": swing_opt["label"],
        "swing_short": swing_opt["short_label"],
        "swing_desc": swing_opt["desc"],
        "swing_active_pct": round(lsw_w * 100, 1),
        "title": f"MA{ma_len} + {kelly_opt['label']} + {swing_opt['short_label']}",
        "as_of_date": ld.strftime("%Y-%m-%d"),
        "holding_key": holding_key,
        "holding_name": holding_name,
        "holding_ticker": holding_ticker,
        "pos_summary": fmt_pos_label(lk2, lkq, lk2s, lkqs),
        "equity_weight_pct": round(k2_pct + kq_pct + k2s_pct + kqs_pct, 1),
        "allocation": alloc,
        "prev_allocation": prev_alloc,
        "action_code": action_code,
        "action_title": action_title,
        "action_detail": reasons[-1],
        "next_rebal_date": month_end_bday.strftime("%Y-%m-%d"),
        "bdays_to_rebal": bdays_to_me,
        "k200": asset_live_info("k200", k200_c, m_k200, st_k200, mu_k2, vol_k2, fstar_k2),
        "kq": asset_live_info("kq", kq_c, m_kq, st_kq, mu_kq, vol_kq, fstar_kq),
        "kofr_nav": round(float(data["kofr"].loc[ld, "nav_close"]), 1) if ld in data["kofr"].index else 110000.0,
        "asset_days": {
            "k200": {"name": "KODEX 200 롱", "days": asset_days["k200"], "pct": round(asset_days["k200"] / n * 100, 1)},
            "kq": {"name": "코스닥150 롱 (추세+스윙)", "days": asset_days["kq"], "pct": round(asset_days["kq"] / n * 100, 1)},
            "swing": {"name": "코스닥150 단타 스윙", "days": asset_days["swing"], "pct": round(asset_days["swing"] / n * 100, 1)},
            "pair_short": {"name": "페어 인버스 숏", "days": asset_days["pair_short"], "pct": round(asset_days["pair_short"] / n * 100, 1)},
            "kofr": {"name": "KOFR 현금 보유일", "days": asset_days["kofr"], "pct": round(asset_days["kofr"] / n * 100, 1)},
        },
        "metrics_11_24": m_11_24,
        "metrics_11_26": m_11_26,
        "metrics": m_11_26,
        "yearly": yearly,
        "recent_events": events[-20:][::-1],
        "switch_events": switch_only_events[-20:][::-1],
        "swing_events": swing_only_events[-20:][::-1],
        "series": {
            "strategy": [round(float(eq[j]) / 1e4, 1) for j in sample_idx],
            "dd_pct": [round(float(dd_s.iloc[j]), 2) for j in sample_idx],
        },
    }

    # 벤치마크 14년/16년 수치도 함께 전달
    bh2_s = pd.Series(bh_k200, index=bt_idx)
    bhq_s = pd.Series(bh_kq, index=bt_idx)
    bench = {
        "dates": [bt_idx[j].strftime("%Y-%m-%d") for j in sample_idx],
        "end_2024_pos": end_2024_sample_pos,
        "bh_k200": [round(float(bh_k200[j]) / 1e4, 1) for j in sample_idx],
        "bh_kq": [round(float(bh_kq[j]) / 1e4, 1) for j in sample_idx],
        "bh_k200_11_24_cagr": round(((bh2_s.iloc[idx_2024_end] / bh2_s.iloc[0]) ** (1.0 / m_11_24["years"]) - 1.0) * 100, 2),
        "bh_k200_11_26_cagr": round(((bh2_s.iloc[-1] / bh2_s.iloc[0]) ** (1.0 / m_11_26["years"]) - 1.0) * 100, 2),
        "bh_kq_11_24_cagr": round(((bhq_s.iloc[idx_2024_end] / bhq_s.iloc[0]) ** (1.0 / m_11_24["years"]) - 1.0) * 100, 2),
        "bh_kq_11_26_cagr": round(((bhq_s.iloc[-1] / bhq_s.iloc[0]) ** (1.0 / m_11_26["years"]) - 1.0) * 100, 2),
    }
    return result, bench


def build_interactive_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>국내 2대 지수 모멘텀·켈리 + 코스닥150 단타 스윙 (2011~2024 / 2011~2026)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {{ --bg:#f4f6fb; --card:#ffffff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; --accent:#1d4ed8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Malgun Gothic","Apple SD Gothic Neo",sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
    .wrap {{ max-width:1160px; margin:0 auto; padding:22px 16px 56px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; margin-bottom:14px; }}
    .topbar h1 {{ margin:0; font-size:22px; font-weight:800; letter-spacing:-0.4px; }}
    .topbar .meta {{ font-size:13px; color:var(--muted); }}

    .selector-card {{ background:#fff; border:1px solid var(--border); border-radius:14px; padding:16px 20px; margin-bottom:18px; box-shadow:0 2px 6px rgba(15,23,42,0.04); display:flex; flex-direction:column; gap:14px; }}
    .sel-row {{ display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start; justify-content:space-between; }}
    .sel-group {{ display:flex; flex-direction:column; gap:6px; }}
    .sel-label {{ font-size:12px; font-weight:700; color:var(--muted); text-transform:uppercase; }}
    .btn-row {{ display:flex; gap:6px; flex-wrap:wrap; }}
    .seg-btn {{ border:1px solid #d1d5db; background:#f9fafb; color:#374151; padding:7px 13px; border-radius:9px; font-size:13px; font-weight:700; cursor:pointer; transition:all 0.15s; }}
    .seg-btn:hover {{ background:#eff6ff; border-color:#93c5fd; }}
    .seg-btn.active {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8; box-shadow:0 2px 6px rgba(29,78,216,0.28); }}
    .seg-btn.swing-active {{ background:#7c3aed; color:#fff; border-color:#7c3aed; box-shadow:0 2px 6px rgba(124,58,237,0.28); }}
    .sel-desc {{ font-size:12.5px; color:#4b5563; margin-top:2px; }}

    .hero {{ border-radius:16px; padding:24px 26px; color:#fff; box-shadow:0 10px 25px rgba(15,23,42,0.13); margin-bottom:18px; transition:background 0.25s; }}
    .hero-top {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }}
    .hero-tag {{ background:rgba(255,255,255,0.18); padding:4px 12px; border-radius:999px; font-size:13px; font-weight:700; }}
    .hero-action-badge {{ background:#fef08a; color:#854d0e; padding:5px 13px; border-radius:999px; font-size:13px; font-weight:800; }}
    .hero-main {{ font-size:28px; font-weight:800; margin:12px 0 6px; letter-spacing:-0.6px; }}
    .hero-sub {{ font-size:14px; opacity:0.93; display:flex; flex-wrap:wrap; gap:16px; margin-top:6px; }}

    .swing-card {{ background:linear-gradient(135deg, #faf5ff 0%, #ffffff 100%); border:2px solid #8b5cf6; border-radius:14px; padding:16px 20px; margin-bottom:18px; box-shadow:0 3px 10px rgba(139,92,246,0.08); }}
    .rsi-bar-wrap {{ margin-top:10px; }}
    .rsi-bar-bg {{ height:10px; background:linear-gradient(90deg, #dc2626 0%, #f59e0b 20%, #e5e7eb 50%, #10b981 100%); border-radius:999px; position:relative; }}
    .rsi-marker {{ position:absolute; top:-4px; width:18px; height:18px; border-radius:50%; background:#111827; border:3px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,0.4); transform:translateX(-50%); }}
    .rsi-scale {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-top:4px; }}

    .action-card {{ background:#fff; border:2px solid #2563eb; border-radius:14px; padding:18px 20px; margin-bottom:18px; box-shadow:0 3px 10px rgba(37,99,235,0.08); }}
    .action-head {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:14px; }}
    .cap-input-wrap {{ display:flex; align-items:center; gap:8px; background:#f3f4f6; padding:6px 12px; border-radius:10px; }}
    .cap-input-wrap label {{ font-size:13px; font-weight:700; color:#374151; }}
    .cap-input-wrap input {{ width:120px; border:1px solid #d1d5db; border-radius:7px; padding:5px 8px; font-size:14px; font-weight:800; text-align:right; color:#111827; }}
    .alloc-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }}
    .alloc-box {{ background:#f9fafb; border:1px solid var(--border); border-radius:12px; padding:14px; }}
    .alloc-box.active-alloc {{ background:#eff6ff; border-color:#93c5fd; }}
    .alloc-name {{ font-size:14px; font-weight:800; }}
    .alloc-pct {{ font-size:26px; font-weight:900; margin:4px 0; color:#111827; }}
    .alloc-delta {{ font-size:12.5px; font-weight:700; }}
    .alloc-krw {{ font-size:13.5px; font-weight:700; color:#1d4ed8; margin-top:4px; }}

    .period-banner {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; padding-bottom:10px; margin-bottom:12px; border-bottom:2px solid #f3f4f6; }}
    .period-tag-14 {{ background:#fef3c7; color:#92400e; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:800; }}
    .period-tag-16 {{ background:#dbeafe; color:#1e40af; padding:4px 11px; border-radius:999px; font-size:12px; font-weight:800; }}

    .kpi-strip {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(145px, 1fr)); gap:10px; margin-bottom:14px; }}
    .stat-card {{ background:#f9fafb; border:1px solid var(--border); border-radius:12px; padding:12px 13px; }}
    .stat-label {{ font-size:12px; color:var(--muted); font-weight:600; }}
    .stat-val {{ font-size:19px; font-weight:800; margin-top:3px; }}
    .stat-sub {{ font-size:11.5px; color:var(--muted); margin-top:2px; }}

    .grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(330px, 1fr)); gap:16px; margin-bottom:18px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
    .card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:12px; }}
    .asset-title {{ font-size:17.5px; font-weight:800; }}
    .ticker {{ font-size:12.5px; font-weight:600; color:var(--muted); background:#f3f4f6; padding:2px 7px; border-radius:6px; margin-left:4px; }}
    .sub {{ font-size:12.5px; color:var(--muted); margin-top:2px; }}
    .badges {{ display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }}
    .pill {{ font-size:11.5px; font-weight:700; padding:3px 9px; border-radius:999px; }}
    .pill-blue {{ background:#dbeafe; color:#1d4ed8; }}
    .pill-purple {{ background:#ede9fe; color:#6d28d9; }}
    .pill-green {{ background:#dcfce7; color:#15803d; }}
    .pill-red {{ background:#fee2e2; color:#b91c1c; }}
    .pill-gray {{ background:#f3f4f6; color:#4b5563; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; }}
    .kpi {{ background:#f9fafb; border-radius:9px; padding:9px 10px; }}
    .kpi-label {{ font-size:11.5px; color:var(--muted); }}
    .kpi-val {{ font-size:16px; font-weight:800; margin-top:2px; }}
    .pos {{ color:#dc2626; }}
    .neg {{ color:#2563eb; }}
    h2 {{ font-size:16.5px; font-weight:800; margin:0 0 12px; }}
    table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
    th, td {{ padding:8px 7px; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }}
    th {{ font-size:11.5px; color:var(--muted); font-weight:700; background:#f9fafb; }}
    .col-sep {{ border-left:2px solid #e5e7eb; }}
    tr.clickable {{ cursor:pointer; transition:background 0.12s; }}
    tr.clickable:hover {{ background:#f3f4f6; }}
    tr.selected-row {{ background:#eff6ff !important; font-weight:700; }}
    .section {{ margin-bottom:18px; }}
    .chart-box {{ height:290px; position:relative; }}
    .chart-box-sm {{ height:165px; position:relative; margin-top:12px; }}
    .yr-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(84px, 1fr)); gap:8px; }}
    .yr-chip {{ border-radius:10px; padding:8px; text-align:center; border:1px solid var(--border); background:#f9fafb; }}
    .yr-chip.boxpi-yr {{ border-color:#fcd34d; background:#fffbeb; }}
    .yr-chip.rally-yr {{ border-color:#93c5fd; background:#eff6ff; }}
    .yr-chip .y {{ font-size:11.5px; color:var(--muted); font-weight:700; }}
    .yr-chip .r {{ font-size:15px; font-weight:800; margin-top:2px; }}
    .rules {{ font-size:13.5px; color:#374151; margin:0; padding-left:18px; }}
    .rules li {{ margin-bottom:6px; }}
    .foot {{ text-align:center; font-size:12px; color:var(--muted); margin-top:24px; }}
    @media (max-width:640px) {{ .hero-main {{ font-size:21px; }} .kpi-grid {{ grid-template-columns:repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1>📈 국내 2대 지수 모멘텀·켈리 + 코스닥150 단타 스윙</h1>
      <div class="meta">기준 거래일: <strong id="asOfDate">-</strong> (KOFIA 공식 종가) · 갱신: <span id="genAt">-</span></div>
    </div>

    <!-- 전략 선택기 -->
    <div class="selector-card">
      <div class="sel-row">
        <div class="sel-group">
          <div class="sel-label">1. 일간 추세 방어선</div>
          <div class="btn-row" id="maBtns">
            <button class="seg-btn" data-ma="100">MA100 (기본 권장)</button>
            <button class="seg-btn" data-ma="80">MA80 (중립)</button>
            <button class="seg-btn" data-ma="60">MA60 (보수)</button>
          </div>
        </div>
        <div class="sel-group">
          <div class="sel-label">2. 추세 켈리 사이징</div>
          <div class="btn-row" id="kellyBtns">
            <button class="seg-btn" data-kelly="full">풀 켈리 (1.0x)</button>
            <button class="seg-btn" data-kelly="half">하프 켈리 (0.5x)</button>
            <button class="seg-btn" data-kelly="quarter">쿼터 켈리 (0.25x)</button>
            <button class="seg-btn" data-kelly="fixed">고정 100%</button>
          </div>
          <div class="sel-desc" id="kellyDesc">-</div>
        </div>
      </div>
      <div class="sel-group" style="border-top:1px solid #f3f4f6; padding-top:12px;">
        <div class="sel-label" style="color:#6d28d9;">3. ⚡ 박스피 단타 스윙 / 차별화 페어 롱숏 선택</div>
        <div class="btn-row" id="swingBtns">
          <button class="seg-btn" data-swing="overlay25">유휴현금 25% 스윙 (공격 권장)</button>
          <button class="seg-btn" data-swing="split75">계좌 분리 75:25 (박스피 방어)</button>
          <button class="seg-btn" data-swing="pair60">차별화 페어 롱숏 60/40 (CAGR·MDD 최강)</button>
          <button class="seg-btn" data-swing="none">스윙 미사용 (순수 추세)</button>
        </div>
        <div class="sel-desc" id="swingDesc">-</div>
      </div>
    </div>

    <!-- Hero Signal -->
    <div class="hero" id="heroBox">
      <div class="hero-top">
        <span class="hero-tag" id="heroRegime">-</span>
        <span class="hero-action-badge" id="heroActionBadge">-</span>
      </div>
      <div class="hero-main" id="heroMain">-</div>
      <div class="hero-sub" id="heroSub"></div>
    </div>

    <!-- 코스닥150 1주일 분할 스윙 실시간 모니터 -->
    <div class="swing-card" id="swingMonitor"></div>

    <!-- 오늘의 액션 & 주문 금액 계산기 -->
    <div class="action-card">
      <div class="action-head">
        <div>
          <h2 style="margin:0;">⚡ 오늘의 액션 & 종목별 매수 금액 계산기</h2>
          <div class="sub" id="actionReasonText">-</div>
        </div>
        <div class="cap-input-wrap">
          <label for="capInput">내 투자원금 (만 원)</label>
          <input type="number" id="capInput" value="5000" min="100" step="100" />
        </div>
      </div>
      <div class="alloc-grid" id="allocGrid"></div>
    </div>

    <!-- 2개 지수 실시간 켈리 & 추세 진단 -->
    <div class="grid-2" id="assetCards"></div>

    <!-- [검증 1] 2011 ~ 2024 (14년, 박스피 포함 · 25~26 폭등 제외) -->
    <div class="card section" style="border-top:4px solid #f59e0b;">
      <div class="period-banner">
        <h2 id="title14" style="margin:0;">📊 [검증 1] 2011 ~ 2024 (14.0년 · 박스피 6년 포함 · 25~26 폭등 제외)</h2>
        <span class="period-tag-14">스트레스 검증 구간 (2025~26 대세장 완전 배제)</span>
      </div>
      <div class="kpi-strip" id="kpiStrip14"></div>
      <div class="chart-box"><canvas id="eqChart14"></canvas></div>
      <div class="chart-box-sm"><canvas id="ddChart14"></canvas></div>
    </div>

    <!-- [검증 2] 2011 ~ 2026 (15.9년, 박스피 + 25~26 전체 풀사이클) -->
    <div class="card section" style="border-top:4px solid #2563eb;">
      <div class="period-banner">
        <h2 id="title16" style="margin:0;">📊 [검증 2] 2011 ~ 2026 (15.9년 · 박스피 + 25~26 풀사이클 전체)</h2>
        <span class="period-tag-16">장기 전체 구간 (2011.01 ~ 최신 거래일)</span>
      </div>
      <div class="kpi-strip" id="kpiStrip16"></div>
      <div class="chart-box"><canvas id="eqChart16"></canvas></div>
      <div class="chart-box-sm"><canvas id="ddChart16"></canvas></div>
    </div>

    <!-- 연도별 수익률 (2011~2026 16년 전체) -->
    <div class="card section">
      <h2>📅 선택 전략 연도별 세후 수익률 (2011 ~ 2026, 16년 전체) <span style="font-size:12px;color:#6b7280;font-weight:500;">노랑=박스피(11~16) · 파랑=대세랠리(25~26)</span></h2>
      <div class="yr-grid" id="yrGrid"></div>
    </div>

    <!-- 전략 비교표 (2011~2024 vs 2011~2026 나란히 비교) -->
    <div class="card section">
      <div class="action-head" style="margin-bottom:10px;">
        <h2 style="margin:0;">🏆 전략 백테스트 비교: 2011~2024 (14년) vs 2011~2026 (15.9년) <span style="font-size:12px;color:#6b7280;font-weight:500;">(행 클릭 시 즉시 전환)</span></h2>
        <div class="btn-row" id="cmpFilterBtns">
          <button class="seg-btn active" data-cmp="featured">핵심 추천 12종 (4가지 모드 비교)</button>
          <button class="seg-btn" data-cmp="cur_swing">현재 모드 전체 (12종)</button>
          <button class="seg-btn" data-cmp="all">전체 48종</button>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>전략</th>
              <th>오늘 포지션</th>
              <th class="col-sep">11~24 CAGR</th>
              <th>11~24 MDD</th>
              <th>11~24 샤프</th>
              <th>박스피(11~16)</th>
              <th class="col-sep">11~26 CAGR</th>
              <th>11~26 MDD</th>
              <th>11~26 샤프</th>
              <th>칼마</th>
            </tr>
          </thead>
          <tbody id="compareTbody"></tbody>
        </table>
      </div>
    </div>

    <!-- 최근 매매 / 스윙 / 비중조절 이력 -->
    <div class="card section">
      <div class="action-head" style="margin-bottom:10px;">
        <h2 id="eventsTitle" style="margin:0;">🔄 최근 시그널 · 스윙 · 비중조절 이력</h2>
        <div class="btn-row" id="evFilterBtns">
          <button class="seg-btn active" data-ev="all">전체 (추세+단타스윙+켈리)</button>
          <button class="seg-btn" data-ev="swing">코스닥150 단타 스윙만</button>
          <button class="seg-btn" data-ev="switches">추세 종목 전환만</button>
        </div>
      </div>
      <div class="kpi-strip" id="breakdownStrip" style="margin-bottom:12px;"></div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr><th>전환일</th><th>포지션 변경</th><th>사유</th><th>6M 모멘텀 (200 / 코스닥)</th><th>코스닥 RSI(5)</th></tr>
          </thead>
          <tbody id="eventsTbody"></tbody>
        </table>
      </div>
    </div>

    <!-- 규칙 & 데이터 안내 -->
    <div class="card section">
      <h2>📐 옵션 A 매매 규칙 & 2011~2026 장기 검증 데이터 기준</h2>
      <ul class="rules">
        <li><strong>① 추세 엔진:</strong> 매월 말 <code>KODEX 200</code>·<code>코스닥150</code> 중 6M 모멘텀 &gt; 0% &amp; 일간 MA ON 1위 보유, 일간 MA 이탈 시 즉시 <code>KOFR</code> 대피. 축소추정 켈리(<code>f* = μ_shrink / max(σ_20, σ_60)²</code>)로 0~100% 자동 사이징.</li>
        <li><strong>② 코스닥150 1주일 분할 스윙 (최대 25% 캡):</strong> <code>RSI(5) &lt; 20</code> 시 1차 <strong>12.5%</strong>, <code>RSI(5) &lt; 15</code> 추가 급락 시 2차 <strong>25%</strong> 진입 → <code>RSI(5) ≥ 50</code> 반등 익절 또는 <strong>5영업일(1주일)</strong> 만기 자동 청산.</li>
        <li><strong>③ 차별화 페어 롱숏 60/40 (4번째 옵션):</strong> 하락장 맨몸 숏은 절대 치지 않고, <strong>한쪽 지수는 추세 ON 롱 + 반대 지수는 <code>MA OFF &amp; 6M &lt; 0%</code> 약세인 차별화 장세</strong>에서만 강한 지수 <strong>60% 롱</strong> + 반대 1x 인버스 ETF(<code>KODEX 인버스 114800</code> / <code>KODEX 코스닥150선물인버스 251340</code>) <strong>40% 숏</strong>을 동시 보유합니다. 인버스 ETF의 파생상품 <strong>매매차익 15.4% 과세</strong>와 연 0.8% 롤오버·보수를 전부 차감한 세후 수치입니다.</li>
        <li><strong>④ 장기 데이터 투명성 고지:</strong> <code>KODEX 200</code>은 2009년부터 전 구간 KOFIA DIS 공식 수정기준가(분배금 재투자·실제 일별 CD91/KOFR 금리)를 사용했고, <code>KODEX 코스닥150</code> 상장(2015-09-30) 이전인 <code>2011-01 ~ 2015-09-29</code> 구간은 금융투자협회 FreeSIS 공식 코스닥 종합지수 일별 수익률을 접합했습니다. (실전 국내 ETF 편도 비용 0.05%, 미래참조 0)</li>
      </ul>
    </div>

    <div class="foot">
      데이터 출처: 금융투자협회 전자공시(KOFIA DIS) &amp; FreeSIS 공식 데이터 · 평일 오전 08:40 KST GitHub Actions 자동 업데이트 &amp; ntfy 푸시<br/>
      ※ 본 대시보드는 과거 공식 데이터 기반 전략 검증 및 참고용이며 최종 투자 판단과 책임은 투자자 본인에게 있습니다.
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    let selMa = 100;
    let selKelly = "full";
    let selSwing = "overlay25";
    let evFilter = "all";
    let cmpFilter = "featured";
    let eq14Inst = null, dd14Inst = null, eq16Inst = null, dd16Inst = null;

    document.getElementById("asOfDate").textContent = DATA.as_of_date;
    document.getElementById("genAt").textContent = DATA.generated_at_kst;

    function getStrat(ma, kelly, swing) {{
      return DATA.strategies[`ma${{ma}}_${{kelly}}_${{swing}}`];
    }}
    function fmtPct(v) {{
      return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
    }}
    function fmtW(v) {{
      return Number(v.toFixed(1)) + "%";
    }}
    function fmtDelta(d) {{
      const rd = Math.round(d * 10) / 10;
      if (Math.abs(rd) < 0.05) return '<span style="color:#6b7280;">변동 없음 (0%p)</span>';
      return rd > 0 ? `<span class="pos">▲ +${{rd}}%p 매수</span>` : `<span class="neg">▼ ${{rd}}%p 매도</span>`;
    }}

    function renderSwingMonitor(s) {{
      const sw = DATA.live_swing;
      const st14 = sw.stats_11_24;
      const st16 = sw.stats_11_26;
      const rsi = sw.rsi5;
      const markerPos = Math.max(2, Math.min(98, rsi));
      let stageBadge = '<span class="pill pill-gray">단타 대기 (0%)</span>';
      if (sw.stage === 1) stageBadge = `<span class="pill pill-purple">⚡ 1차 진입 (12.5% · ${{sw.hold_days}}/${{sw.max_hold_days}}일)</span>`;
      if (sw.stage === 2) stageBadge = `<span class="pill pill-red">🔥 2차 풀캡 (25% · ${{sw.hold_days}}/${{sw.max_hold_days}}일)</span>`;
      const modeNote = s.swing_id === "none"
        ? '<span class="pill pill-gray">현재 선택 전략은 스윙 미적용</span>'
        : `<span class="pill pill-purple">현재 전략 단타 적용 비중: ${{fmtW(s.swing_active_pct)}}</span>`;

      document.getElementById("swingMonitor").innerHTML = `
        <div class="card-head" style="margin-bottom:8px;">
          <div>
            <div class="asset-title" style="color:#5b21b6;">⚡ 코스닥150 극단 과매도 1주일 분할 스윙 모니터</div>
            <div class="sub">${{sw.status_text}} · <strong>11~24(14년):</strong> ${{st14.trades}}회 / 승률 ${{st14.win_rate_pct}}% / PF ${{st14.profit_factor}} / 켈리 f*=${{st14.kelly_f_pct}}% · <strong>11~26(15.9년):</strong> ${{st16.trades}}회 / 승률 ${{st16.win_rate_pct}}% / PF ${{st16.profit_factor}}</div>
          </div>
          <div class="badges">${{stageBadge}} ${{modeNote}}</div>
        </div>
        <div class="kpi-grid" style="grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));">
          <div class="kpi"><div class="kpi-label">코스닥150 RSI(5)</div><div class="kpi-val ${{rsi < 20 ? 'pos' : (rsi >= 50 ? 'neg' : '')}}">${{rsi.toFixed(1)}}</div></div>
          <div class="kpi"><div class="kpi-label">1차 / 2차 매수선</div><div class="kpi-val" style="font-size:15px;">&lt; 20 (12.5%) / &lt; 15 (25%)</div></div>
          <div class="kpi"><div class="kpi-label">반등 익절 / 만기</div><div class="kpi-val" style="font-size:15px;">RSI ≥ 50 또는 5영업일</div></div>
          <div class="kpi"><div class="kpi-label">보유 경과 / 잔여일</div><div class="kpi-val" style="font-size:15px;">${{sw.stage > 0 ? sw.hold_days + '일 경과 (잔여 ' + sw.remain_days + '일)' : '대기 중'}}</div></div>
        </div>
        <div class="rsi-bar-wrap">
          <div class="rsi-bar-bg"><div class="rsi-marker" style="left:${{markerPos}}%;" title="RSI(5)=${{rsi}}"></div></div>
          <div class="rsi-scale"><span>0 (극단과매도)</span><span>15 (2차 25%)</span><span>20 (1차 12.5%)</span><span>50 (반등 익절)</span><span>100</span></div>
        </div>
      `;
    }}

    function renderAll() {{
      const s = getStrat(selMa, selKelly, selSwing);
      const sPure = getStrat(selMa, selKelly, "none");
      const m14 = s.metrics_11_24;
      const m16 = s.metrics_11_26;

      document.querySelectorAll("#maBtns .seg-btn").forEach(b => b.classList.toggle("active", Number(b.dataset.ma) === selMa));
      document.querySelectorAll("#kellyBtns .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.kelly === selKelly));
      document.querySelectorAll("#swingBtns .seg-btn").forEach(b => b.classList.toggle("swing-active", b.dataset.swing === selSwing));
      document.getElementById("kellyDesc").textContent = s.kelly_desc;
      document.getElementById("swingDesc").textContent = s.swing_desc;

      // Hero Banner
      const hero = document.getElementById("heroBox");
      let bg = "linear-gradient(135deg, #4b5563 0%, #374151 100%)", regTxt = `🛡️ 방어 국면 · KOFR 100% 대피 (${{s.title}})`;
      if (s.holding_key === "pair") {{
        bg = "linear-gradient(135deg, #0f766e 0%, #991b1b 100%)";
        regTxt = `⚖️ 차별화 페어 롱/숏 발동 (${{s.title}})`;
      }} else if (s.holding_key === "both") {{
        bg = "linear-gradient(135deg, #4f46e5 0%, #0e7490 100%)";
        regTxt = `⚡ 복합 국면 · 추세 + 코스닥150 단타 동시 보유 (${{s.title}})`;
      }} else if (s.holding_key === "k200") {{
        bg = "linear-gradient(135deg, #0f4c81 0%, #1b6ca8 100%)";
        regTxt = `🔵 공격 국면 · 대형주 주도 (${{s.title}})`;
      }} else if (s.holding_key === "kq") {{
        bg = s.swing_active_pct > 0 && s.allocation.kq_pct === s.swing_active_pct
          ? "linear-gradient(135deg, #6d28d9 0%, #7c3aed 100%)"
          : "linear-gradient(135deg, #0e6b4e 0%, #179c72 100%)";
        regTxt = s.swing_active_pct > 0 && s.allocation.kq_pct === s.swing_active_pct
          ? `⚡ 단타 스윙 발동 · 코스닥150 과매도 반등 노림 (${{s.title}})`
          : `🟢 공격 국면 · 코스닥 주도 (${{s.title}})`;
      }}
      hero.style.background = bg;
      document.getElementById("heroRegime").textContent = regTxt;
      document.getElementById("heroActionBadge").textContent = `오늘 액션: ${{s.action_title}}`;
      document.getElementById("heroMain").textContent = s.pos_summary;
      document.getElementById("heroSub").innerHTML = `
        <span>판정 근거: <strong>${{s.action_detail}}</strong></span>
        <span>11~24(14년): <strong>${{fmtPct(m14.cagr_pct)}}</strong> (MDD ${{m14.mdd_pct.toFixed(1)}}% / Shp ${{m14.ex_sharpe.toFixed(2)}})</span>
        <span>11~26(15.9년): <strong>${{fmtPct(m16.cagr_pct)}}</strong> (MDD ${{m16.mdd_pct.toFixed(1)}}% / Shp ${{m16.ex_sharpe.toFixed(2)}})</span>
      `;

      renderSwingMonitor(s);
      document.getElementById("actionReasonText").textContent = `기준일 ${{s.as_of_date}} 종가 판정: ${{s.action_title}} — ${{s.action_detail}}`;
      renderCalculator(s);

      document.getElementById("assetCards").innerHTML = [
        renderAssetCard(s.k200, s.ma_len, s.kelly_label, s.allocation.k200_pct, false),
        renderAssetCard(s.kq, s.ma_len, s.kelly_label, s.allocation.kq_pct, true),
      ].join("");

      // [검증 1] 2011~2024 KPI Strip
      const d14 = m14.cagr_pct - sPure.metrics_11_24.cagr_pct;
      document.getElementById("title14").textContent = `📊 [검증 1] 2011 ~ 2024 (${{m14.years}}년 · 박스피 6년 포함 · 25~26 폭등 제외) — ${{s.title}}`;
      const kpis14 = [
        ["14년 세후 최종자산", `${{m14.final_equity_man.toLocaleString()}}만 원`, "2011.01 1,000만 원 시작", "pos"],
        ["14년 세후 CAGR", fmtPct(m14.cagr_pct), s.swing_id === "none" ? `KODEX200 B&H: ${{fmtPct(DATA.benchmark.bh_k200_11_24_cagr)}}` : `순수추세 대비 ${{d14 >= 0 ? '+' : ''}}${{d14.toFixed(2)}}%p`, "pos"],
        ["14년 최대낙폭(MDD)", `${{m14.mdd_pct.toFixed(1)}}%`, `순수추세: ${{sPure.metrics_11_24.mdd_pct.toFixed(1)}}%`, "neg"],
        ["초과샤프 / 칼마", `${{m14.ex_sharpe.toFixed(2)}} / ${{m14.calmar.toFixed(2)}}`, `연 변동성 ${{m14.vol_pct.toFixed(1)}}%`, ""],
        ["박스피 (2011~16)", `${{m14.boxpi_cagr_pct >= 0 ? '+' : ''}}${{m14.boxpi_cagr_pct.toFixed(1)}}%`, `박스피 MDD ${{m14.boxpi_mdd_pct.toFixed(1)}}%`, m14.boxpi_cagr_pct >= 0 ? "pos" : "neg"],
        ["후기장 (2017~24)", `${{m14.post_cagr_pct >= 0 ? '+' : ''}}${{m14.post_cagr_pct.toFixed(1)}}%`, `후기 MDD ${{m14.post_mdd_pct.toFixed(1)}}% · 비중 ${{m14.avg_weight_pct}}%`, "pos"],
      ];
      document.getElementById("kpiStrip14").innerHTML = kpis14.map(([l, v, sub, cls]) => `
        <div class="stat-card"><div class="stat-label">${{l}}</div><div class="stat-val ${{cls}}">${{v}}</div><div class="stat-sub">${{sub}}</div></div>
      `).join("");

      // [검증 2] 2011~2026 KPI Strip
      const d16 = m16.cagr_pct - sPure.metrics_11_26.cagr_pct;
      document.getElementById("title16").textContent = `📊 [검증 2] 2011 ~ 2026 (${{m16.years}}년 · 박스피 + 25~26 풀사이클 전체) — ${{s.title}}`;
      const kpis16 = [
        ["15.9년 세후 최종자산", `${{m16.final_equity_man.toLocaleString()}}만 원`, "2011.01 1,000만 원 시작", "pos"],
        ["15.9년 세후 CAGR", fmtPct(m16.cagr_pct), s.swing_id === "none" ? `KODEX200 B&H: ${{fmtPct(DATA.benchmark.bh_k200_11_26_cagr)}}` : `순수추세 대비 ${{d16 >= 0 ? '+' : ''}}${{d16.toFixed(2)}}%p`, "pos"],
        ["15.9년 최대낙폭(MDD)", `${{m16.mdd_pct.toFixed(1)}}%`, `순수추세: ${{sPure.metrics_11_26.mdd_pct.toFixed(1)}}%`, "neg"],
        ["초과샤프 / 칼마", `${{m16.ex_sharpe.toFixed(2)}} / ${{m16.calmar.toFixed(2)}}`, `연 변동성 ${{m16.vol_pct.toFixed(1)}}%`, ""],
        ["2025~26 랠리 구간", `${{m16.rally_cagr_pct >= 0 ? '+' : ''}}${{m16.rally_cagr_pct.toFixed(1)}}%`, `구간 MDD ${{m16.rally_mdd_pct.toFixed(1)}}%`, "pos"],
        ["평균 비중 / 연회전", `${{m16.avg_weight_pct}}%`, `연 회전 ${{m16.ann_turnover.toFixed(1)}}회 (총 ${{m16.total_switches}}회)`, ""],
      ];
      document.getElementById("kpiStrip16").innerHTML = kpis16.map(([l, v, sub, cls]) => `
        <div class="stat-card"><div class="stat-label">${{l}}</div><div class="stat-val ${{cls}}">${{v}}</div><div class="stat-sub">${{sub}}</div></div>
      `).join("");

      // Yearly Returns Grid (2011~2026)
      document.getElementById("yrGrid").innerHTML = s.yearly.map(y => {{
        const cls = y.year <= 2016 ? "boxpi-yr" : (y.year >= 2025 ? "rally-yr" : "");
        return `
          <div class="yr-chip ${{cls}}">
            <div class="y">${{y.year}}년</div>
            <div class="r ${{y.ret_pct >= 0 ? 'pos' : 'neg'}}">${{y.ret_pct >= 0 ? '+' : ''}}${{y.ret_pct.toFixed(1)}}%</div>
          </div>
        `;
      }}).join("");

      // Compare Table
      document.querySelectorAll("#cmpFilterBtns .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.cmp === cmpFilter));
      let cmpIds = [];
      if (cmpFilter === "featured") {{
        cmpIds = [
          "ma100_full_pair60", "ma100_full_overlay25", "ma100_full_split75", "ma100_full_none",
          "ma100_half_pair60", "ma100_half_overlay25", "ma100_half_split75", "ma100_half_none",
          "ma100_fixed_pair60", "ma100_fixed_overlay25", "ma100_fixed_split75", "ma100_fixed_none",
        ];
      }} else if (cmpFilter === "cur_swing") {{
        cmpIds = DATA.strategy_order.filter(id => id.endsWith(`_${{selSwing}}`));
      }} else {{
        cmpIds = DATA.strategy_order;
      }}
      const tbody = document.getElementById("compareTbody");
      tbody.innerHTML = cmpIds.map(id => {{
        const st = DATA.strategies[id];
        const a14 = st.metrics_11_24;
        const a16 = st.metrics_11_26;
        const isSel = id === s.id;
        return `
          <tr class="clickable ${{isSel ? 'selected-row' : ''}}" data-ma="${{st.ma_len}}" data-kelly="${{st.kelly_id}}" data-swing="${{st.swing_id}}">
            <td>${{isSel ? "▶ " : ""}}<strong>${{st.title}}</strong></td>
            <td>${{st.pos_summary}}</td>
            <td class="col-sep pos"><strong>${{fmtPct(a14.cagr_pct)}}</strong></td>
            <td class="neg">${{a14.mdd_pct.toFixed(1)}}%</td>
            <td><strong>${{a14.ex_sharpe.toFixed(2)}}</strong></td>
            <td>${{a14.boxpi_cagr_pct >= 0 ? '+' : ''}}${{a14.boxpi_cagr_pct.toFixed(1)}}%/${{a14.boxpi_mdd_pct.toFixed(0)}}%</td>
            <td class="col-sep pos"><strong>${{fmtPct(a16.cagr_pct)}}</strong></td>
            <td class="neg">${{a16.mdd_pct.toFixed(1)}}%</td>
            <td><strong>${{a16.ex_sharpe.toFixed(2)}}</strong></td>
            <td>${{a16.calmar.toFixed(2)}}</td>
          </tr>
        `;
      }}).join("");
      tbody.querySelectorAll("tr.clickable").forEach(tr => {{
        tr.addEventListener("click", () => {{
          selMa = Number(tr.dataset.ma);
          selKelly = tr.dataset.kelly;
          selSwing = tr.dataset.swing;
          renderAll();
        }});
      }});

      // Breakdown & Events
      const ad = s.asset_days;
      const bdKeys = s.swing_id === "pair60" ? ["k200", "kq", "pair_short", "swing", "kofr"] : ["k200", "kq", "swing", "kofr"];
      document.getElementById("breakdownStrip").innerHTML = bdKeys.map(k => {{
        const b = ad[k];
        return `
          <div class="stat-card" style="background:#f9fafb;">
            <div class="stat-label">${{b.name}}</div>
            <div class="stat-val" style="font-size:17px;">${{b.days.toLocaleString()}}일 (${{b.pct}}%)</div>
            <div class="stat-sub">15.9년 보유 일수 비중</div>
          </div>
        `;
      }}).join("");

      document.querySelectorAll("#evFilterBtns .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.ev === evFilter));
      const evList = evFilter === "switches" ? s.switch_events : (evFilter === "swing" ? s.swing_events : s.recent_events);
      const modeLbl = evFilter === "switches" ? "추세 종목 전환 이력" : (evFilter === "swing" ? "코스닥150 단타 스윙 이력" : "전체 매매·스윙·비중조절 이력");
      document.getElementById("eventsTitle").textContent = `🔄 ${{modeLbl}} (${{s.title}}, 최근 ${{evList.length}}회)`;
      document.getElementById("eventsTbody").innerHTML = evList.length === 0
        ? '<tr><td colspan="5" style="text-align:center;color:#6b7280;">해당 필터에 최근 이력이 없습니다.</td></tr>'
        : evList.map(ev => `
          <tr>
            <td>${{ev.date}}</td>
            <td>${{ev.from}} &rarr; <strong>${{ev.to}}</strong></td>
            <td>${{ev.reason}}</td>
            <td>${{ev.k200_mom6m !== null ? (ev.k200_mom6m >= 0 ? '+' : '') + ev.k200_mom6m.toFixed(1) + '%' : '-'}} / ${{ev.kq_mom6m !== null ? (ev.kq_mom6m >= 0 ? '+' : '') + ev.kq_mom6m.toFixed(1) + '%' : '-'}}</td>
            <td>${{ev.kq_rsi5 !== undefined ? ev.kq_rsi5.toFixed(1) : '-'}}</td>
          </tr>
        `).join("");

      renderDualCharts(s, sPure);
    }}

    function renderCalculator(s) {{
      const capMan = Math.max(0, Number(document.getElementById("capInput").value) || 0);
      const capWon = capMan * 10000;
      const items = [
        {{ key: "k200", name: "KODEX 200", ticker: "069500", pct: s.allocation.k200_pct, prev: s.prev_allocation.k200_pct, price: s.k200.nav_close }},
        {{ key: "kq", name: "KODEX 코스닥150", ticker: "229200", pct: s.allocation.kq_pct, prev: s.prev_allocation.kq_pct, price: s.kq.nav_close }},
      ];
      if (s.swing_id === "pair60" || s.allocation.k200_short_pct > 0 || s.allocation.kq_short_pct > 0) {{
        items.push(
          {{ key: "k200s", name: "KODEX 인버스 (1x 숏)", ticker: "114800", pct: s.allocation.k200_short_pct || 0, prev: s.prev_allocation.k200_short_pct || 0, price: 0 }},
          {{ key: "kqs", name: "코스닥150선물인버스 (1x 숏)", ticker: "251340", pct: s.allocation.kq_short_pct || 0, prev: s.prev_allocation.kq_short_pct || 0, price: 0 }}
        );
      }}
      items.push({{ key: "kofr", name: "KODEX KOFR금리액티브", ticker: "423160", pct: s.allocation.kofr_pct, prev: s.prev_allocation.kofr_pct, price: s.kofr_nav }});

      document.getElementById("allocGrid").innerHTML = items.map(it => {{
        const amtWon = Math.round(capWon * it.pct / 100);
        const sharesTxt = it.price > 0 ? `(약 ${{Math.floor(amtWon / it.price).toLocaleString()}}주)` : `(1x 인버스 · 세후 15.4%)`;
        const delta = it.pct - it.prev;
        return `
          <div class="alloc-box ${{it.pct > 0 ? 'active-alloc' : ''}}">
            <div class="alloc-name">${{it.name}} <span class="ticker">${{it.ticker}}</span></div>
            <div class="alloc-pct">${{fmtW(it.pct)}}</div>
            <div class="alloc-delta">${{fmtDelta(delta)}}</div>
            <div class="alloc-krw">${{(amtWon / 10000).toLocaleString(undefined, {{maximumFractionDigits:1}})}}만 원 ${{sharesTxt}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderAssetCard(d, maLen, kellyLabel, curPct, isKq) {{
      const selBadge = curPct > 0 ? `<span class="pill pill-blue">보유 ${{fmtW(curPct)}}</span>` : `<span class="pill pill-gray">대기 0%</span>`;
      const trBadge = d.trend_on ? `<span class="pill pill-green">MA${{maLen}} ON</span>` : `<span class="pill pill-red">MA${{maLen}} OFF</span>`;
      const rsiBadge = isKq ? `<span class="pill ${{d.rsi5 < 20 ? 'pill-red' : 'pill-purple'}}">RSI(5) ${{d.rsi5.toFixed(1)}}</span>` : "";
      const streakTxt = d.above_streak > 0 ? `상회 ${{d.above_streak}}일` : `하회 ${{d.below_streak}}일`;
      return `
        <div class="card" style="border:${{curPct > 0 ? '2px solid #2563eb' : '1px solid #e5e7eb'}};">
          <div class="card-head">
            <div>
              <div class="asset-title">${{d.name}} <span class="ticker">${{d.ticker}}</span></div>
              <div class="sub">기준가 ${{d.nav_close.toLocaleString()}}원 · 방어선(MA${{maLen}} -1%) ≈ ${{d.stop_nav.toLocaleString()}}원</div>
            </div>
            <div class="badges">${{selBadge}} ${{trBadge}} ${{rsiBadge}}</div>
          </div>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">전일 등락</div><div class="kpi-val ${{d.ret_1d_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(d.ret_1d_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">6M 모멘텀</div><div class="kpi-val ${{d.mom_6m_pct > 0 ? 'pos' : 'neg'}}">${{fmtPct(d.mom_6m_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">MA${{maLen}} 괴리</div><div class="kpi-val ${{d.disp_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(d.disp_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">축소기대수익(μ)</div><div class="kpi-val ${{d.ann_excess_mu_pct >= 0 ? 'pos' : 'neg'}}">${{d.ann_excess_mu_pct >= 0 ? '+' : ''}}${{d.ann_excess_mu_pct.toFixed(1)}}%</div></div>
            <div class="kpi"><div class="kpi-label">보수변동성(σ)</div><div class="kpi-val">${{d.ann_vol_60d_pct.toFixed(1)}}%</div></div>
            <div class="kpi"><div class="kpi-label">원시 f* / 적용비중</div><div class="kpi-val" style="font-size:14.5px;">${{d.raw_kelly_f.toFixed(2)}} → ${{d.kelly_target_pct}}%</div></div>
          </div>
          <div class="sub" style="margin-top:8px;">MA${{maLen}} 연속 상태: <strong>${{streakTxt}}</strong> · 선택 추세(${{kellyLabel}}) 산출 비중: <strong>${{d.kelly_target_pct}}%</strong></div>
        </div>
      `;
    }}

    function buildChartPair(eqId, ddId, labels, sData, sDd, pData, pDd, bh2, bhq, sTitle, sCagr, sMdd, pCagr, pMdd, showPure) {{
      const eqSets = [
        {{ label: `${{sTitle}} (세후 ${{fmtPct(sCagr)}})`, data: sData, borderColor: "#dc2626", borderWidth: 2.3, pointRadius: 0, tension: 0.1 }},
      ];
      if (showPure) {{
        eqSets.push({{ label: `순수 추세 비교 (세후 ${{fmtPct(pCagr)}})`, data: pData, borderColor: "#f59e0b", borderWidth: 1.5, borderDash: [5,3], pointRadius: 0, tension: 0.1 }});
      }}
      eqSets.push(
        {{ label: "KODEX 200 단순보유", data: bh2, borderColor: "#2563eb", borderWidth: 1.3, borderDash: [3,3], pointRadius: 0, tension: 0.1 }},
        {{ label: "코스닥 단순보유", data: bhq, borderColor: "#9ca3af", borderWidth: 1.1, pointRadius: 0, tension: 0.1 }}
      );
      const eqInst = new Chart(document.getElementById(eqId), {{
        type: "line",
        data: {{ labels, datasets: eqSets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{ tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Number(ctx.parsed.y).toLocaleString()}}만 원` }} }} }},
          scales: {{ y: {{ type: "logarithmic", ticks: {{ callback: v => Number(v).toLocaleString() + "만" }} }}, x: {{ ticks: {{ maxTicksLimit: 12 }} }} }}
        }}
      }});

      const ddSets = [
        {{ label: `${{sTitle}} 낙폭 (MDD ${{sMdd.toFixed(1)}}%)`, data: sDd, borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,0.12)", fill: true, borderWidth: 1.5, pointRadius: 0 }},
      ];
      if (showPure) {{
        ddSets.push({{ label: `순수 추세 낙폭 (MDD ${{pMdd.toFixed(1)}}%)`, data: pDd, borderColor: "#f59e0b", borderDash: [4,3], borderWidth: 1.2, pointRadius: 0 }});
      }}
      const ddInst = new Chart(document.getElementById(ddId), {{
        type: "line",
        data: {{ labels, datasets: ddSets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{ tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Number(ctx.parsed.y).toFixed(2)}}%` }} }} }},
          scales: {{ y: {{ max: 0, ticks: {{ callback: v => v + "%" }} }}, x: {{ ticks: {{ maxTicksLimit: 12 }} }} }}
        }}
      }});
      return [eqInst, ddInst];
    }}

    function renderDualCharts(s, sPure) {{
      if (eq14Inst) eq14Inst.destroy();
      if (dd14Inst) dd14Inst.destroy();
      if (eq16Inst) eq16Inst.destroy();
      if (dd16Inst) dd16Inst.destroy();

      const cut = DATA.benchmark.end_2024_pos + 1;
      const showPure = s.swing_id !== "none";

      [eq14Inst, dd14Inst] = buildChartPair(
        "eqChart14", "ddChart14",
        DATA.benchmark.dates.slice(0, cut),
        s.series.strategy.slice(0, cut), s.series.dd_pct.slice(0, cut),
        sPure.series.strategy.slice(0, cut), sPure.series.dd_pct.slice(0, cut),
        DATA.benchmark.bh_k200.slice(0, cut), DATA.benchmark.bh_kq.slice(0, cut),
        s.title, s.metrics_11_24.cagr_pct, s.metrics_11_24.mdd_pct,
        sPure.metrics_11_24.cagr_pct, sPure.metrics_11_24.mdd_pct, showPure
      );

      [eq16Inst, dd16Inst] = buildChartPair(
        "eqChart16", "ddChart16",
        DATA.benchmark.dates,
        s.series.strategy, s.series.dd_pct,
        sPure.series.strategy, sPure.series.dd_pct,
        DATA.benchmark.bh_k200, DATA.benchmark.bh_kq,
        s.title, s.metrics_11_26.cagr_pct, s.metrics_11_26.mdd_pct,
        sPure.metrics_11_26.cagr_pct, sPure.metrics_11_26.mdd_pct, showPure
      );
    }}

    document.querySelectorAll("#maBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ selMa = Number(btn.dataset.ma); renderAll(); }});
    }});
    document.querySelectorAll("#kellyBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ selKelly = btn.dataset.kelly; renderAll(); }});
    }});
    document.querySelectorAll("#swingBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ selSwing = btn.dataset.swing; renderAll(); }});
    }});
    document.querySelectorAll("#evFilterBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ evFilter = btn.dataset.ev; renderAll(); }});
    }});
    document.querySelectorAll("#cmpFilterBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ cmpFilter = btn.dataset.cmp; renderAll(); }});
    }});
    document.getElementById("capInput").addEventListener("input", () => renderCalculator(getStrat(selMa, selKelly, selSwing)));

    renderAll();
  </script>
</body>
</html>
"""


def send_ntfy_notification(strategies: dict, live_swing: dict):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("[알림] NTFY_TOPIC 환경변수가 없어 푸시 발송을 건너뜁니다.")
        return

    default_id = STRATEGY_CONFIG["default_strategy_id"]
    raw_id = os.environ.get("NTFY_STRATEGY_ID", default_id).strip() or default_id
    if raw_id not in strategies and f"{raw_id}_overlay25" in strategies:
        raw_id = f"{raw_id}_overlay25"
    s = strategies.get(raw_id) or strategies[default_id]

    force_send = os.environ.get("NTFY_FORCE_SEND", "false").lower() in ("1", "true", "yes")
    now_kst = dt.datetime.utcnow() + dt.timedelta(hours=9)
    is_monday = (now_kst.weekday() == 0)
    is_signal = (s["action_code"] != "HOLD")
    is_rebal_eve = (s["bdays_to_rebal"] == 1)

    if not (is_signal or is_rebal_eve or is_monday or force_send):
        print(f"[알림] {s['title']} 오늘 매매 없음(HOLD) -> 푸시 생략")
        return

    dashboard_url = "https://bokseop.github.io/kr-index-momentum-signal/"
    if is_signal:
        priority = 4
        tags = ["rotating_light", "zap" if "SWING" in s["action_code"] else "chart_with_upwards_trend"]
        title = f"[매매 시그널] {s['action_title']}"
    elif force_send:
        priority = 3
        tags = ["white_check_mark", "bell"]
        title = f"[2011~2026 장기검증 반영] {s['title']}"
    elif is_rebal_eve:
        priority = 3
        tags = ["calendar", "chart_with_upwards_trend"]
        title = f"[월말 D-1] 내일({s['next_rebal_date']}) 정기 리밸런싱 예정"
    else:
        priority = 2
        tags = ["bar_chart"]
        title = f"[주간 리포트] {s['title']} 정상 감시 중"

    k2 = s["k200"]
    kq = s["kq"]
    msg = (
        f"기준일: {s['as_of_date']} ({s['title']})\n"
        f"오늘 포지션: {s['pos_summary']}\n"
        f"액션: {s['action_title']} ({s['action_detail']})\n"
        f"──────────────\n"
        f"코스닥150 단타 RSI(5): {live_swing['rsi5']:.1f} ({live_swing['status_text']})\n"
        f"KODEX 200: 6M {k2['mom_6m_pct']:+.1f}% | MA{s['ma_len']} {k2['disp_pct']:+.1f}% ({'ON' if k2['trend_on'] else 'OFF'})\n"
        f"코스닥150: 6M {kq['mom_6m_pct']:+.1f}% | MA{s['ma_len']} {kq['disp_pct']:+.1f}% ({'ON' if kq['trend_on'] else 'OFF'})\n"
        f"다음 월말 리밸런싱: {s['next_rebal_date']} (D-{s['bdays_to_rebal']}영업일)"
    )

    payload = {
        "topic": topic,
        "title": title,
        "message": msg,
        "priority": priority,
        "tags": tags,
        "click": dashboard_url,
        "actions": [{"action": "view", "label": "대시보드 열기", "url": dashboard_url}],
    }
    try:
        r = requests.post(
            "https://ntfy.sh",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=15,
        )
        r.raise_for_status()
        print(f"[알림 발송 완료] ntfy 푸시 전송 성공 (HTTP {r.status_code}, priority={priority})")
    except Exception as e:
        print(f"[알림 경고] ntfy 발송 실패: {e}")


def main():
    print("[1/4] KOFIA 전자공시(DIS) 공식 기준가격 갱신 및 2009~2026 장기 시계열 연결 중...")
    s = get_session()
    aligned_data = {}
    for key, meta in ETF_META.items():
        raw = update_csv_data(s, meta)
        aligned_data[key] = reconstruct_total_return_nav(raw)
        if key == "kq":
            aligned_data[key] = splice_kosdaq_2009(aligned_data[key])
        print(f"  - {meta['name']} ({meta['ticker']}): {len(aligned_data[key])}영업일 ({aligned_data[key].index[0].strftime('%Y-%m-%d')} ~ {aligned_data[key].index[-1].strftime('%Y-%m-%d')})")

    print("[2/4] 코스닥150 RSI(5)<20 1주일 분할 스윙 엔진 계산 중 (2011~2024 / 2011~2026)...")
    idx_common = aligned_data["k200"].index.intersection(aligned_data["kq"].index).sort_values()
    bt_idx = idx_common[idx_common >= pd.Timestamp(STRATEGY_CONFIG["bt_start"])]
    tax = STRATEGY_CONFIG["tax_rate"]
    r_kq_at_bt = aligned_data["kq"].loc[bt_idx, "total_ret"] - tax * aligned_data["kq"].loc[bt_idx, "div_ret"]
    swing_bundle = compute_kq_swing_signal(aligned_data["kq"]["adj_close"].loc[idx_common], r_kq_at_bt, bt_idx)
    ls = swing_bundle["live"]
    print(f"  - 11~24(14년): {ls['stats_11_24']} | 11~26(15.9년): {ls['stats_11_26']}")

    print("[3/4] 48개 전략 조합 계산 중 (MA 3종 × 켈리 4종 × 스윙·페어롱숏 4종, 2011~2024 & 2011~2026)...")
    strategies = {}
    order = []
    bench = None
    for ma in STRATEGY_CONFIG["ma_options"]:
        for k_opt in STRATEGY_CONFIG["kelly_options"]:
            for sw_opt in STRATEGY_CONFIG["swing_options"]:
                res, b = run_strategy_variant(aligned_data, ma_len=ma, kelly_opt=k_opt, swing_opt=sw_opt, swing_bundle=swing_bundle)
                strategies[res["id"]] = res
                order.append(res["id"])
                if bench is None:
                    bench = b

    now_kst = (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")
    default_id = STRATEGY_CONFIG["default_strategy_id"]
    default_s = strategies[default_id]
    payload = {
        "generated_at_kst": now_kst,
        "as_of_date": default_s["as_of_date"],
        "default_strategy_id": default_id,
        "strategy_order": order,
        "live_swing": ls,
        "benchmark": bench,
        "strategies": strategies,
    }

    print("[4/4] 인터랙티브 대시보드(docs/index.html) & JSON(docs/latest_signal.json) 발행 중...")
    slim_json = {
        "generated_at_kst": now_kst,
        "as_of_date": default_s["as_of_date"],
        "default_strategy_id": default_id,
        "live_swing": ls,
        "strategies": {
            k: {kk: vv for kk, vv in v.items() if kk != "series"}
            for k, v in strategies.items()
        },
    }
    json_path = os.path.join(DOCS_DIR, "latest_signal.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_json, f, ensure_ascii=False, indent=2)

    html = build_interactive_html(payload)
    html_path = os.path.join(DOCS_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    send_ntfy_notification(strategies, ls)

    print("\n" + "=" * 116)
    print(f" [옵션 A 듀얼 장기 검증] 기준일: {default_s['as_of_date']} (검증1: 2011~2024 14년 | 검증2: 2011~2026 15.9년)")
    print("=" * 116)
    for sid in [
        "ma100_full_pair60", "ma100_full_overlay25", "ma100_full_split75", "ma100_full_none",
        "ma100_half_pair60", "ma100_half_overlay25", "ma100_half_split75", "ma100_half_none",
        "ma100_fixed_pair60", "ma100_fixed_overlay25", "ma100_fixed_split75", "ma100_fixed_none",
    ]:
        st = strategies[sid]
        a14 = st["metrics_11_24"]
        a16 = st["metrics_11_26"]
        print(
            f" - {st['title']:32s} | "
            f"11~24: {a14['cagr_pct']:+5.2f}% / {a14['mdd_pct']:5.1f}% / Shp {a14['ex_sharpe']:.2f} (박스 {a14['boxpi_cagr_pct']:+.1f}%) | "
            f"11~26: {a16['cagr_pct']:+5.2f}% / {a16['mdd_pct']:5.1f}% / Shp {a16['ex_sharpe']:.2f}"
        )
    print("=" * 116)
    print(f"[완료] {html_path} ({os.path.getsize(html_path)//1024}KB) 발행 완료.")


if __name__ == "__main__":
    main()
