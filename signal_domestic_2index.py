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
# 해법 1 (+ 켈리 베팅 선택형): 일반계좌 · 노레버리지(1x) · 국내 2대 지수 비과세 전략
# ==============================================================================
ETF_META = {
    "k200": {
        "name": "KODEX 200",
        "ticker": "069500",
        "kofia_cd": "KR5105352888",
        "csv": "kodex_200_nav.csv",
        "start_year": 2015,
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
        {"id": "fixed",   "label": "고정 100%",        "frac": None, "desc": "선정 지수에 100% 고정 투자 (최고 수익, MDD -30%)"},
        {"id": "full",    "label": "풀 켈리 (1.0x)",    "frac": 1.00, "desc": "축소추정 켈리 f* 100% 적용 (고수익·샤프 1.25, 보유시 평균 ~88%)"},
        {"id": "half",    "label": "하프 켈리 (0.5x)",  "frac": 0.50, "desc": "축소추정 켈리 f* 50% 적용 (연 ~16% · MDD -16%대, 칼마 0.95 권장)"},
        {"id": "quarter", "label": "쿼터 켈리 (0.25x)", "frac": 0.25, "desc": "축소추정 켈리 f* 25% 적용 (초방어 MDD -10%대 · 칼마 1.05)"},
    ],
    "kelly_mu_win": 120,       # 단기 추세 초과수익 보조 추정 기간 (120영업일)
    "kelly_erp_prior": 0.07,   # 장기 주식위험프리미엄 사전분포 (연 7%, 단기 급등 착시 방지)
    "kelly_mu_cap": 0.14,      # 기대초과수익률 상한 (연 14% 캡 -> 100% 포화 현상 방지)
    "kelly_vol_fast": 20,      # 단기 급등락 변동성 감지 기간 (20영업일)
    "kelly_vol_win": 60,       # 중기 변동성 추정 기간 (60영업일)
    "kelly_rebal_band": 0.10,  # 월중 켈리 비중 ±10%p 이상 변화 시만 리밸런싱
    "confirm_days": 3,         # 3영업일 연속 돌파/이탈 확인
    "breakout_pct": 0.01,      # ±1% 이상 즉시 돌파/이탈
    "fee_oneway": 0.0008,      # 편도 8bp (수수료+슬리피지)
    "tax_rate": 0.154,         # 일반계좌 배당/이자소득세 15.4%
    "bt_start": "2016-06-29",  # 백테스트 공통 시작일
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


def run_strategy_variant(data: dict, ma_len: int, kelly_opt: dict):
    """
    해법 1 (고정 100% 또는 연속 켈리 사이징 0~100%) 단일 조합 실행.
    - 연속 켈리 f* = (연율화 초과수익률 μ_120 - rf) / (연율화 분산 σ_60^2)
    - 노레버리지 1x 일반계좌: equity_weight = clip(kelly_frac * f*, 0%, 100%), 잔여 비중 = KOFR
    - 리밸런싱 밴드 ±10%p & 5% 단위 반올림으로 실전 회전율 최소화
    """
    mom_days = STRATEGY_CONFIG["mom_days"]
    mu_win = STRATEGY_CONFIG["kelly_mu_win"]
    vol_win = STRATEGY_CONFIG["kelly_vol_win"]
    rebal_band = STRATEGY_CONFIG["kelly_rebal_band"]
    kelly_frac = kelly_opt["frac"]

    idx = data["k200"].index.intersection(data["kq"].index).sort_values()
    k200_c = data["k200"]["adj_close"].loc[idx]
    kq_c = data["kq"]["adj_close"].loc[idx]
    r_k2 = data["k200"]["total_ret"].loc[idx]
    r_kq = data["kq"]["total_ret"].loc[idx]

    m_k200 = k200_c.pct_change(mom_days)
    m_kq = kq_c.pct_change(mom_days)
    st_k200 = compute_ma_state(k200_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])
    st_kq = compute_ma_state(kq_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])

    kofr_ret = data["kofr"]["total_ret"].reindex(idx)
    cd91_daily = (1.0 + data["kq"]["cd91"].reindex(idx).ffill().bfill() / 100.0) ** (1.0 / 252.0) - 1.0
    kofr_ret = kofr_ret.fillna(cd91_daily)

    # 축소추정(Shrinkage) 연속 켈리:
    # 단기 120일 급등(+150~200%)을 미래 기대수익으로 과대추정해 f*가 100% 상한에 포화되는 현상을 방지.
    # μ_shrink = 0.50*ERP(7%) + 0.25*clip(장기확장초과수익, 0~14%) + 0.25*clip(120일초과수익, 0~14%)
    # σ_cons = max(σ_20일, σ_60일) -> 변동성 급등 초입에 즉시 비중 자동 축소!
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
    n = len(bt_idx)

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
    r_k2_at = data["k200"].loc[bt_idx, "total_ret"] - tax * data["k200"].loc[bt_idx, "div_ret"]
    r_kq_at = data["kq"].loc[bt_idx, "total_ret"] - tax * data["kq"].loc[bt_idx, "div_ret"]
    r_kofr_at = kofr_ret.loc[bt_idx] * (1.0 - tax)

    init_cap = 10_000_000.0
    eq = np.zeros(n)
    eq[0] = init_cap
    bh_k200 = np.zeros(n); bh_k200[0] = init_cap
    bh_kq = np.zeros(n); bh_kq[0] = init_cap
    rets = np.zeros(n)
    w_hist = np.zeros(n)

    cur_asset = "kofr"
    cur_w = 0.0
    turns = 0.0
    events = []
    reasons = []
    holdings_hist = []

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

        mk2, mkq = m_k200.loc[d], m_kq.loc[d]
        on_k2, on_kq = int(st_k200.loc[d, "trend_on"]), int(st_kq.loc[d, "trend_on"])
        prev_asset, prev_w = cur_asset, cur_w
        reason = "기존 비중 유지"

        # 2. 종목 선정 & 일간 MA 방어
        if is_me[i]:
            cands = []
            if not np.isnan(mk2) and mk2 > 0 and on_k2 == 1:
                cands.append((mk2, "k200"))
            if not np.isnan(mkq) and mkq > 0 and on_kq == 1:
                cands.append((mkq, "kq"))
            cands.sort(reverse=True)
            new_asset = cands[0][1] if cands else "kofr"
            if new_asset != prev_asset:
                reason = "월말 모멘텀 선정" if new_asset != "kofr" else "월말 전 종목 추세이탈 → KOFR 대피"
            else:
                reason = "월말 정기 점검 (동일 종목)"
        else:
            new_asset = cur_asset
            if cur_asset == "k200" and on_k2 == 0:
                new_asset = "kofr"
                reason = f"월중 KODEX 200 MA{ma_len} 이탈 → 즉시 KOFR 대피"
            elif cur_asset == "kq" and on_kq == 0:
                new_asset = "kofr"
                reason = f"월중 코스닥150 MA{ma_len} 이탈 → 즉시 KOFR 대피"

        # 3. 켈리 포지션 사이징
        raw_f = float(fstar_k2.loc[d]) if new_asset == "k200" else (float(fstar_kq.loc[d]) if new_asset == "kq" else 0.0)
        if new_asset == "kofr":
            new_w = 0.0
        elif kelly_frac is None:
            new_w = 1.0
        else:
            target_w = float(np.clip(raw_f * kelly_frac, 0.0, 1.0))
            if new_asset == prev_asset and not is_me[i] and abs(target_w - prev_w) < rebal_band:
                new_w = prev_w
            else:
                new_w = round(target_w * 20) / 20.0
                if new_asset == prev_asset and abs(new_w - prev_w) >= 0.01:
                    direction = "확대" if new_w > prev_w else "축소"
                    reason = f"켈리 비중 {direction} ({int(prev_w*100)}% → {int(new_w*100)}%, f*={raw_f:.2f})"
                if new_w == 0.0:
                    new_asset = "kofr"
                    reason = f"켈리 기대우위 소멸(f*={raw_f:.2f}) → KOFR 대피"

        # 4. 회전율 및 비용
        if i > 0:
            if new_asset == prev_asset:
                t = abs(new_w - prev_w)
            else:
                t = 0.5 * (prev_w + new_w + abs((1.0 - new_w) - (1.0 - prev_w)))
        else:
            t = 0.0

        turns += t
        rets[i] = gross - t * fee
        if i > 0:
            eq[i] = eq[i - 1] * (1.0 + rets[i])
            bh_k200[i] = bh_k200[i - 1] * (1.0 + float(r_k2_at.iloc[i]))
            bh_kq[i] = bh_kq[i - 1] * (1.0 + float(r_kq_at.iloc[i]))

        w_hist[i] = new_w
        holdings_hist.append((new_asset, new_w))
        reasons.append(reason)

        if i > 0 and (new_asset != prev_asset or abs(new_w - prev_w) >= 0.01):
            from_txt = f"{ETF_META[prev_asset]['name']} {int(round(prev_w*100))}%" if prev_asset != "kofr" else "KOFR 100%"
            to_txt = f"{ETF_META[new_asset]['name']} {int(round(new_w*100))}%" + (f" + KOFR {int(round((1-new_w)*100))}%" if 0 < new_w < 1 else "") if new_asset != "kofr" else "KOFR 100%"
            events.append({
                "date": d.strftime("%Y-%m-%d"),
                "from": from_txt,
                "to": to_txt,
                "reason": reason,
                "k200_mom6m": round(float(mk2) * 100, 1) if not np.isnan(mk2) else None,
                "kq_mom6m": round(float(mkq) * 100, 1) if not np.isnan(mkq) else None,
                "kelly_f": round(raw_f, 2),
            })

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

    # Pre-2025 (2016~2024, 8.5년 — 2025년 폭등 제외 검증)
    r_p25 = ret_s.loc[:"2024-12-31"]
    eq_p25 = (1 + r_p25).cumprod()
    y_p25 = (r_p25.index[-1] - r_p25.index[0]).days / 365.25
    cagr_p25 = eq_p25.iloc[-1] ** (1.0 / y_p25) - 1.0
    mdd_p25 = (eq_p25 / eq_p25.cummax() - 1.0).min()

    # 연도별 수익률
    yearly = []
    for yr, grp in ret_s.groupby(ret_s.index.year):
        yearly.append({"year": int(yr), "ret_pct": round(float((1 + grp).prod() - 1) * 100, 1)})

    # 최신일 스냅샷 & 오늘 액션 판정
    ld = bt_idx[-1]
    lat_asset, lat_w = holdings_hist[-1]
    prv_asset, prv_w = holdings_hist[-2] if n >= 2 else (lat_asset, lat_w)

    if lat_asset != prv_asset:
        if lat_asset == "kofr":
            action_code = "EXIT_TO_CASH"
            action_title = "즉시 전량 매도 → KOFR 대피"
        elif prv_asset == "kofr":
            action_code = "NEW_ENTRY"
            action_title = f"신규 매수 ({ETF_META[lat_asset]['name']} {int(round(lat_w*100))}%)"
        else:
            action_code = "SWITCH"
            action_title = f"종목 교체 ({ETF_META[prv_asset]['name']} → {ETF_META[lat_asset]['name']} {int(round(lat_w*100))}%)"
    elif abs(lat_w - prv_w) >= 0.01:
        delta_p = int(round((lat_w - prv_w) * 100))
        action_code = "REBALANCE_WEIGHT"
        action_title = f"켈리 비중 {'확대' if delta_p > 0 else '축소'} ({delta_p:+d}%p)"
    else:
        action_code = "HOLD"
        action_title = "오늘 매매 없음 (기존 포지션 유지)"

    month_end_bday = ld + pd.offsets.BMonthEnd(0)
    if month_end_bday <= ld:
        month_end_bday = ld + pd.offsets.BMonthEnd(1)
    bdays_to_me = int(len(pd.bdate_range(ld, month_end_bday)) - 1)

    alloc = {
        "k200_pct": int(round(lat_w * 100)) if lat_asset == "k200" else 0,
        "kq_pct": int(round(lat_w * 100)) if lat_asset == "kq" else 0,
        "kofr_pct": int(round((1.0 - lat_w) * 100)) if lat_asset != "kofr" else 100,
    }
    prev_alloc = {
        "k200_pct": int(round(prv_w * 100)) if prv_asset == "k200" else 0,
        "kq_pct": int(round(prv_w * 100)) if prv_asset == "kq" else 0,
        "kofr_pct": int(round((1.0 - prv_w) * 100)) if prv_asset != "kofr" else 100,
    }

    def asset_live_info(key: str, close_s, mom_s, st_df, mu_s, vol_s, fstar_s):
        f_val = float(fstar_s.loc[ld])
        k_w = 100 if kelly_frac is None else int(round(float(np.clip(f_val * kelly_frac, 0.0, 1.0)) * 20) * 5)
        nav_c = float(data[key].loc[ld, "nav_close"])
        adj_c = float(close_s.loc[ld])
        ma_adj = float(st_df.loc[ld, "ma"])
        # 기준가(NAV) 환산 MA선 및 -1% 이탈선 가격
        ratio_nav = nav_c / adj_c if adj_c > 0 else 1.0
        ma_nav = ma_adj * ratio_nav
        return {
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

    # 차트용 주간 샘플링 (5영업일 + 마지막 날)
    dd_s = (eq_s / eq_s.cummax() - 1.0) * 100.0
    sample_idx = list(range(0, n, 5))
    if sample_idx[-1] != n - 1:
        sample_idx.append(n - 1)

    strategy_id = f"ma{ma_len}_{kelly_opt['id']}"
    result = {
        "id": strategy_id,
        "ma_len": ma_len,
        "kelly_id": kelly_opt["id"],
        "kelly_label": kelly_opt["label"],
        "kelly_frac": kelly_frac,
        "kelly_desc": kelly_opt["desc"],
        "title": f"MA{ma_len} + {kelly_opt['label']}",
        "as_of_date": ld.strftime("%Y-%m-%d"),
        "holding_key": lat_asset,
        "holding_name": ETF_META[lat_asset]["name"],
        "holding_ticker": ETF_META[lat_asset]["ticker"],
        "equity_weight_pct": int(round(lat_w * 100)),
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
        "metrics": {
            "bt_start": bt_idx[0].strftime("%Y-%m-%d"),
            "bt_end": ld.strftime("%Y-%m-%d"),
            "years": round(years, 2),
            "final_equity_man": round(float(eq_s.iloc[-1]) / 1e4, 1),
            "cagr_pct": round(cagr * 100, 2),
            "mdd_pct": round(mdd * 100, 2),
            "vol_pct": round(vol * 100, 2),
            "ex_sharpe": round(ex_sharpe, 2),
            "calmar": round(calmar, 2),
            "p25_cagr_pct": round(cagr_p25 * 100, 2),
            "p25_mdd_pct": round(mdd_p25 * 100, 2),
            "avg_weight_pct": round(float(w_hist.mean()) * 100, 1),
            "ann_turnover": round(turns / years, 2),
            "total_switches": len(events),
        },
        "yearly": yearly,
        "recent_events": events[-15:][::-1],
        "series": {
            "strategy": [round(float(eq[j]) / 1e4, 1) for j in sample_idx],
            "dd_pct": [round(float(dd_s.iloc[j]), 2) for j in sample_idx],
        },
    }

    bench = {
        "dates": [bt_idx[j].strftime("%Y-%m-%d") for j in sample_idx],
        "bh_k200": [round(float(bh_k200[j]) / 1e4, 1) for j in sample_idx],
        "bh_kq": [round(float(bh_kq[j]) / 1e4, 1) for j in sample_idx],
    }
    return result, bench


def build_interactive_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>국내 2대 지수 비과세 모멘텀 + 켈리 베팅 시그널 (해법 1)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {{ --bg:#f4f6fb; --card:#ffffff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; --accent:#1d4ed8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Malgun Gothic","Apple SD Gothic Neo",sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
    .wrap {{ max-width:1120px; margin:0 auto; padding:22px 16px 56px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; margin-bottom:14px; }}
    .topbar h1 {{ margin:0; font-size:22px; font-weight:800; letter-spacing:-0.4px; }}
    .topbar .meta {{ font-size:13px; color:var(--muted); }}

    /* Strategy Selector Bar */
    .selector-card {{ background:#fff; border:1px solid var(--border); border-radius:14px; padding:16px 20px; margin-bottom:18px; box-shadow:0 2px 6px rgba(15,23,42,0.04); display:flex; flex-wrap:wrap; gap:18px; align-items:center; justify-content:space-between; }}
    .sel-group {{ display:flex; flex-direction:column; gap:6px; }}
    .sel-label {{ font-size:12px; font-weight:700; color:var(--muted); text-transform:uppercase; }}
    .btn-row {{ display:flex; gap:6px; flex-wrap:wrap; }}
    .seg-btn {{ border:1px solid #d1d5db; background:#f9fafb; color:#374151; padding:7px 13px; border-radius:9px; font-size:13px; font-weight:700; cursor:pointer; transition:all 0.15s; }}
    .seg-btn:hover {{ background:#eff6ff; border-color:#93c5fd; }}
    .seg-btn.active {{ background:#1d4ed8; color:#fff; border-color:#1d4ed8; box-shadow:0 2px 6px rgba(29,78,216,0.28); }}
    .sel-desc {{ font-size:12.5px; color:#4b5563; margin-top:2px; }}

    /* Hero Action Banner */
    .hero {{ border-radius:16px; padding:24px 26px; color:#fff; box-shadow:0 10px 25px rgba(15,23,42,0.13); margin-bottom:18px; transition:background 0.25s; }}
    .hero-top {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }}
    .hero-tag {{ background:rgba(255,255,255,0.18); padding:4px 12px; border-radius:999px; font-size:13px; font-weight:700; }}
    .hero-action-badge {{ background:#fef08a; color:#854d0e; padding:5px 13px; border-radius:999px; font-size:13px; font-weight:800; }}
    .hero-main {{ font-size:30px; font-weight:800; margin:12px 0 6px; letter-spacing:-0.6px; }}
    .hero-sub {{ font-size:14px; opacity:0.93; display:flex; flex-wrap:wrap; gap:16px; margin-top:6px; }}

    /* Today's Action & Order Calculator */
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

    /* KPI Strip */
    .kpi-strip {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(145px, 1fr)); gap:10px; margin-bottom:18px; }}
    .stat-card {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:13px 14px; }}
    .stat-label {{ font-size:12px; color:var(--muted); font-weight:600; }}
    .stat-val {{ font-size:20px; font-weight:800; margin-top:3px; }}
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
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:9px 8px; border-bottom:1px solid var(--border); text-align:left; }}
    th {{ font-size:12px; color:var(--muted); font-weight:700; background:#f9fafb; }}
    tr.clickable {{ cursor:pointer; transition:background 0.12s; }}
    tr.clickable:hover {{ background:#f3f4f6; }}
    tr.selected-row {{ background:#eff6ff !important; font-weight:700; }}
    .section {{ margin-bottom:18px; }}
    .chart-box {{ height:310px; position:relative; }}
    .chart-box-sm {{ height:180px; position:relative; margin-top:14px; }}
    .yr-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(82px, 1fr)); gap:8px; }}
    .yr-chip {{ border-radius:10px; padding:8px; text-align:center; border:1px solid var(--border); background:#f9fafb; }}
    .yr-chip .y {{ font-size:11.5px; color:var(--muted); font-weight:700; }}
    .yr-chip .r {{ font-size:15px; font-weight:800; margin-top:2px; }}
    .rules {{ font-size:13.5px; color:#374151; margin:0; padding-left:18px; }}
    .rules li {{ margin-bottom:6px; }}
    .foot {{ text-align:center; font-size:12px; color:var(--muted); margin-top:24px; }}
    @media (max-width:640px) {{ .hero-main {{ font-size:22px; }} .kpi-grid {{ grid-template-columns:repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1>📈 국내 2대 지수 비과세 모멘텀 + 켈리 베팅 시그널</h1>
      <div class="meta">기준 거래일: <strong id="asOfDate">-</strong> (KOFIA 공식 종가) · 갱신: <span id="genAt">-</span></div>
    </div>

    <!-- 전략 선택기 -->
    <div class="selector-card">
      <div class="sel-group">
        <div class="sel-label">1. 일간 추세 방어선 선택</div>
        <div class="btn-row" id="maBtns">
          <button class="seg-btn" data-ma="100">MA100 (기본 권장)</button>
          <button class="seg-btn" data-ma="80">MA80 (중립)</button>
          <button class="seg-btn" data-ma="60">MA60 (보수)</button>
        </div>
      </div>
      <div class="sel-group">
        <div class="sel-label">2. 베팅 방식 (포지션 사이징) 선택</div>
        <div class="btn-row" id="kellyBtns">
          <button class="seg-btn" data-kelly="half">하프 켈리 (0.5x 권장)</button>
          <button class="seg-btn" data-kelly="quarter">쿼터 켈리 (0.25x 방어)</button>
          <button class="seg-btn" data-kelly="full">풀 켈리 (1.0x 공격)</button>
          <button class="seg-btn" data-kelly="fixed">고정 100% (기본)</button>
        </div>
        <div class="sel-desc" id="kellyDesc">-</div>
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

    <!-- 선택 전략 백테스트 핵심 지표 -->
    <div class="kpi-strip" id="kpiStrip"></div>

    <!-- 2개 지수 실시간 켈리 & 추세 진단 -->
    <div class="grid-2" id="assetCards"></div>

    <!-- 인터랙티브 차트 -->
    <div class="card section">
      <h2 id="chartTitle">📊 10.4년 일반계좌 세후 자산 곡선 & 낙폭(DD) 비교</h2>
      <div class="chart-box"><canvas id="eqChart"></canvas></div>
      <div class="chart-box-sm"><canvas id="ddChart"></canvas></div>
    </div>

    <!-- 연도별 수익률 -->
    <div class="card section">
      <h2>📅 선택 전략 연도별 세후 수익률 (2016 ~ 2026)</h2>
      <div class="yr-grid" id="yrGrid"></div>
    </div>

    <!-- 12개 전체 전략 비교표 (클릭 선택) -->
    <div class="card section">
      <h2>🏆 전체 전략(방어선 × 켈리 베팅) 10.4년 일반계좌 세후 백테스트 비교 <span style="font-size:12px;color:#6b7280;font-weight:500;">(행을 클릭하면 즉시 전환됩니다)</span></h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>전략</th><th>오늘 목표 포지션</th><th>오늘 액션</th><th>세후 CAGR</th><th>MDD</th><th>초과샤프</th><th>칼마</th><th>16~24(8.5년)</th><th>평균비중</th><th>연회전</th>
            </tr>
          </thead>
          <tbody id="compareTbody"></tbody>
        </table>
      </div>
    </div>

    <!-- 최근 매매 / 비중조절 이력 -->
    <div class="card section">
      <h2 id="eventsTitle">🔄 최근 시그널 · 켈리 비중조절 이력 (최근 15회)</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr><th>전환일</th><th>포지션 변경</th><th>사유</th><th>6M 모멘텀 (200 / 코스닥)</th><th>당시 원시 f*</th></tr>
          </thead>
          <tbody id="eventsTbody"></tbody>
        </table>
      </div>
    </div>

    <!-- 규칙 & 켈리 공식 설명 -->
    <div class="card section">
      <h2>📐 켈리 베팅 결합 규칙 & 일반계좌 세금 포인트</h2>
      <ul class="rules">
        <li><strong>종목 선정 (매월 말):</strong> <code>KODEX 200</code>과 <code>KODEX 코스닥150</code> 중 6개월(120영업일) 수익률 &gt; 0% &amp; 일간 MA 추세 ON을 모두 만족하는 지수 중 모멘텀 1위 종목을 선택합니다. 둘 다 탈락하면 <code>KOFR</code> 100% 대피.</li>
        <li><strong>일간 MA 방어 (매일):</strong> 보유 지수가 선택 MA선(100/80/60일)을 3영업일 연속 하회하거나 종가 기준 -1% 이상 이탈 시 즉시 <code>KOFR</code> 100%로 대피합니다.</li>
        <li><strong>축소추정(Shrinkage) 연속 켈리 사이징 (노레버리지 0~100%):</strong> 단기 급등 수익률을 그대로 미래에 외삽해 100% 상한에 고착되는 오류를 막기 위해, 장기 주식위험프리미엄(7%)·과거 누적 초과수익·최근 120일 추세를 결합한 현실적 기대초과수익률(<code>μ_shrink</code>, 3.5~10.5%)을 보수적 변동성 분산(<code>max(σ_20일, σ_60일)²</code>)으로 나눠 원시 켈리 <code>f* = μ_shrink / σ²</code>를 산출합니다. 여기에 선택 배수(풀 1.0x / 하프 0.5x / 쿼터 0.25x)를 곱해 <code>0% ~ 100%</code>(5% 단위)로 배분하고 잔여 현금은 <code>KOFR</code>에 예치합니다(월중 ±10%p 이상 변동 시에만 리밸런싱).</li>
        <li><strong>일반계좌 세금 우위:</strong> <code>KODEX 200</code>·<code>KODEX 코스닥150</code>은 국내주식형 1x ETF라 <strong>매매차익이 0% 완전 비과세</strong>이며 금융소득종합과세(연 2,000만 원) 합산 대상에서 제외됩니다.</li>
      </ul>
    </div>

    <div class="foot">
      데이터 출처: 금융투자협회 전자공시(KOFIA DIS) 공식 수정기준가 · 평일 오전 08:40 KST GitHub Actions 자동 업데이트<br/>
      ※ 본 대시보드는 과거 공식 데이터 기반 전략 검증 및 참고용이며 최종 투자 판단과 책임은 투자자 본인에게 있습니다.
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    let selMa = 100;
    let selKelly = "half"; // 기본 권장: MA100 + 하프 켈리
    let eqChartInst = null;
    let ddChartInst = null;

    document.getElementById("asOfDate").textContent = DATA.as_of_date;
    document.getElementById("genAt").textContent = DATA.generated_at_kst;

    function getStrat(ma, kelly) {{
      return DATA.strategies[`ma${{ma}}_${{kelly}}`];
    }}

    function fmtPct(v) {{
      return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
    }}

    function fmtDelta(d) {{
      if (d === 0) return '<span style="color:#6b7280;">변동 없음 (0%p)</span>';
      return d > 0 ? `<span class="pos">▲ +${{d}}%p 매수</span>` : `<span class="neg">▼ ${{d}}%p 매도</span>`;
    }}

    function renderAll() {{
      const s = getStrat(selMa, selKelly);
      const sFixed = getStrat(selMa, "fixed");
      const m = s.metrics;

      // Active button styles
      document.querySelectorAll("#maBtns .seg-btn").forEach(b => b.classList.toggle("active", Number(b.dataset.ma) === selMa));
      document.querySelectorAll("#kellyBtns .seg-btn").forEach(b => b.classList.toggle("active", b.dataset.kelly === selKelly));
      document.getElementById("kellyDesc").textContent = s.kelly_desc;

      // Hero Banner
      const hero = document.getElementById("heroBox");
      let bg = "linear-gradient(135deg, #4b5563 0%, #374151 100%)", regTxt = "🛡️ 방어 국면 · 현금성 대피";
      if (s.holding_key === "k200") {{
        bg = "linear-gradient(135deg, #0f4c81 0%, #1b6ca8 100%)";
        regTxt = `🔵 공격 국면 · 대형주 주도 (${{s.title}})`;
      }} else if (s.holding_key === "kq") {{
        bg = "linear-gradient(135deg, #0e6b4e 0%, #179c72 100%)";
        regTxt = `🟢 공격 국면 · 코스닥 주도 (${{s.title}})`;
      }} else {{
        regTxt = `🛡️ 방어 국면 · KOFR 100% 대피 (${{s.title}})`;
      }}
      hero.style.background = bg;
      document.getElementById("heroRegime").textContent = regTxt;
      document.getElementById("heroActionBadge").textContent = `오늘 액션: ${{s.action_title}}`;

      const mainLine = s.holding_key === "kofr"
        ? `KODEX KOFR금리액티브 (423160) — 100%`
        : `${{s.holding_name}} (${{s.holding_ticker}}) ${{s.equity_weight_pct}}%` + (s.allocation.kofr_pct > 0 ? ` + KOFR ${{s.allocation.kofr_pct}}%` : "");
      document.getElementById("heroMain").textContent = mainLine;
      document.getElementById("heroSub").innerHTML = `
        <span>판정 근거: <strong>${{s.action_detail}}</strong></span>
        <span>다음 월말 리밸런싱: <strong>${{s.next_rebal_date}}</strong> (D-${{s.bdays_to_rebal}}영업일)</span>
        <span>10.4년 세후 CAGR: <strong>${{fmtPct(m.cagr_pct)}}</strong> (MDD ${{m.mdd_pct.toFixed(1)}}%)</span>
      `;

      // Today's Action & Order Calculator
      document.getElementById("actionReasonText").textContent = `기준일 ${{s.as_of_date}} 종가 판정: ${{s.action_title}} — ${{s.action_detail}}`;
      renderCalculator(s);

      // KPI Strip
      const kpis = [
        ["10.4년 세후 최종자산", `${{m.final_equity_man.toLocaleString()}}만 원`, "초기 1,000만 원 기준", "pos"],
        ["세후 연복리 (CAGR)", fmtPct(m.cagr_pct), `고정100% 대비 ${{(m.cagr_pct - sFixed.metrics.cagr_pct >= 0 ? "+" : "")}}${{(m.cagr_pct - sFixed.metrics.cagr_pct).toFixed(2)}}%p`, "pos"],
        ["최대낙폭 (MDD)", `${{m.mdd_pct.toFixed(1)}}%`, `고정100%: ${{sFixed.metrics.mdd_pct.toFixed(1)}}%`, "neg"],
        ["초과샤프 / 칼마", `${{m.ex_sharpe.toFixed(2)}} / ${{m.calmar.toFixed(2)}}`, `연 변동성 ${{m.vol_pct.toFixed(1)}}%`, ""],
        ["2016~24 (폭등제외)", `${{m.p25_cagr_pct >= 0 ? "+" : ""}}${{m.p25_cagr_pct.toFixed(1)}}%`, `8.5년 MDD ${{m.p25_mdd_pct.toFixed(1)}}%`, "pos"],
        ["평균 주식비중 / 회전", `${{m.avg_weight_pct}}%`, `연 회전율 ${{m.ann_turnover.toFixed(1)}}회 (총 ${{m.total_switches}}회)`, ""],
      ];
      document.getElementById("kpiStrip").innerHTML = kpis.map(([l, v, sub, cls]) => `
        <div class="stat-card">
          <div class="stat-label">${{l}}</div>
          <div class="stat-val ${{cls}}">${{v}}</div>
          <div class="stat-sub">${{sub}}</div>
        </div>
      `).join("");

      // Asset Diagnostic Cards (with Live Kelly math)
      document.getElementById("assetCards").innerHTML = [
        renderAssetCard(s.k200, s.ma_len, s.kelly_label, s.allocation.k200_pct),
        renderAssetCard(s.kq, s.ma_len, s.kelly_label, s.allocation.kq_pct),
      ].join("");

      // Yearly Returns Grid
      document.getElementById("yrGrid").innerHTML = s.yearly.map(y => `
        <div class="yr-chip">
          <div class="y">${{y.year}}년</div>
          <div class="r ${{y.ret_pct >= 0 ? 'pos' : 'neg'}}">${{y.ret_pct >= 0 ? '+' : ''}}${{y.ret_pct.toFixed(1)}}%</div>
        </div>
      `).join("");

      // Compare Table (all 12 strategies)
      const tbody = document.getElementById("compareTbody");
      tbody.innerHTML = DATA.strategy_order.map(id => {{
        const st = DATA.strategies[id];
        const sm = st.metrics;
        const isSel = id === s.id;
        const holdTxt = st.holding_key === "kofr"
          ? "KOFR 100%"
          : `${{st.holding_name}} ${{st.equity_weight_pct}}%` + (st.allocation.kofr_pct > 0 ? ` + KOFR ${{st.allocation.kofr_pct}}%` : "");
        return `
          <tr class="clickable ${{isSel ? 'selected-row' : ''}}" data-ma="${{st.ma_len}}" data-kelly="${{st.kelly_id}}">
            <td>${{isSel ? "▶ " : ""}}<strong>${{st.title}}</strong></td>
            <td>${{holdTxt}}</td>
            <td>${{st.action_code === 'HOLD' ? '유지' : '<strong>' + st.action_title + '</strong>'}}</td>
            <td class="pos"><strong>${{fmtPct(sm.cagr_pct)}}</strong></td>
            <td class="neg">${{sm.mdd_pct.toFixed(1)}}%</td>
            <td>${{sm.ex_sharpe.toFixed(2)}}</td>
            <td><strong>${{sm.calmar.toFixed(2)}}</strong></td>
            <td>${{sm.p25_cagr_pct >= 0 ? '+' : ''}}${{sm.p25_cagr_pct.toFixed(1)}}% / ${{sm.p25_mdd_pct.toFixed(0)}}%</td>
            <td>${{sm.avg_weight_pct}}%</td>
            <td>${{sm.ann_turnover.toFixed(1)}}회</td>
          </tr>
        `;
      }}).join("");
      tbody.querySelectorAll("tr.clickable").forEach(tr => {{
        tr.addEventListener("click", () => {{
          selMa = Number(tr.dataset.ma);
          selKelly = tr.dataset.kelly;
          renderAll();
        }});
      }});

      // Recent Events
      document.getElementById("eventsTitle").textContent = `🔄 최근 시그널 · 켈리 비중조절 이력 (${{s.title}}, 최근 15회)`;
      document.getElementById("eventsTbody").innerHTML = s.recent_events.map(ev => `
        <tr>
          <td>${{ev.date}}</td>
          <td>${{ev.from}} &rarr; <strong>${{ev.to}}</strong></td>
          <td>${{ev.reason}}</td>
          <td>${{ev.k200_mom6m !== null ? (ev.k200_mom6m >= 0 ? '+' : '') + ev.k200_mom6m.toFixed(1) + '%' : '-'}} / ${{ev.kq_mom6m !== null ? (ev.kq_mom6m >= 0 ? '+' : '') + ev.kq_mom6m.toFixed(1) + '%' : '-'}}</td>
          <td>${{ev.kelly_f.toFixed(2)}}</td>
        </tr>
      `).join("");

      // Charts
      renderCharts(s, sFixed);
    }}

    function renderCalculator(s) {{
      const capMan = Math.max(0, Number(document.getElementById("capInput").value) || 0);
      const capWon = capMan * 10000;
      const items = [
        {{ key: "k200", name: "KODEX 200", ticker: "069500", pct: s.allocation.k200_pct, prev: s.prev_allocation.k200_pct, price: s.k200.nav_close }},
        {{ key: "kq", name: "KODEX 코스닥150", ticker: "229200", pct: s.allocation.kq_pct, prev: s.prev_allocation.kq_pct, price: s.kq.nav_close }},
        {{ key: "kofr", name: "KODEX KOFR금리액티브", ticker: "423160", pct: s.allocation.kofr_pct, prev: s.prev_allocation.kofr_pct, price: s.kofr_nav }},
      ];
      document.getElementById("allocGrid").innerHTML = items.map(it => {{
        const amtWon = Math.round(capWon * it.pct / 100);
        const shares = it.price > 0 ? Math.floor(amtWon / it.price) : 0;
        const delta = it.pct - it.prev;
        return `
          <div class="alloc-box ${{it.pct > 0 ? 'active-alloc' : ''}}">
            <div class="alloc-name">${{it.name}} <span class="ticker">${{it.ticker}}</span></div>
            <div class="alloc-pct">${{it.pct}}%</div>
            <div class="alloc-delta">${{fmtDelta(delta)}}</div>
            <div class="alloc-krw">${{(amtWon / 10000).toLocaleString()}}만 원 (약 ${{shares.toLocaleString()}}주)</div>
          </div>
        `;
      }}).join("");
    }}

    function renderAssetCard(d, maLen, kellyLabel, curPct) {{
      const selBadge = curPct > 0 ? `<span class="pill pill-blue">보유 ${{curPct}}%</span>` : `<span class="pill pill-gray">대기 0%</span>`;
      const trBadge = d.trend_on ? `<span class="pill pill-green">MA${{maLen}} ON</span>` : `<span class="pill pill-red">MA${{maLen}} OFF</span>`;
      const streakTxt = d.above_streak > 0 ? `상회 ${{d.above_streak}}일` : `하회 ${{d.below_streak}}일`;
      return `
        <div class="card" style="border:${{curPct > 0 ? '2px solid #2563eb' : '1px solid #e5e7eb'}};">
          <div class="card-head">
            <div>
              <div class="asset-title">${{d.name}} <span class="ticker">${{d.ticker}}</span></div>
              <div class="sub">기준가 ${{d.nav_close.toLocaleString()}}원 · 방어선(MA${{maLen}} -1%) ≈ ${{d.stop_nav.toLocaleString()}}원</div>
            </div>
            <div class="badges">${{selBadge}} ${{trBadge}}</div>
          </div>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">전일 등락</div><div class="kpi-val ${{d.ret_1d_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(d.ret_1d_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">6M 모멘텀</div><div class="kpi-val ${{d.mom_6m_pct > 0 ? 'pos' : 'neg'}}">${{fmtPct(d.mom_6m_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">MA${{maLen}} 괴리</div><div class="kpi-val ${{d.disp_pct >= 0 ? 'pos' : 'neg'}}">${{fmtPct(d.disp_pct)}}</div></div>
            <div class="kpi"><div class="kpi-label">축소기대수익(μ)</div><div class="kpi-val ${{d.ann_excess_mu_pct >= 0 ? 'pos' : 'neg'}}">${{d.ann_excess_mu_pct >= 0 ? '+' : ''}}${{d.ann_excess_mu_pct.toFixed(1)}}%</div></div>
            <div class="kpi"><div class="kpi-label">보수변동성(σ)</div><div class="kpi-val">${{d.ann_vol_60d_pct.toFixed(1)}}%</div></div>
            <div class="kpi"><div class="kpi-label">원시 f* / 적용비중</div><div class="kpi-val" style="font-size:14.5px;">${{d.raw_kelly_f.toFixed(2)}} → ${{d.kelly_target_pct}}%</div></div>
          </div>
          <div class="sub" style="margin-top:8px;">MA${{maLen}} 연속 상태: <strong>${{streakTxt}}</strong> · 선택 베팅(${{kellyLabel}}) 산출 비중: <strong>${{d.kelly_target_pct}}%</strong></div>
        </div>
      `;
    }}

    function renderCharts(s, sFixed) {{
      document.getElementById("chartTitle").textContent = `📊 10.4년 일반계좌 세후 자산 곡선 & 낙폭(DD) — ${{s.title}}`;
      const labels = DATA.benchmark.dates;

      if (eqChartInst) eqChartInst.destroy();
      const datasets = [
        {{ label: `${{s.title}} (세후 ${{fmtPct(s.metrics.cagr_pct)}})`, data: s.series.strategy, borderColor: "#dc2626", borderWidth: 2.4, pointRadius: 0, tension: 0.1 }},
      ];
      if (s.kelly_id !== "fixed") {{
        datasets.push({{ label: `MA${{s.ma_len}} 고정 100% (세후 ${{fmtPct(sFixed.metrics.cagr_pct)}})`, data: sFixed.series.strategy, borderColor: "#f59e0b", borderWidth: 1.6, borderDash: [5,3], pointRadius: 0, tension: 0.1 }});
      }}
      datasets.push(
        {{ label: "KODEX 200 단순보유 (비과세)", data: DATA.benchmark.bh_k200, borderColor: "#2563eb", borderWidth: 1.4, borderDash: [3,3], pointRadius: 0, tension: 0.1 }},
        {{ label: "KODEX 코스닥150 단순보유 (비과세)", data: DATA.benchmark.bh_kq, borderColor: "#9ca3af", borderWidth: 1.2, pointRadius: 0, tension: 0.1 }}
      );

      eqChartInst = new Chart(document.getElementById("eqChart"), {{
        type: "line",
        data: {{ labels, datasets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{ tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Number(ctx.parsed.y).toLocaleString()}}만 원` }} }} }},
          scales: {{ y: {{ type: "logarithmic", ticks: {{ callback: v => Number(v).toLocaleString() + "만" }} }}, x: {{ ticks: {{ maxTicksLimit: 11 }} }} }}
        }}
      }});

      if (ddChartInst) ddChartInst.destroy();
      const ddSets = [
        {{ label: `${{s.title}} 낙폭 (MDD ${{s.metrics.mdd_pct.toFixed(1)}}%)`, data: s.series.dd_pct, borderColor: "#dc2626", backgroundColor: "rgba(220,38,38,0.12)", fill: true, borderWidth: 1.6, pointRadius: 0 }},
      ];
      if (s.kelly_id !== "fixed") {{
        ddSets.push({{ label: `MA${{s.ma_len}} 고정 100% 낙폭 (MDD ${{sFixed.metrics.mdd_pct.toFixed(1)}}%)`, data: sFixed.series.dd_pct, borderColor: "#f59e0b", borderDash: [4,3], borderWidth: 1.3, pointRadius: 0 }});
      }}
      ddChartInst = new Chart(document.getElementById("ddChart"), {{
        type: "line",
        data: {{ labels, datasets: ddSets }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{ tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{Number(ctx.parsed.y).toFixed(2)}}%` }} }} }},
          scales: {{ y: {{ max: 0, ticks: {{ callback: v => v + "%" }} }}, x: {{ ticks: {{ maxTicksLimit: 11 }} }} }}
        }}
      }});
    }}

    document.querySelectorAll("#maBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ selMa = Number(btn.dataset.ma); renderAll(); }});
    }});
    document.querySelectorAll("#kellyBtns .seg-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{ selKelly = btn.dataset.kelly; renderAll(); }});
    }});
    document.getElementById("capInput").addEventListener("input", () => renderCalculator(getStrat(selMa, selKelly)));

    renderAll();
  </script>
</body>
</html>
"""


def main():
    print("[1/3] KOFIA 전자공시(DIS) 공식 기준가격 갱신 중...")
    s = get_session()
    aligned_data = {}
    for key, meta in ETF_META.items():
        raw = update_csv_data(s, meta)
        aligned_data[key] = reconstruct_total_return_nav(raw)
        print(f"  - {meta['name']} ({meta['ticker']}): {len(aligned_data[key])}영업일 (최신 거래일: {aligned_data[key].index[-1].strftime('%Y-%m-%d')})")

    print("[2/3] 12개 전략 조합 계산 중 (MA100/80/60 × 고정100%/하프켈리/쿼터켈리/풀켈리)...")
    strategies = {}
    order = []
    bench = None
    for ma in STRATEGY_CONFIG["ma_options"]:
        for k_opt in STRATEGY_CONFIG["kelly_options"]:
            res, b = run_strategy_variant(aligned_data, ma_len=ma, kelly_opt=k_opt)
            strategies[res["id"]] = res
            order.append(res["id"])
            if bench is None:
                bench = b

    now_kst = (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST")
    default_s = strategies["ma100_half"]
    payload = {
        "generated_at_kst": now_kst,
        "as_of_date": default_s["as_of_date"],
        "default_strategy_id": "ma100_half",
        "strategy_order": order,
        "benchmark": bench,
        "strategies": strategies,
    }

    print("[3/3] 인터랙티브 대시보드(docs/index.html) & JSON(docs/latest_signal.json) 발행 중...")
    # JSON 파일은 차트 시계열 제외 요약본 + 기본/대안 전부 포함 (가볍고 API 친화적)
    slim_json = {
        "generated_at_kst": now_kst,
        "as_of_date": default_s["as_of_date"],
        "default_strategy_id": "ma100_half",
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

    print("\n" + "=" * 92)
    print(f" [해법 1 + 켈리 베팅 실시간 시그널] 기준 거래일: {default_s['as_of_date']} (일반계좌 · 1x 비과세)")
    print("=" * 92)
    print(f" ▶ 기본 권장(MA100 + 하프 켈리): {default_s['holding_name']} ({default_s['holding_ticker']}) {default_s['equity_weight_pct']}% | KOFR {default_s['allocation']['kofr_pct']}%")
    print(f" ▶ 오늘 액션: {default_s['action_title']} ({default_s['action_detail']})")
    print("-" * 92)
    for sid in ["ma100_fixed", "ma100_half", "ma100_quarter", "ma100_full", "ma80_half", "ma60_half"]:
        st = strategies[sid]
        m = st["metrics"]
        print(f" - {st['title']:22s} | 세후CAGR {m['cagr_pct']:+6.2f}% | MDD {m['mdd_pct']:5.1f}% | 칼마 {m['calmar']:.2f} | 16~24 {m['p25_cagr_pct']:+5.1f}%/{m['p25_mdd_pct']:.0f}% | 오늘: {st['holding_name']} {st['equity_weight_pct']}%")
    print("=" * 92)
    print(f"[완료] {html_path} ({os.path.getsize(html_path)//1024}KB) 및 {json_path} 발행 완료.")


if __name__ == "__main__":
    main()
