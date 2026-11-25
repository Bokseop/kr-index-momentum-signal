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
# 해법 1: 일반계좌 · 노레버리지(1x) · 국내 2대 지수(매매차익 비과세) 시그널 설정
# ==============================================================================
ETF_META = {
    "k200": {
        "name": "KODEX 200",
        "ticker": "069500",
        "kofia_cd": "KR5105352888",
        "csv": "kodex_200_nav.csv",
        "start_year": 2015,
        "taxable_capgain": False,  # 국내주식형 1x -> 매매차익 0% 비과세
    },
    "kq": {
        "name": "KODEX 코스닥150",
        "ticker": "229200",
        "kofia_cd": "K55105B74603",
        "csv": "kodex_kosdaq150_nav.csv",
        "start_year": 2015,
        "taxable_capgain": False,  # 국내주식형 1x -> 매매차익 0% 비과세
    },
    "kofr": {
        "name": "KODEX KOFR금리액티브",
        "ticker": "423160",
        "kofia_cd": "K55105DR9373",
        "csv": "kodex_kofr_nav.csv",
        "start_year": 2022,
        "taxable_capgain": True,   # 금리형 -> 이자/차익 15.4% 과세
    },
}

STRATEGY_CONFIG = {
    "mom_days": 120,          # 6개월(120영업일) 모멘텀
    "primary_ma": 100,        # 기본 일간 방어 이평선 (MA100)
    "alt_mas": [80, 60],      # 보수형 참고 방어선 (MA80, MA60)
    "confirm_days": 3,        # 3영업일 연속 돌파/이탈 확인
    "breakout_pct": 0.01,     # ±1% 이상 즉시 돌파/이탈
    "fee_oneway": 0.0008,     # 편도 8bp (수수료+슬리피지)
    "tax_rate": 0.154,        # 일반계좌 배당/이자소득세 15.4%
    "bt_start": "2016-06-29", # 백테스트 공통 시작일
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
    """KOFIA 전자공시(DIS)에서 공식 일별 기준가격·수정기준가 등락·CD91 금리 조회."""
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
    """로컬 캐시 CSV가 있으면 최근 2년만 증분 갱신, 없으면 전체 기간 수집."""
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
                # 같은 날짜는 최신(fresh) 데이터로 덮어쓰기 (마지막날 diff=0 -> 확정값 갱신)
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
    """
    KOFIA DIS 기준가격 -> 시장 T일 종가 정렬(shift(-1)) & 분배금 재투자 총수익(adj_close) 정확 복원.
    2015년 이전 상장 펀드(KODEX 200 등)의 초기 누적분배배수 F0까지 역산 반영.
    """
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

    # DIS T+1 고시 기준가 = 시장 T일 종가 NAV
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


def run_solution1_engine(data: dict, ma_len: int, mom_days: int = 120):
    """
    해법 1 (국내 2대 지수 월간 6M 모멘텀 Top-1 + 일간 MA 방어) 시뮬레이션 및 실시간 시그널 산출.
    - 매월 마지막 영업일: 6M 모멘텀 > 0 & 일간 MA 추세 ON 통과 지수 중 승자 100% 선정 (둘 다 탈락 시 KOFR)
    - 월중: 보유 지수가 일간 MA 추세 OFF로 전환되면 즉시 KOFR 100% 대피 (다음 월말까지 유지)
    - 일반계좌 실전 세법: 국내 1x 매매차익 비과세 / 분배금·KOFR 이자 15.4% 과세
    """
    idx = data["k200"].index.intersection(data["kq"].index).sort_values()
    k200_c = data["k200"]["adj_close"].loc[idx]
    kq_c = data["kq"]["adj_close"].loc[idx]

    m_k200 = k200_c.pct_change(mom_days)
    m_kq = kq_c.pct_change(mom_days)
    st_k200 = compute_ma_state(k200_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])
    st_kq = compute_ma_state(kq_c, ma_len, STRATEGY_CONFIG["confirm_days"], STRATEGY_CONFIG["breakout_pct"])

    # KOFR 수익률 (2022-04 이전은 실제 일별 CD91 금리 보간)
    kofr_ret = data["kofr"]["total_ret"].reindex(idx)
    cd91_daily = (1.0 + data["kq"]["cd91"].reindex(idx).ffill().bfill() / 100.0) ** (1.0 / 252.0) - 1.0
    kofr_ret = kofr_ret.fillna(cd91_daily)

    bt_idx = idx[idx >= pd.Timestamp(STRATEGY_CONFIG["bt_start"])]
    n = len(bt_idx)

    # 월말 여부 (주의: 마지막 거래일은 아직 월말이 아닐 수 있으므로 날짜 기준 월말 판단:
    # 과거 날짜는 다음 거래일 month != 현재 month, 마지막 날은 당월의 마지막 평일(금/말일) 여부 체크)
    is_month_end = np.zeros(n, dtype=bool)
    is_month_end[0] = True
    for i in range(n - 1):
        if bt_idx[i].month != bt_idx[i + 1].month:
            is_month_end[i] = True
    # 마지막 행이 그 달의 마지막 영업일(다음 날이 다음 달이거나, 주말 건너뛰면 다음 달)인지 체크
    last_dt = bt_idx[-1]
    next_bday = last_dt + pd.offsets.BDay(1)
    if next_bday.month != last_dt.month:
        is_month_end[-1] = True

    holdings = []
    reasons = []
    events = []

    cur_hold = "kofr"
    for i in range(n):
        d = bt_idx[i]
        mk2, mkq = m_k200.loc[d], m_kq.loc[d]
        on_k2, on_kq = int(st_k200.loc[d, "trend_on"]), int(st_kq.loc[d, "trend_on"])

        prev_hold = cur_hold
        reason = "보유 유지"

        if is_month_end[i]:
            cands = []
            if not np.isnan(mk2) and mk2 > 0 and on_k2 == 1:
                cands.append((mk2, "k200"))
            if not np.isnan(mkq) and mkq > 0 and on_kq == 1:
                cands.append((mkq, "kq"))
            cands.sort(reverse=True)
            new_hold = cands[0][1] if cands else "kofr"
            if new_hold != prev_hold:
                reason = "월말 모멘텀 선정" if new_hold != "kofr" else "월말 전 종목 추세이탈 -> 현금대피"
            else:
                reason = "월말 점검 (기존 유지)"
            cur_hold = new_hold
        else:
            # 월중 일간 MA 방어 체크
            if cur_hold == "k200" and on_k2 == 0:
                cur_hold = "kofr"
                reason = f"월중 KODEX 200 MA{ma_len} 이탈 -> 즉시 KOFR 대피"
            elif cur_hold == "kq" and on_kq == 0:
                cur_hold = "kofr"
                reason = f"월중 코스닥150 MA{ma_len} 이탈 -> 즉시 KOFR 대피"

        holdings.append(cur_hold)
        reasons.append(reason)
        if i > 0 and cur_hold != prev_hold:
            events.append({
                "date": d.strftime("%Y-%m-%d"),
                "from": ETF_META[prev_hold]["name"],
                "to": ETF_META[cur_hold]["name"],
                "to_ticker": ETF_META[cur_hold]["ticker"],
                "reason": reason,
                "k200_mom6m": round(float(mk2) * 100, 2) if not np.isnan(mk2) else None,
                "kq_mom6m": round(float(mkq) * 100, 2) if not np.isnan(mkq) else None,
            })

    # 일반계좌 세후 수익곡선 계산 (초기자본 1,000만 원)
    init_cap = 10_000_000.0
    fee = STRATEGY_CONFIG["fee_oneway"]
    tax = STRATEGY_CONFIG["tax_rate"]

    eq = np.zeros(n)
    eq[0] = init_cap
    bh_k200 = np.zeros(n)
    bh_k200[0] = init_cap
    bh_kq = np.zeros(n)
    bh_kq[0] = init_cap

    # 국내 1x: 매매차익 비과세, 분배금만 15.4% 과세
    r_k2_at = data["k200"].loc[bt_idx, "total_ret"] - tax * data["k200"].loc[bt_idx, "div_ret"]
    r_kq_at = data["kq"].loc[bt_idx, "total_ret"] - tax * data["kq"].loc[bt_idx, "div_ret"]
    r_kofr_at = kofr_ret.loc[bt_idx] * (1.0 - tax)

    rets = np.zeros(n)
    turns = 0.0
    for i in range(1, n):
        h_prev = holdings[i - 1]
        if h_prev == "k200":
            gross = float(r_k2_at.iloc[i])
        elif h_prev == "kq":
            gross = float(r_kq_at.iloc[i])
        else:
            gross = float(r_kofr_at.iloc[i])

        t = 1.0 if holdings[i] != h_prev else 0.0
        turns += t
        rets[i] = gross - t * fee
        eq[i] = eq[i - 1] * (1.0 + rets[i])
        bh_k200[i] = bh_k200[i - 1] * (1.0 + float(r_k2_at.iloc[i]))
        bh_kq[i] = bh_kq[i - 1] * (1.0 + float(r_kq_at.iloc[i]))

    eq_s = pd.Series(eq, index=bt_idx)
    ret_s = pd.Series(rets, index=bt_idx)
    years = (bt_idx[-1] - bt_idx[0]).days / 365.25
    cagr = (eq_s.iloc[-1] / eq_s.iloc[0]) ** (1.0 / years) - 1.0
    mdd = (eq_s / eq_s.cummax() - 1.0).min()
    vol = ret_s.iloc[1:].std() * np.sqrt(252)
    ex = ret_s.iloc[1:] - r_kofr_at.iloc[1:]
    ex_sharpe = (ex.mean() * 252) / (ex.std() * np.sqrt(252)) if ex.std() > 0 else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0

    # 최신일 스냅샷
    ld = bt_idx[-1]
    latest_hold = holdings[-1]
    prev_day_hold = holdings[-2] if n >= 2 else latest_hold
    today_changed = latest_hold != prev_day_hold

    # 다음 월말 예정일 계산
    month_end_bday = (ld + pd.offsets.BMonthEnd(0))
    if month_end_bday <= ld:
        month_end_bday = ld + pd.offsets.BMonthEnd(1)
    bdays_to_me = int(len(pd.bdate_range(ld, month_end_bday)) - 1)

    snapshot = {
        "ma_len": ma_len,
        "as_of_date": ld.strftime("%Y-%m-%d"),
        "holding_key": latest_hold,
        "holding_name": ETF_META[latest_hold]["name"],
        "holding_ticker": ETF_META[latest_hold]["ticker"],
        "weight_pct": 100,
        "action": "교체/전환 필요" if today_changed else "기존 보유 유지",
        "action_detail": reasons[-1],
        "prev_holding_name": ETF_META[prev_day_hold]["name"],
        "next_rebal_date": month_end_bday.strftime("%Y-%m-%d"),
        "bdays_to_rebal": bdays_to_me,
        "k200": {
            "nav_close": round(float(data["k200"].loc[ld, "nav_close"]), 2),
            "adj_close": round(float(k200_c.loc[ld]), 2),
            "ret_1d_pct": round(float(data["k200"].loc[ld, "total_ret"]) * 100, 2),
            "mom_6m_pct": round(float(m_k200.loc[ld]) * 100, 2),
            "ma_val": round(float(st_k200.loc[ld, "ma"]), 2),
            "disp_pct": round(float(st_k200.loc[ld, "disp"]) * 100, 2),
            "above_streak": int(st_k200.loc[ld, "above_streak"]),
            "below_streak": int(st_k200.loc[ld, "below_streak"]),
            "trend_on": bool(st_k200.loc[ld, "trend_on"] == 1),
            "pass_all": bool((m_k200.loc[ld] > 0) and (st_k200.loc[ld, "trend_on"] == 1)),
        },
        "kq": {
            "nav_close": round(float(data["kq"].loc[ld, "nav_close"]), 2),
            "adj_close": round(float(kq_c.loc[ld]), 2),
            "ret_1d_pct": round(float(data["kq"].loc[ld, "total_ret"]) * 100, 2),
            "mom_6m_pct": round(float(m_kq.loc[ld]) * 100, 2),
            "ma_val": round(float(st_kq.loc[ld, "ma"]), 2),
            "disp_pct": round(float(st_kq.loc[ld, "disp"]) * 100, 2),
            "above_streak": int(st_kq.loc[ld, "above_streak"]),
            "below_streak": int(st_kq.loc[ld, "below_streak"]),
            "trend_on": bool(st_kq.loc[ld, "trend_on"] == 1),
            "pass_all": bool((m_kq.loc[ld] > 0) and (st_kq.loc[ld, "trend_on"] == 1)),
        },
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
            "ann_turnover": round(turns / years, 2),
            "total_switches": len(events),
        },
        "recent_events": events[-15:][::-1],
    }

    curve_df = pd.DataFrame({
        "date": bt_idx.strftime("%Y-%m-%d"),
        "strategy": np.round(eq / 1e4, 1),
        "bh_k200": np.round(bh_k200 / 1e4, 1),
        "bh_kq": np.round(bh_kq / 1e4, 1),
        "dd_pct": np.round((eq_s / eq_s.cummax() - 1.0).values * 100, 2),
    })
    return snapshot, curve_df


def build_html_dashboard(primary: dict, alts: list, curve_df: pd.DataFrame) -> str:
    # 차트 경량화를 위해 주간 샘플링(5영업일 + 마지막 날 포함)
    sampled = curve_df.iloc[::5].copy()
    if sampled.iloc[-1]["date"] != curve_df.iloc[-1]["date"]:
        sampled = pd.concat([sampled, curve_df.iloc[[-1]]], ignore_index=True)

    now_kst = (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M KST")

    h_key = primary["holding_key"]
    if h_key == "k200":
        hero_bg = "linear-gradient(135deg, #0f4c81 0%, #1b6ca8 100%)"
        badge_emoji = "🔵"
        badge_text = "공격 국면 · 대형주 주도"
    elif h_key == "kq":
        hero_bg = "linear-gradient(135deg, #0e6b4e 0%, #179c72 100%)"
        badge_emoji = "🟢"
        badge_text = "공격 국면 · 코스닥 주도"
    else:
        hero_bg = "linear-gradient(135deg, #4b5563 0%, #374151 100%)"
        badge_emoji = "🛡️"
        badge_text = "방어 국면 · 현금성 대피"

    def asset_card(title, ticker, d, is_selected):
        sel_border = "2px solid #2563eb" if is_selected else "1px solid #e5e7eb"
        sel_badge = '<span class="pill pill-blue">현재 보유 100%</span>' if is_selected else '<span class="pill pill-gray">대기 (0%)</span>'
        tr_badge = '<span class="pill pill-green">추세 ON</span>' if d["trend_on"] else '<span class="pill pill-red">추세 OFF</span>'
        mom_cls = "pos" if d["mom_6m_pct"] > 0 else "neg"
        disp_cls = "pos" if d["disp_pct"] >= 0 else "neg"
        ret1_cls = "pos" if d["ret_1d_pct"] >= 0 else "neg"
        streak_txt = f"상회 {d['above_streak']}일 연속" if d["above_streak"] > 0 else f"하회 {d['below_streak']}일 연속"
        return f"""
        <div class="card" style="border: {sel_border};">
          <div class="card-head">
            <div>
              <div class="asset-title">{title} <span class="ticker">{ticker}</span></div>
              <div class="sub">수정기준가 {d['adj_close']:,.1f}원 (기준가 {d['nav_close']:,.1f}원)</div>
            </div>
            <div class="badges">{sel_badge} {tr_badge}</div>
          </div>
          <div class="kpi-grid">
            <div class="kpi">
              <div class="kpi-label">전일 등락</div>
              <div class="kpi-val {ret1_cls}">{d['ret_1d_pct']:+.2f}%</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">6개월(120일) 모멘텀</div>
              <div class="kpi-val {mom_cls}">{d['mom_6m_pct']:+.2f}%</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">MA{primary['ma_len']} 괴리율</div>
              <div class="kpi-val {disp_cls}">{d['disp_pct']:+.2f}%</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">MA{primary['ma_len']} 연속 상태</div>
              <div class="kpi-val" style="font-size:15px;">{streak_txt}</div>
            </div>
          </div>
        </div>
        """

    alt_rows = ""
    for a in [primary] + alts:
        m = a["metrics"]
        tag = " (기본 권장)" if a["ma_len"] == primary["ma_len"] else ""
        alt_rows += f"""
        <tr>
          <td><strong>MA{a['ma_len']} 방어{tag}</strong></td>
          <td><strong>{a['holding_name']} ({a['holding_ticker']})</strong></td>
          <td>{a['action']}</td>
          <td class="pos"><strong>{m['cagr_pct']:+.2f}%</strong></td>
          <td class="neg">{m['mdd_pct']:.1f}%</td>
          <td>{m['ex_sharpe']:.2f}</td>
          <td>연 {m['ann_turnover']:.1f}회</td>
        </tr>
        """

    ev_rows = ""
    for ev in primary["recent_events"]:
        k2m = f"{ev['k200_mom6m']:+.1f}%" if ev["k200_mom6m"] is not None else "-"
        kqm = f"{ev['kq_mom6m']:+.1f}%" if ev["kq_mom6m"] is not None else "-"
        ev_rows += f"""
        <tr>
          <td>{ev['date']}</td>
          <td>{ev['from']} &rarr; <strong>{ev['to']} ({ev['to_ticker']})</strong></td>
          <td>{ev['reason']}</td>
          <td>{k2m} / {kqm}</td>
        </tr>
        """

    m = primary["metrics"]
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>국내 2대 지수 비과세 모멘텀 시그널 (해법 1)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root {{ --bg:#f5f7fb; --card:#ffffff; --text:#111827; --muted:#6b7280; --border:#e5e7eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Malgun Gothic","Apple SD Gothic Neo",sans-serif; background:var(--bg); color:var(--text); line-height:1.5; }}
    .wrap {{ max-width:1080px; margin:0 auto; padding:24px 16px 56px; }}
    .topbar {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:8px; margin-bottom:16px; }}
    .topbar h1 {{ margin:0; font-size:22px; font-weight:800; letter-spacing:-0.4px; }}
    .topbar .meta {{ font-size:13px; color:var(--muted); }}
    .hero {{ border-radius:16px; padding:26px 28px; color:#fff; background:{hero_bg}; box-shadow:0 10px 25px rgba(15,23,42,0.12); margin-bottom:20px; }}
    .hero-top {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }}
    .hero-tag {{ background:rgba(255,255,255,0.18); padding:4px 12px; border-radius:999px; font-size:13px; font-weight:600; }}
    .hero-main {{ font-size:32px; font-weight:800; margin:12px 0 6px; letter-spacing:-0.6px; }}
    .hero-sub {{ font-size:14.5px; opacity:0.92; display:flex; flex-wrap:wrap; gap:18px; margin-top:8px; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin-bottom:20px; }}
    .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:18px 20px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
    .card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:14px; }}
    .asset-title {{ font-size:18px; font-weight:800; }}
    .ticker {{ font-size:13px; font-weight:600; color:var(--muted); background:#f3f4f6; padding:2px 7px; border-radius:6px; margin-left:4px; }}
    .sub {{ font-size:12.5px; color:var(--muted); margin-top:2px; }}
    .badges {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}
    .pill {{ font-size:12px; font-weight:700; padding:3px 9px; border-radius:999px; }}
    .pill-blue {{ background:#dbeafe; color:#1d4ed8; }}
    .pill-green {{ background:#dcfce7; color:#15803d; }}
    .pill-red {{ background:#fee2e2; color:#b91c1c; }}
    .pill-gray {{ background:#f3f4f6; color:#4b5563; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:10px; }}
    .kpi {{ background:#f9fafb; border-radius:10px; padding:10px 12px; }}
    .kpi-label {{ font-size:12px; color:var(--muted); }}
    .kpi-val {{ font-size:18px; font-weight:800; margin-top:2px; }}
    .pos {{ color:#dc2626; }}
    .neg {{ color:#2563eb; }}
    h2 {{ font-size:17px; font-weight:800; margin:0 0 12px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
    th, td {{ padding:10px 10px; border-bottom:1px solid var(--border); text-align:left; }}
    th {{ font-size:12px; color:var(--muted); font-weight:700; background:#f9fafb; }}
    .section {{ margin-bottom:20px; }}
    .chart-box {{ height:340px; position:relative; }}
    .rules {{ font-size:13.5px; color:#374151; margin:0; padding-left:18px; }}
    .rules li {{ margin-bottom:6px; }}
    .foot {{ text-align:center; font-size:12px; color:var(--muted); margin-top:28px; }}
    @media (max-width:640px) {{ .hero-main {{ font-size:24px; }} th, td {{ padding:8px 6px; font-size:12.5px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1>📈 국내 2대 지수 비과세 모멘텀 시그널 (해법 1)</h1>
      <div class="meta">기준 거래일: <strong>{primary['as_of_date']}</strong> (KOFIA 공식 종가 NAV) · 갱신: {now_kst}</div>
    </div>

    <div class="hero">
      <div class="hero-top">
        <span class="hero-tag">{badge_emoji} {badge_text}</span>
        <span class="hero-tag">오늘 액션: {primary['action']}</span>
      </div>
      <div class="hero-main">{primary['holding_name']} ({primary['holding_ticker']}) — 100%</div>
      <div class="hero-sub">
        <span>판정 근거: <strong>{primary['action_detail']}</strong></span>
        <span>다음 월말 정기 리밸런싱: <strong>{primary['next_rebal_date']}</strong> (D-{primary['bdays_to_rebal']}영업일)</span>
        <span>10.4년 일반계좌 세후 CAGR: <strong>{m['cagr_pct']:+.2f}%</strong> (MDD {m['mdd_pct']:.1f}%)</span>
      </div>
    </div>

    <div class="grid-2">
      {asset_card("KODEX 200", "069500", primary["k200"], primary["holding_key"] == "k200")}
      {asset_card("KODEX 코스닥150", "229200", primary["kq"], primary["holding_key"] == "kq")}
    </div>

    <div class="card section">
      <h2>🎯 일간 방어선별(MA100 · MA80 · MA60) 실시간 시그널 & 10.4년 일반계좌 세후 성과</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>모드</th><th>오늘 보유 종목 (100%)</th><th>액션</th><th>세후 CAGR</th><th>MDD</th><th>초과샤프</th><th>연 회전율</th>
            </tr>
          </thead>
          <tbody>{alt_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card section">
      <h2>📊 10.4년 일반계좌 세후 자산 곡선 (초기자본 1,000만 원 · 편도 8bp · 배당/KOFR 15.4% 과세 반영)</h2>
      <div class="chart-box"><canvas id="eqChart"></canvas></div>
    </div>

    <div class="card section">
      <h2>🔄 최근 시그널 전환 이력 (MA{primary['ma_len']} 기준, 최근 15회)</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr><th>전환일</th><th>포지션 변경</th><th>사유</th><th>당시 6M 모멘텀 (200 / 코스닥150)</th></tr>
          </thead>
          <tbody>{ev_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card section">
      <h2>📋 해법 1 실전 운용 규칙 & 일반계좌 세금 포인트</h2>
      <ul class="rules">
        <li><strong>매월 마지막 영업일 종가 (월 1회 선정):</strong> <code>KODEX 200</code>과 <code>KODEX 코스닥150</code> 중 ① 6개월(120영업일) 총수익률 &gt; 0% ② <code>MA{primary['ma_len']}</code> 추세 ON 조건을 모두 통과한 지수 중 모멘텀 1위 종목에 100% 투자합니다. 둘 다 탈락하면 <code>KODEX KOFR금리액티브</code>로 100% 대피합니다.</li>
        <li><strong>월중 일간 방어 (매일 체크):</strong> 보유 중인 지수가 <code>MA{primary['ma_len']}</code>을 3영업일 연속 하회하거나 종가 기준 -1% 이상 이탈하면 월말을 기다리지 않고 즉시 전량 매도해 <code>KOFR</code>로 대피합니다.</li>
        <li><strong>일반계좌 세금 장점:</strong> 두 국내 주식형 1x ETF는 매매차익이 <strong>전액 비과세(0원)</strong>이며 금융소득종합과세(연 2,000만 원) 합산 대상에도 포함되지 않습니다. (분배금과 KOFR 이자에만 15.4% 과세)</li>
      </ul>
    </div>

    <div class="foot">
      데이터 출처: 금융투자협회 전자공시(KOFIA DIS) 공식 수정기준가 · 매일 평일 오전 08:40 KST GitHub Actions 자동 업데이트<br/>
      ※ 본 대시보드는 전략 검증 및 시그널 참고용이며 투자 판단의 최종 책임은 투자자 본인에게 있습니다.
    </div>
  </div>

  <script>
    const labels = {json.dumps(sampled["date"].tolist())};
    const stratData = {json.dumps(sampled["strategy"].tolist())};
    const k200Data = {json.dumps(sampled["bh_k200"].tolist())};
    const kqData = {json.dumps(sampled["bh_kq"].tolist())};

    new Chart(document.getElementById("eqChart"), {{
      type: "line",
      data: {{
        labels,
        datasets: [
          {{ label: "해법 1 (MA{primary['ma_len']} 세후, {m['cagr_pct']:+.1f}%)", data: stratData, borderColor: "#dc2626", borderWidth: 2.3, pointRadius: 0, tension: 0.1 }},
          {{ label: "KODEX 200 단순보유 (비과세)", data: k200Data, borderColor: "#2563eb", borderWidth: 1.5, borderDash: [4,3], pointRadius: 0, tension: 0.1 }},
          {{ label: "KODEX 코스닥150 단순보유 (비과세)", data: kqData, borderColor: "#9ca3af", borderWidth: 1.3, pointRadius: 0, tension: 0.1 }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{
          legend: {{ position: "top" }},
          tooltip: {{ callbacks: {{ label: (ctx) => `${{ctx.dataset.label}}: ${{Number(ctx.parsed.y).toLocaleString()}}만 원` }} }}
        }},
        scales: {{
          y: {{ type: "logarithmic", ticks: {{ callback: (v) => Number(v).toLocaleString() + "만" }} }},
          x: {{ ticks: {{ maxTicksLimit: 11 }} }}
        }}
      }}
    }});
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

    print("[2/3] 해법 1 시그널 계산 중 (MA100 기본 + MA80 / MA60 비교)...")
    primary, curve_df = run_solution1_engine(aligned_data, ma_len=STRATEGY_CONFIG["primary_ma"], mom_days=STRATEGY_CONFIG["mom_days"])
    alts = []
    for ma_alt in STRATEGY_CONFIG["alt_mas"]:
        snap, _ = run_solution1_engine(aligned_data, ma_len=ma_alt, mom_days=STRATEGY_CONFIG["mom_days"])
        alts.append(snap)

    print("[3/3] 대시보드(docs/index.html) 및 JSON(docs/latest_signal.json) 발행 중...")
    out_json = {
        "generated_at_kst": (dt.datetime.utcnow() + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S KST"),
        "primary": primary,
        "alternatives": alts,
    }
    json_path = os.path.join(DOCS_DIR, "latest_signal.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    html = build_html_dashboard(primary, alts, curve_df)
    html_path = os.path.join(DOCS_DIR, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 터미널 요약 리포트 출력
    print("\n" + "=" * 78)
    print(f" [해법 1 실시간 시그널] 기준 거래일: {primary['as_of_date']} (일반계좌 · 1x 비과세)")
    print("=" * 78)
    print(f" ▶ 오늘 보유 종목 (MA100 기본): {primary['holding_name']} ({primary['holding_ticker']}) — 100%")
    print(f" ▶ 오늘 액션: {primary['action']} ({primary['action_detail']})")
    print(f" ▶ 다음 월말 리밸런싱: {primary['next_rebal_date']} (D-{primary['bdays_to_rebal']}영업일)")
    print("-" * 78)
    for lbl, d in [("KODEX 200 (069500)", primary["k200"]), ("KODEX 코스닥150 (229200)", primary["kq"])]:
        st = "추세 ON" if d["trend_on"] else "추세 OFF"
        print(f" - {lbl:25s} | 6M모멘텀: {d['mom_6m_pct']:+6.2f}% | MA100괴리: {d['disp_pct']:+6.2f}% | {st}")
    print("-" * 78)
    for a in [primary] + alts:
        m = a["metrics"]
        print(f" - [MA{a['ma_len']:3d}] 보유: {a['holding_name']} ({a['holding_ticker']}) | 10.4년 세후 CAGR {m['cagr_pct']:+.2f}% / MDD {m['mdd_pct']:.1f}%")
    print("=" * 78)
    print(f"[완료] {html_path} 및 {json_path} 생성 완료.")


if __name__ == "__main__":
    main()
