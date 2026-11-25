import os
import ssl
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter

urllib3.disable_warnings()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def get_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", LegacySSLAdapter())
    return s


def fetch_freesis_table(s: requests.Session, obj_nm: str, start_ymd: str, end_ymd: str) -> pd.DataFrame:
    url = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"
    headers = {
        "Content-Type": "application/json;charset=UTF-8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://freesis.kofia.or.kr/",
    }
    payload = {
        "dmSearch": {
            "OBJ_NM": obj_nm,
            "tmpV1": "D",
            "tmpV40": "1000000",
            "tmpV41": "1",
            "tmpV45": start_ymd,
            "tmpV46": end_ymd,
        }
    }
    r = s.post(url, data=json.dumps(payload), headers=headers, timeout=30)
    r.raise_for_status()
    ds = r.json().get("ds1", [])
    return pd.DataFrame(ds)


def fetch_dis_nav(s: requests.Session, standard_cd: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch official daily NAV from KOFIA DIS (max 3 years per query -> use 2-year chunks)."""
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
        st = f"{y}0101"
        ed = f"{y_end}1231"
        xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<message><proframeHeader>"
            "<pfmAppName>FS-DIS2</pfmAppName>"
            "<pfmSvcName>DISFundStdPrcStutSO</pfmSvcName>"
            "<pfmFnName>select</pfmFnName>"
            "</proframeHeader><systemHeader></systemHeader>"
            "<DISCondFuncDTO>"
            f"<tmpV30>{st}</tmpV30>"
            f"<tmpV31>{ed}</tmpV31>"
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
                    "aum_m": float(d["tmpV5"]) if d.get("tmpV5") not in (None, "", "-") else np.nan,
                    "kospi": float(d["tmpV6"]) if d.get("tmpV6") not in (None, "", "-") else np.nan,
                    "ktb3y": float(d["tmpV7"]) if d.get("tmpV7") not in (None, "", "-") else np.nan,
                    "cd91": float(d["tmpV10"]) if d.get("tmpV10") not in (None, "", "-") else np.nan,
                })
        y = y_end + 1
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def main():
    s = get_session()

    print("[1/4] Fetching FreeSIS KOSDAQ credit balance (STATSCU0100000070BO)...")
    cr_raw = fetch_freesis_table(s, "STATSCU0100000070BO", "20151001", "20261125")
    cr = pd.DataFrame({
        "date": pd.to_datetime(cr_raw["TMPV1"], format="%Y%m%d"),
        "credit_total_eok": pd.to_numeric(cr_raw["TMPV2"], errors="coerce") / 100.0,   # 백만원 -> 억원
        "credit_kospi_eok": pd.to_numeric(cr_raw["TMPV3"], errors="coerce") / 100.0,   # 백만원 -> 억원
        "credit_balance": pd.to_numeric(cr_raw["TMPV4"], errors="coerce") / 100.0,     # 코스닥 신용융자 (억원)
    }).sort_values("date").drop_duplicates("date")
    print(f"   -> {len(cr)} rows ({cr['date'].min().date()} ~ {cr['date'].max().date()})")

    print("[2/4] Fetching FreeSIS KOSDAQ market cap & index (STATSCU0100000030BO)...")
    mc_raw = fetch_freesis_table(s, "STATSCU0100000030BO", "20151001", "20261125")
    mc = pd.DataFrame({
        "date": pd.to_datetime(mc_raw["TMPV1"], format="%Y%m%d"),
        "kosdaq_index": pd.to_numeric(mc_raw["TMPV2"], errors="coerce"),
        "market_cap": pd.to_numeric(mc_raw["TMPV5"], errors="coerce") / 100.0,         # 백만원 -> 억원
        "foreign_ratio": pd.to_numeric(mc_raw["TMPV7"], errors="coerce"),
    }).sort_values("date").drop_duplicates("date")
    print(f"   -> {len(mc)} rows ({mc['date'].min().date()} ~ {mc['date'].max().date()})")

    # Merge FreeSIS credit & market cap -> freesis_credit.csv
    freesis = pd.merge(mc, cr[["date", "credit_balance", "credit_kospi_eok", "credit_total_eok"]], on="date", how="outer").sort_values("date")
    freesis["credit_balance"] = freesis["credit_balance"].ffill()
    freesis["market_cap"] = freesis["market_cap"].ffill()
    freesis_out = os.path.join(BASE_DIR, "freesis_credit.csv")
    freesis.to_csv(freesis_out, index=False)
    print(f"   Saved {freesis_out} ({len(freesis)} rows)")

    print("[3/4] Fetching KOFIA DIS daily NAV: KODEX 코스닥150 (K55105B74603)...")
    etf = fetch_dis_nav(s, "K55105B74603", 2015, 2026)
    etf_out = os.path.join(BASE_DIR, "kodex_kosdaq150_nav.csv")
    etf.to_csv(etf_out, index=False)
    print(f"   -> {len(etf)} rows ({etf['date'].min().date()} ~ {etf['date'].max().date()})")

    print("[4/4] Fetching KOFIA DIS daily NAV: KODEX KOFR금리액티브 (K55105DR9373)...")
    kofr = fetch_dis_nav(s, "K55105DR9373", 2022, 2026)
    kofr_out = os.path.join(BASE_DIR, "kodex_kofr_nav.csv")
    kofr.to_csv(kofr_out, index=False)
    print(f"   -> {len(kofr)} rows ({kofr['date'].min().date()} ~ {kofr['date'].max().date()})")

    # --- CRITICAL DATA INTEGRITY & DATE-ALIGNMENT VERIFICATION ---
    print("\n================ DATA INTEGRITY CHECKS ================")
    m = pd.merge(etf[["date", "nav", "diff", "kospi", "cd91"]], mc[["date", "kosdaq_index", "market_cap"]], on="date", how="inner")
    m["nav_ret"] = m["nav"].pct_change()
    m["kq_ret"] = m["kosdaq_index"].pct_change()

    print("Correlation of KODEX 코스닥150 NAV return vs KOSDAQ Index return:")
    for lag in [-2, -1, 0, 1, 2]:
        corr = m["nav_ret"].corr(m["kq_ret"].shift(lag))
        print(f"  corr(nav_ret[t], kosdaq_ret[t - ({lag:+d})]) = {corr:+.6f}")

    # Check if KOFIA DIS NAV date t reflects market close of date t-1 (lag=+1) or date t (lag=0)!
    # Also check KOSPI column in DIS vs FreeSIS KOSPI (STATSCU0100000020BO)
    kp_raw = fetch_freesis_table(s, "STATSCU0100000020BO", "20250101", "20250120")
    kp = pd.DataFrame({
        "date": pd.to_datetime(kp_raw["TMPV1"], format="%Y%m%d"),
        "kospi_freesis": pd.to_numeric(kp_raw["TMPV2"], errors="coerce"),
    }).sort_values("date")
    chk = pd.merge(etf[["date", "nav", "diff", "kospi"]], kp, on="date", how="inner")
    print("\nSample alignment check (2025-01):")
    print(chk.to_string(index=False))

    # Check distribution/dividend events where |nav.diff() - diff| > 5
    m["raw_diff"] = m["nav"].diff()
    m["gap"] = (m["raw_diff"] - m["diff"]).abs()
    big_gaps = m[m["gap"] > 5][["date", "nav", "raw_diff", "diff", "gap"]]
    print(f"\nRows where |nav.diff() - KOFIA_diff| > 5 won: {len(big_gaps)}")
    if len(big_gaps) > 0:
        print(big_gaps.head(20).to_string(index=False))

    # Check credit ratio summary
    f2 = freesis.dropna(subset=["credit_balance", "market_cap"]).copy()
    f2["cr_ratio"] = f2["credit_balance"] / f2["market_cap"] * 100.0
    print("\nReal FreeSIS KOSDAQ Credit Ratio (%) summary (2016~2026):")
    f2_16 = f2[f2["date"] >= "2016-01-01"]
    print(f2_16["cr_ratio"].describe().to_string())
    print(f"Days with cr_ratio >= 2.0%: {(f2_16['cr_ratio'] >= 2.0).sum()} / {len(f2_16)} ({(f2_16['cr_ratio'] >= 2.0).mean()*100:.1f}%)")
    print(f"Days with cr_ratio >= 2.5%: {(f2_16['cr_ratio'] >= 2.5).sum()} / {len(f2_16)} ({(f2_16['cr_ratio'] >= 2.5).mean()*100:.1f}%)")
    print(f"Days with cr_ratio <= 1.8%: {(f2_16['cr_ratio'] <= 1.8).sum()} / {len(f2_16)} ({(f2_16['cr_ratio'] <= 1.8).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
