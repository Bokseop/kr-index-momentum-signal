import os
import ssl
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from fetch_real_data import fetch_dis_nav, get_session

urllib3.disable_warnings()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def search_kofia_fund(s: requests.Session, keyword: str):
    url = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/"
    xh = {"Content-Type": "text/xml; charset=utf-8", "User-Agent": "Mozilla/5.0", "Referer": "https://dis.kofia.or.kr/"}
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<message><proframeHeader>"
        "<pfmAppName>FS-DIS2</pfmAppName>"
        "<pfmSvcName>DISComFundSrchSO</pfmSvcName>"
        "<pfmFnName>search</pfmFnName>"
        "</proframeHeader><systemHeader></systemHeader>"
        "<DISComFundSrchListDTO>"
        f"<uFundNm>{keyword}</uFundNm>"
        "<manageCompCd></manageCompCd>"
        "<manageStt>2</manageStt>"
        "<uTotCnt>1</uTotCnt>"
        "</DISComFundSrchListDTO></message>"
    )
    r = s.post(url, data=xml.encode("utf-8"), headers=xh, timeout=15)
    root = ET.fromstring(r.text)
    out = []
    for it in root.findall(".//list"):
        d = {c.tag: c.text for c in it}
        out.append((d.get("standardCd", ""), d.get("trustEstDt", ""), d.get("koreanFundNm", "")))
    return out


def main():
    s = get_session()

    # 1. Find exact codes for TIGER 미국나스닥100 (133690), KODEX 200 (069500), TIGER 미국S&P500선물(H) or Gold
    for kw in ["미국나스닥100", "KODEX 200", "골드선물"]:
        items = search_kofia_fund(s, kw)
        print(f"=== Search '{kw}' ({len(items)} hits) ===")
        for cd, est, nm in items:
            if est and est <= "20160601" and ("상장지수" in nm):
                print(f"  {cd} | est={est} | {nm}")

    # 2. Fetch KODEX 코스닥150레버리지 (K55105B91912) & 인버스 (K55105BF4255)
    targets = {
        "kodex_kosdaq150_lev2x_nav.csv": ("K55105B91912", 2015, 2026),
        "kodex_kosdaq150_inv1x_nav.csv": ("K55105BF4255", 2016, 2026),
    }
    for fname, (cd, sy, ey) in targets.items():
        print(f"Fetching {fname} ({cd})...")
        df = fetch_dis_nav(s, cd, sy, ey)
        out = os.path.join(BASE_DIR, fname)
        df.to_csv(out, index=False)
        print(f"  -> {len(df)} rows ({df['date'].min().date()} ~ {df['date'].max().date()})")


if __name__ == "__main__":
    main()
