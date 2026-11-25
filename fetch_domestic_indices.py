import os
import pandas as pd
from fetch_real_data import fetch_freesis_table, fetch_dis_nav, get_session
from fetch_extra_etfs import search_kofia_fund

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    s = get_session()

    # 1. FreeSIS KOSPI market cap & index (STATSCU0100000020BO)
    print("[1/2] Fetching FreeSIS KOSPI market cap & index...")
    kp_raw = fetch_freesis_table(s, "STATSCU0100000020BO", "20151001", "20261125")
    kp = pd.DataFrame({
        "date": pd.to_datetime(kp_raw["TMPV1"], format="%Y%m%d"),
        "kospi_index": pd.to_numeric(kp_raw["TMPV2"], errors="coerce"),
        "kospi_market_cap": pd.to_numeric(kp_raw["TMPV5"], errors="coerce") / 100.0,  # 억원
    }).sort_values("date").drop_duplicates("date")
    kp.to_csv(os.path.join(BASE_DIR, "freesis_kospi_mcap.csv"), index=False)
    print(f"  -> {len(kp)} rows ({kp['date'].min().date()} ~ {kp['date'].max().date()})")

    # 2. Find pre-2016 KRX major sector 1x ETFs (KODEX 반도체, 은행, 자동차, 헬스케어/바이오 등)
    print("[2/2] Searching pre-2016 KRX domestic 1x sector index ETFs...")
    picks = {}
    for kw, key in [("KODEX 반도체", "semi"), ("KODEX 은행", "bank"), ("KODEX 자동차", "auto"), ("KODEX 철강", "steel"), ("KODEX 에너지화학", "chem")]:
        items = search_kofia_fund(s, kw)
        for cd, est, nm in items:
            if est and est <= "20160101" and "상장지수" in nm and "레버리지" not in nm and "인버스" not in nm and "미국" not in nm and "글로벌" not in nm:
                print(f"  [{key}] {cd} | est={est} | {nm}")
                if key not in picks:
                    picks[key] = (cd, nm)

    for key, (cd, nm) in picks.items():
        df = fetch_dis_nav(s, cd, 2015, 2026)
        out = os.path.join(BASE_DIR, f"kodex_{key}_nav.csv")
        df.to_csv(out, index=False)
        print(f"  Saved {out}: {len(df)} rows ({df['date'].min().date()} ~ {df['date'].max().date()})")


if __name__ == "__main__":
    main()
