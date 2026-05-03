import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objects as go

# ============================================================
# 1. 定数・マスターデータ
# ============================================================
LOCATIONS: dict = {
    "西之表市":         {"lat": 30.730, "lon": 131.002},
    "南さつま市":        {"lat": 31.416, "lon": 130.318},
    "長島町":            {"lat": 32.186, "lon": 130.141},
    "鹿屋市":            {"lat": 31.378, "lon": 130.852},
    "根占町(南大隅町)":  {"lat": 31.196, "lon": 130.767},
    "伊仙町":            {"lat": 27.683, "lon": 128.932},
    "知名町":            {"lat": 27.381, "lon": 128.599},
}

DAILY_VARS = "temperature_2m_mean,precipitation_sum"

# ============================================================
# 2. データ取得
# ============================================================
def _parse_response(j: dict, source: str) -> pd.DataFrame:
    d = j.get("daily", {})
    n = len(d.get("time", []))
    return pd.DataFrame({
        "Date":          pd.to_datetime(d.get("time", [])),
        "Temp_Avg":      d.get("temperature_2m_mean", [np.nan] * n),
        "Precipitation": d.get("precipitation_sum",   [0.0]    * n),
        "Source":        source,
    })

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_weather(
    location_name: str,
    lat: float,
    lon: float,
    start_str: str,
    end_str: str,
) -> pd.DataFrame:
    start  = datetime.date.fromisoformat(start_str)
    end    = datetime.date.fromisoformat(end_str)
    today  = datetime.date.today()
    cutoff = today - datetime.timedelta(days=5)
    frames = []

    def _get(url: str, params: dict) -> dict | None:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            st.warning(f"タイムアウト: {url.split('/')[2]}")
        except requests.exceptions.HTTPError as e:
            st.warning(f"HTTP エラー {e.response.status_code}: {url.split('/')[2]}\n{e.response.text[:200]}")
        except Exception as e:
            st.warning(f"取得エラー: {e}")
        return None

    if start < cutoff:
        hist_end = min(end, cutoff - datetime.timedelta(days=1))
        j = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start.isoformat(),
            "end_date":   hist_end.isoformat(),
            "daily":      DAILY_VARS,
            "timezone":   "Asia/Tokyo",
        })
        if j:
            frames.append(_parse_response(j, "historical"))

    if end >= cutoff:
        fore_start = max(start, cutoff)
        max_forecast = today + datetime.timedelta(days=14)
        fore_end = min(end, max_forecast)
        
        j = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": fore_start.isoformat(),
            "end_date":   fore_end.isoformat(),
            "daily":      DAILY_VARS,
            "timezone":   "Asia/Tokyo",
        })
        if j:
            frames.append(_parse_response(j, "forecast"))

    if not frames:
        return pd.DataFrame()

    df = (
        pd.concat(frames)
          .drop_duplicates(subset="Date")
          .sort_values("Date")
          .reset_index(drop=True)
    )

    df["Temp_Avg"]      = df["Temp_Avg"].interpolate(method="linear").bfill().ffill()
    df["Precipitation"] = df["Precipitation"].fillna(0.0)
    return df

# ============================================================
# 3. FLABS 計算
# ============================================================
def calculate_flabs(
    df: pd.DataFrame,
    emergence_date: datetime.date,
    temp_min: float,
    temp_max: float,
    precip_thresh: float,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df["Date"] >= pd.to_datetime(emergence_date)].copy()
    df["Days_From_Emergence"] = (df["Date"] - pd.to_datetime(emergence_date)).dt.days

    temp_ok  = (df["Temp_Avg"] >= temp_min) & (df["Temp_Avg"] <= temp_max)
    prec_ok  = df["Precipitation"] > precip_thresh
    df["Favorable"]   = np.where(temp_ok & prec_ok, 1, 0)
    df["FLABS_Index"] = df["Favorable"].cumsum()
    return df

# ============================================================
# 4. グラフ生成
# ============================================================
def build_realtime_chart(df: pd.DataFrame, threshold: float, spray_lead: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Precipitation"], name="降水量 (mm)", yaxis="y1", marker_color="steelblue", opacity=0.35,
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Temp_Avg"], name="平均気温 (℃)", yaxis="y1", line=dict(color="orange", width=1.8, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["FLABS_Index"], name="FLABS 指数", yaxis="y2", line=dict(color="crimson", width=3),
    ))
    
    favorable_days = df[df["Favorable"] == 1]
    fig.add_trace(go.Scatter(
        x=favorable_days["Date"], y=[0] * len(favorable_days), mode="markers", name="感染好適日", yaxis="y1", marker=dict(color="crimson", symbol="triangle-up", size=8),
    ))

    forecast_rows = df[df["Source"] == "forecast"]
    if not forecast_rows.empty:
        boundary = forecast_rows["Date"].min()
        fig.add_vline(x=boundary.timestamp() * 1000, line_dash="dot", line_color="gray", annotation_text="← 実績 | 予報 →", annotation_position="top")

    spray_trigger = max(0.0, threshold - spray_lead)
    win_rows   = df[df["FLABS_Index"] >= spray_trigger]
    alert_rows = df[df["FLABS_Index"] >= threshold]
    if not win_rows.empty:
        win_start = win_rows.iloc[0]["Date"]
        win_end   = alert_rows.iloc[0]["Date"] if not alert_rows.empty else df["Date"].max()
        fig.add_vrect(x0=win_start, x1=win_end, fillcolor="rgba(255, 165, 0, 0.18)", line_width=0, annotation_text="🌿 散布適期", annotation_position="top left")

    fig.add_hline(y=threshold, line_dash="dash", line_color="crimson", annotation_text=f"防除閾値 ({threshold})", annotation_position="right", yref="y2")
    y2_max = max(threshold + 3, df["FLABS_Index"].max() + 2)
    fig.update_layout(
        yaxis=dict(title="気温(℃) / 降水量(mm)"),
        yaxis2=dict(title="FLABS 指数", overlaying="y", side="right", range=[0, y2_max]),
        hovermode="x unified", height=540, margin=dict(t=40, b=60), legend=dict(orientation="h", y=-0.2),
    )
    return fig

def build_comparison_chart(year_results: dict, threshold: float, current_year: int) -> go.Figure:
    fig = go.Figure()
    for year, df in sorted(year_results.items()):
        is_current = (year == current_year)
        fig.add_trace(go.Scatter(
            x=df["Days_From_Emergence"], y=df["FLABS_Index"], mode="lines", name=f"{year}年{'（今年・途中）' if is_current else ''}", line=dict(width=4 if is_current else 2),
        ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="crimson", annotation_text=f"防除閾値 ({threshold})")
    fig.update_layout(
        xaxis_title="出芽日からの経過日数", yaxis_title="FLABS 指数", hovermode="x unified", height=560, margin=dict(t=40, b=60), legend=dict(orientation="h", y=-0.2),
    )
    return fig

# ============================================================
# 5. Streamlit UI
# ============================================================
st.set_page_config(page_title="ジャガイモ疫病発生予察システム", layout="wide")
st.title("🥔 ジャガイモ疫病発生予察システム（FLABSモデル）")

# ---- サイドバー ----
with st.sidebar:
    st.header("⚙️ 共通設定")
    location_name = st.selectbox("産地", list(LOCATIONS.keys()))
    lat = LOCATIONS[location_name]["lat"]
    lon = LOCATIONS[location_name]["lon"]

    st.divider()
    st.subheader("防除パラメータ")
    threshold  = st.number_input("防除閾値（感染好適日数）", value=7, min_value=1, max_value=30)
    spray_lead = st.number_input("散布リード日数", value=2, min_value=0, max_value=10, help="閾値到達 N 日前から散布適期としてハイライトします")

    st.divider()
    st.subheader("気象閾値")
    temp_min      = st.slider("気温 下限 (℃)",  5.0, 15.0, 10.0, 0.5)
    temp_max      = st.slider("気温 上限 (℃)", 20.0, 35.0, 25.0, 0.5)
    precip_thresh = st.slider("降水量 閾値 (mm)", 0.0,  5.0,  1.0, 0.5)

# ---- タブ構成 ----
tab_about, tab1, tab2 = st.tabs(["📖 FLABS概要", "📡 リアルタイム予察", "📊 年次比較解析"])

# ========== Tab_about: FLABS概要（1枚紙） ==========
with tab_about:
    st.header("ジャガイモ疫病とFLABS（長崎モデル）について")
    st.markdown("本システムは、気象データからジャガイモ疫病（*Phytophthora infestans*）の感染リスクを定量化し、**最適な予防防除のタイミング**を意思決定するための支援ツールです。")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.error("#### 🦠 疫病の恐ろしさと「予防」の鉄則\n"
                 "ジャガイモ疫病は非常に進行が早く、ひとたび発病すると数日で圃場全体に蔓延し、著しい減収や塊茎腐敗（疫病イモ）をもたらします。\n\n"
                 "病斑が見えてから治療剤を散布しても手遅れになることが多く、**「発病する前に保護殺菌剤で植物体をコーティングしておく（予防散布）」**ことが最大の防除対策です。")
        
    with col_b:
        st.info("#### 🌤️ FLABS指数のメカニズム\n"
                "疫病菌は特定の気象条件が揃うと爆発的に増殖します。本モデルでは、**出芽日（50%出芽）**を起点とし、以下の条件を同時に満たした日を「**感染好適日**」としてカウントします。\n\n"
                "* 🌡️ **温度条件**：平均気温が `10℃ 〜 25℃`\n"
                "* 💧 **水分条件**：降水量が `1mm` 以上\n\n"
                "これらの好適日が何日あったかを積算したものが「**FLABS指数**」です。")

    st.markdown("---")
    
    st.subheader("💡 本アプリの活用方法（アラートの読み方）")
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        st.metric(label="1. 出芽日の設定", value="起点の設定")
        st.markdown("圃場の芽が約50%出揃った日をサイドバー、またはタブ内のカレンダーから入力します。ここから指数の積算がスタートします。")
    with col_d:
        st.metric(label="2. アラートの確認", value="閾値への到達予測")
        st.markdown("FLABS指数が設定した「**防除閾値**（デフォルト: 7）」に達する日が予測されると、画面上に赤いアラートが表示されます。")
    with col_e:
        st.metric(label="3. 散布適期ウィンドウ", value="事前の予防散布")
        st.markdown("グラフ上のオレンジ色の帯（**散布適期**）は、閾値に達する数日前のリード期間を示しています。この帯の期間内に保護殺菌剤の散布を完了させてください。")

    st.caption("※ 本ツールは意思決定を支援する計算モデルです。実際の防除にあたっては、圃場の微気象や生育状況を必ず目視で確認し、地域の防除基準に従ってください。")

# ========== Tab1: リアルタイム ==========
with tab1:
    st.subheader("リアルタイム疫病リスク予測")
    c1, c2 = st.columns(2)
    with c1:
        emergence1 = st.date_input("出芽日（分析開始日）", datetime.date(datetime.date.today().year, 2, 15), key="t1_em")
    with c2:
        end_date1 = st.date_input("分析終了日", datetime.date.today() + datetime.timedelta(days=7), key="t1_end")

    if st.button("分析実行", key="btn_tab1", type="primary"):
        if end_date1 <= emergence1:
            st.error("終了日は出芽日より後に設定してください。")
        else:
            with st.spinner("気象データ取得・統合中…"):
                raw1 = fetch_weather(location_name, lat, lon, emergence1.isoformat(), end_date1.isoformat())
            if raw1.empty:
                st.error("データを取得できませんでした。ネットワーク接続等を確認してください。")
            else:
                res1 = calculate_flabs(raw1, emergence1, temp_min, temp_max, precip_thresh)
                spray_trigger = max(0.0, threshold - spray_lead)
                win_rows   = res1[res1["FLABS_Index"] >= spray_trigger]
                alert_rows = res1[res1["FLABS_Index"] >= threshold]

                if not win_rows.empty:
                    sw = win_rows.iloc[0]["Date"].strftime("%Y-%m-%d")
                    al = alert_rows.iloc[0]["Date"].strftime("%Y-%m-%d") if not alert_rows.empty else "期間外"
                    st.warning(f"🌿 **散布適期**: {sw} ～ {al}")

                if not alert_rows.empty:
                    st.error(f"⚠️ **防除アラート**: {alert_rows.iloc[0]['Date'].strftime('%Y-%m-%d')} に FLABS 閾値（{threshold}）到達予測")
                else:
                    st.success(f"✅ 期間内（{end_date1}まで）に閾値 {threshold} への到達なし")

                st.plotly_chart(build_realtime_chart(res1, threshold, spray_lead), use_container_width=True)

                with st.expander("📋 日別データ"):
                    disp_cols = ["Date", "Temp_Avg", "Precipitation", "Favorable", "FLABS_Index", "Source"]
                    st.dataframe(res1[disp_cols], use_container_width=True)

                csv1 = res1[disp_cols].to_csv(index=False, encoding="utf-8-sig")
                st.download_button("📥 CSV ダウンロード", data=csv1, file_name=f"flabs_{location_name}_{emergence1}.csv", mime="text/csv")

# ========== Tab2: 年次比較 ==========
with tab2:
    st.subheader("年次間 疫病リスク比較")
    st.caption("同じ出芽日設定で複数年の FLABS 指数を比較します。今年のデータは取得可能な範囲（予報データ限界まで）で重ねて表示します。")

    ca, cb, cc = st.columns(3)
    with ca:
        base_month = st.number_input("出芽 月", 1, 12, 2)
        base_day   = st.number_input("出芽 日", 1, 31, 15)
    with cb:
        cult_days = st.number_input("評価日数", 30, 150, 90, step=10)
    with cc:
        current_year = datetime.date.today().year
        avail_years  = list(range(current_year, current_year - 6, -1))
        target_years = st.multiselect("比較する年（今年は途中経過として追加可）", avail_years, default=[current_year, current_year - 1, current_year - 2, current_year - 3])

    if st.button("比較解析実行", key="btn_tab2", type="primary"):
        if not target_years:
            st.warning("年を 1 つ以上選択してください。")
        else:
            year_results = {}
            summary_rows = []
            progress_bar = st.progress(0, text="データ取得中…")

            for i, year in enumerate(sorted(target_years)):
                progress_bar.progress(i / len(target_years), text=f"{year}年 取得中…")
                try:
                    e_date = datetime.date(year, base_month, base_day)
                    end_d  = e_date + datetime.timedelta(days=cult_days)

                    if year == current_year:
                        end_d = min(end_d, datetime.date.today() + datetime.timedelta(days=14))

                    raw = fetch_weather(location_name, lat, lon, e_date.isoformat(), end_d.isoformat())
                    res = calculate_flabs(raw, e_date, temp_min, temp_max, precip_thresh)
                    
                    if not res.empty:
                        year_results[year] = res
                        hit     = res[res["FLABS_Index"] >= threshold]["Days_From_Emergence"]
                        days_to = int(hit.min()) if not hit.empty else None
                        summary_rows.append({
                            "年":             f"{year}年{'（今年）' if year == current_year else ''}",
                            "閾値到達日数":  f"{days_to} 日目" if days_to is not None else "到達せず",
                            "最終FLABS指数": res["FLABS_Index"].max(),
                            "感染好適日数":  int(res["Favorable"].sum()),
                        })
                except ValueError:
                    st.error(f"{year}年: {base_month}/{base_day} は無効な日付です。")
                except Exception as e:
                    st.error(f"{year}年 処理エラー: {e}")

            progress_bar.progress(1.0, text="完了")

            if year_results:
                st.plotly_chart(build_comparison_chart(year_results, threshold, current_year), use_container_width=True)
                st.subheader("解析サマリー")
                summary_df = pd.DataFrame(summary_rows).set_index("年")
                st.dataframe(summary_df, use_container_width=True)

                all_df = pd.concat([df.assign(Year=yr) for yr, df in year_results.items()])
                st.download_button("📥 全年データ CSV", data=all_df.to_csv(index=False, encoding="utf-8-sig"), file_name=f"flabs_comparison_{location_name}.csv", mime="text/csv")