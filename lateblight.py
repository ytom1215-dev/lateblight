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
    "長崎県(諫早)":      {"lat": 32.844, "lon": 130.046},
    "西之表市":         {"lat": 30.730, "lon": 131.002},
    "南さつま市":        {"lat": 31.416, "lon": 130.318},
    "長島町":            {"lat": 32.186, "lon": 130.141},
    "鹿屋市":            {"lat": 31.378, "lon": 130.852},
    "根占町(南大隅町)":  {"lat": 31.196, "lon": 130.767},
    "伊仙町":            {"lat": 27.683, "lon": 128.932},
    "知名町":            {"lat": 27.381, "lon": 128.599},
}

DAILY_VARS = "temperature_2m_mean,temperature_2m_min,precipitation_sum"

# ============================================================
# 2. データ取得・前処理
# ============================================================
def _parse_response(j: dict, source: str) -> pd.DataFrame:
    d = j.get("daily", {})
    n = len(d.get("time", []))
    return pd.DataFrame({
        "Date":          pd.to_datetime(d.get("time", [])),
        "Temp_Avg":      d.get("temperature_2m_mean", [np.nan] * n),
        "Temp_Min":      d.get("temperature_2m_min", [np.nan] * n),
        "Precipitation": d.get("precipitation_sum",   [0.0]    * n),
        "Source":        source,
    })

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_weather(location_name: str, lat: float, lon: float, start_str: str, end_str: str) -> pd.DataFrame:
    requested_start = datetime.date.fromisoformat(start_str)
    fetch_start = requested_start - datetime.timedelta(days=10)
    end = datetime.date.fromisoformat(end_str)
    
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=5)
    frames = []

    def _get(url: str, params: dict) -> dict | None:
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.warning(f"取得エラー: {e}")
            return None

    if fetch_start < cutoff:
        hist_end = min(end, cutoff - datetime.timedelta(days=1))
        j = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon,
            "start_date": fetch_start.isoformat(), "end_date": hist_end.isoformat(),
            "daily": DAILY_VARS, "timezone": "Asia/Tokyo",
        })
        if j: frames.append(_parse_response(j, "historical"))

    if end >= cutoff:
        fore_start = max(fetch_start, cutoff)
        max_forecast = today + datetime.timedelta(days=14)
        fore_end = min(end, max_forecast)
        j = _get("https://api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon,
            "start_date": fore_start.isoformat(), "end_date": fore_end.isoformat(),
            "daily": DAILY_VARS, "timezone": "Asia/Tokyo",
        })
        if j: frames.append(_parse_response(j, "forecast"))

    if not frames: return pd.DataFrame()

    df = pd.concat(frames).drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)
    df["Temp_Avg"] = df["Temp_Avg"].interpolate().bfill().ffill()
    df["Temp_Min"] = df["Temp_Min"].interpolate().bfill().ffill()
    df["Precipitation"] = df["Precipitation"].fillna(0.0)

    df["Precip_5d"] = df["Precipitation"].rolling(window=5).sum()
    df["Precip_10d"] = df["Precipitation"].rolling(window=10).sum()
    
    return df[df["Date"] >= pd.to_datetime(requested_start)].copy()

# ============================================================
# 3. FLABS 改変後ロジック計算 (長崎モデル)
# ============================================================
def calculate_flabs_nagasaki(df: pd.DataFrame, emergence_date: datetime.date) -> pd.DataFrame:
    if df.empty: return df
    
    df = df.sort_values("Date").copy()
    df["Days_From_Emergence"] = (df["Date"] - pd.to_datetime(emergence_date)).dt.days
    
    daily_indices = []
    cumulative_indices = []
    current_cum = 0

    for i, row in df.iterrows():
        avg_t = row["Temp_Avg"]
        min_t = row["Temp_Min"]
        prec  = row["Precipitation"]
        p5    = row["Precip_5d"] if not np.isnan(row["Precip_5d"]) else 0
        p10   = row["Precip_10d"] if not np.isnan(row["Precip_10d"]) else 0
        
        idx = 0

        # ① 基本判定
        if avg_t <= 26.5 and min_t > 7.2:
            if 5.5 <= p5 <= 10.0:
                if 11.7 <= avg_t <= 26.5: idx = 1
            elif 10.5 <= p5 <= 20.0:
                if 15.1 <= avg_t <= 26.5: idx = 2
                elif 11.7 <= avg_t <= 15.0: idx = 1
            elif p5 >= 20.5:
                if 15.1 <= avg_t <= 26.5: idx = 2
                elif 11.7 <= avg_t <= 15.0: idx = 2
                elif 7.2 <= avg_t <= 11.6:  idx = 1

        # ② 降雨補正
        if idx == 0 and prec >= 0.5 and avg_t >= 7.2:
            idx = 1

        # ③ 低温多雨補正
        if min_t <= 7.2 and p5 >= 30.0:
            if 7.2 <= avg_t <= 11.6: idx = 1
            elif avg_t >= 11.7:      idx = 2

        # ⑤ 高温無効化
        if avg_t >= 26.6:
            idx = 0

        current_cum += idx

        # ④ 乾燥リセット
        if current_cum <= 5 and p10 == 0:
            current_cum = 0
        
        # ⑤ 高温リセット
        if avg_t >= 26.6:
            current_cum = 0

        daily_indices.append(idx)
        cumulative_indices.append(current_cum)

    df["Daily_Index"] = daily_indices
    df["FLABS_Index"] = cumulative_indices
    df["Favorable"] = np.where(df["Daily_Index"] > 0, 1, 0)
    
    return df

# ============================================================
# 4. グラフ生成
# ============================================================
def build_realtime_chart(df: pd.DataFrame, threshold: float, spray_lead: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Precipitation"], name="降水量 (mm)", yaxis="y1", marker_color="steelblue", opacity=0.3))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Temp_Avg"], name="平均気温 (℃)", yaxis="y1", line=dict(color="orange", dash="dot")))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["FLABS_Index"], name="累積 FLABS 指数", yaxis="y2", line=dict(color="crimson", width=3)))
    
    favorable_days = df[df["Favorable"] == 1]
    fig.add_trace(go.Scatter(x=favorable_days["Date"], y=[0] * len(favorable_days), mode="markers", name="感染好適日", marker=dict(color="crimson", symbol="triangle-up", size=10)))

    if "forecast" in df["Source"].values:
        boundary = df[df["Source"] == "forecast"]["Date"].min()
        fig.add_vline(x=boundary.timestamp() * 1000, line_dash="dot", annotation_text="予報エリア")

    spray_trigger = max(0.0, threshold - spray_lead)
    win_rows   = df[df["FLABS_Index"] >= spray_trigger]
    alert_rows = df[df["FLABS_Index"] >= threshold]
    
    if not win_rows.empty:
        win_start = win_rows.iloc[0]["Date"]
        win_end   = alert_rows.iloc[0]["Date"] if not alert_rows.empty else df["Date"].max()
        fig.add_vrect(x0=win_start, x1=win_end, fillcolor="rgba(255, 165, 0, 0.18)", line_width=0, annotation_text="🌿 散布適期", annotation_position="top left")

    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="防除閾値", yref="y2")
    
    fig.update_layout(
        yaxis=dict(title="気温(℃) / 降水量(mm)"),
        yaxis2=dict(title="累積 FLABS 指数", overlaying="y", side="right", range=[0, max(threshold + 5, df["FLABS_Index"].max() + 2)]),
        hovermode="x unified", height=500, legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=40)
    )
    return fig

def build_comparison_chart(year_results: dict, threshold: float, current_year: int) -> go.Figure:
    fig = go.Figure()
    for year, df in sorted(year_results.items()):
        is_current = (year == current_year)
        fig.add_trace(go.Scatter(x=df["Days_From_Emergence"], y=df["FLABS_Index"], name=f"{year}年作", line=dict(width=4 if is_current else 2)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="防除閾値")
    fig.update_layout(xaxis_title="出芽後日数", yaxis_title="累積 FLABS 指数", hovermode="x unified", height=500)
    return fig

# ============================================================
# 5. Streamlit UI 本体
# ============================================================
st.set_page_config(page_title="長崎モデル・ジャガイモ疫病予察", layout="wide")
st.title("🥔 ジャガイモ疫病予察システム（長崎FLABS改変モデル）")

with st.sidebar:
    st.header("⚙️ 設定")
    location_name = st.selectbox("産地", list(LOCATIONS.keys()))
    lat, lon = LOCATIONS[location_name]["lat"], LOCATIONS[location_name]["lon"]
    
    st.divider()
    threshold = st.number_input("防除閾値（累積指数）", value=21, help="長崎県では通常21を基準月日とします")
    spray_lead = st.number_input("散布リード期間", value=5, help="閾値到達の何日前から『適期』とするか")

tab_about, tab1, tab2 = st.tabs(["📖 モデル解説とシミュレーター", "📡 リアルタイム予察", "📊 年次比較"])

# ========== Tab: モデル解説とシミュレーター ==========
with tab_about:
    st.header("🥔 ジャガイモ疫病アラートの仕組み")
    st.markdown("""
    ジャガイモ疫病は、**「涼しい気温」**と**「多湿（雨）」**の条件が重なると一気に感染が広がります。
    このシステムは、毎日の気象データから「疫病リスクポイント（感染好適指数）」を計算し、**累積ポイントが「21」に達したら「防除（農薬散布）のサイン」を出す**という仕組みです。
    """)
    
    st.divider()
    
    st.subheader("🎮 疫病リスク判定シミュレーター")
    st.markdown("スライダーを動かして、「その日のポイント」がどう判定されるか試してみましょう。")
    
    # シミュレーターのUI
    with st.container(border=True):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sim_avg_temp = st.slider("今日の平均気温 (℃)", min_value=0.0, max_value=35.0, value=20.0, step=0.1)
            sim_today_prec = st.slider("今日の降水量 (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
        with col_s2:
            sim_min_temp = st.slider("今日の最低気温 (℃)", min_value=-5.0, max_value=30.0, value=10.0, step=0.1)
            sim_p5_prec = st.slider("過去5日間の降水量 (mm)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)

        # シミュレーターのロジック計算
        sim_idx = 0
        sim_msg = "リスクなし (安全)"
        
        # ① 基本判定
        if sim_avg_temp <= 26.5 and sim_min_temp > 7.2:
            if 5.5 <= sim_p5_prec <= 10.0:
                if 11.7 <= sim_avg_temp <= 26.5: 
                    sim_idx = 1; sim_msg = "① 基本判定: ややリスクあり (+1点)"
            elif 10.5 <= sim_p5_prec <= 20.0:
                if 15.1 <= sim_avg_temp <= 26.5: 
                    sim_idx = 2; sim_msg = "① 基本判定: 高リスク (+2点)"
                elif 11.7 <= sim_avg_temp <= 15.0: 
                    sim_idx = 1; sim_msg = "① 基本判定: ややリスクあり (+1点)"
            elif sim_p5_prec >= 20.5:
                if 15.1 <= sim_avg_temp <= 26.5: 
                    sim_idx = 2; sim_msg = "① 基本判定: 高リスク (+2点)"
                elif 11.7 <= sim_avg_temp <= 15.0: 
                    sim_idx = 2; sim_msg = "① 基本判定: 高リスク (+2点)"
                elif 7.2 <= sim_avg_temp <= 11.6:  
                    sim_idx = 1; sim_msg = "① 基本判定: ややリスクあり (+1点)"
        
        # ② 降雨補正
        if sim_idx == 0 and sim_today_prec >= 0.5 and sim_avg_temp >= 7.2:
            sim_idx = 1; sim_msg = "② 降雨補正: 今日の雨によりリスク発生 (+1点)"
            
        # ③ 低温多雨補正
        if sim_min_temp <= 7.2 and sim_p5_prec >= 30.0:
            if 7.2 <= sim_avg_temp <= 11.6: 
                sim_idx = 1; sim_msg = "③ 低温多雨補正: ややリスクあり (+1点)"
            elif sim_avg_temp >= 11.7:      
                sim_idx = 2; sim_msg = "③ 低温多雨補正: 高リスク (+2点)"
                
        # ⑤ 高温無効化
        if sim_avg_temp >= 26.6:
            sim_idx = 0; sim_msg = "⑤ 高温無効化: 暑すぎるためリスクなし (※累積もリセットされます)"

        # 結果表示
        st.markdown(f"#### 今日の指数: **{sim_idx} 点**")
        if sim_idx == 2:
            st.error(f"判定理由: {sim_msg}")
        elif sim_idx == 1:
            st.warning(f"判定理由: {sim_msg}")
        else:
            st.success(f"判定理由: {sim_msg}")

    st.divider()

    st.subheader("📊 ポイント計算のルール詳細")
    st.markdown("""
    #### 基本の計算ルール（毎日チェック）
    **「今日の平均気温」**と**「過去5日間の降水量」**の組み合わせで基本ポイントが決まります。
    ※最低気温が7.2℃より高いことが条件です。
    """)
    
    st.markdown("""
    | 今日の平均気温 | 前5日間の雨が少ない<br>(〜5.0mm) | 前5日間の雨が少し<br>(5.5〜10.0mm) | 前5日間の雨が多い<br>(10.5〜20.0mm) | 前5日間の雨がとても多い<br>(20.5mm〜) |
    | :--- | :---: | :---: | :---: | :---: |
    | **15.1 〜 26.5℃** | 0点 | 1点 | **2点** | **2点** |
    | **11.7 〜 15.0℃** | 0点 | 1点 | 1点 | **2点** |
    | **7.2 〜 11.6℃** | 0点 | 0点 | 0点 | 1点 |
    """)

    st.markdown("""
    #### 🌧️ 特別ルール（ポイントの追加・リセット）
    基本ルールに当てはまらなくても、特定の天候になるとポイントが変動（補正）されます。長崎の気候に合わせた重要なポイントです。

    * **① 今日の雨補正（ポイントUP）**
      基本ポイントが「0点」でも、今日0.5mm以上の雨が降り、平均気温が7.2℃以上あれば**「1点」**になります。（少しでも雨が降るとリスク発生）
    * **② 冷え込み・大雨補正（ポイントUP）**
      最低気温が7.2℃以下まで冷え込んでも、過去5日間に合計30mm以上の大雨が降っている場合、平均気温に応じてポイントが入ります。（平均11.7℃以上なら**2点**、7.2〜11.6℃なら**1点**）
    * **③ 猛暑リセット（ポイント0化）**
      平均気温が**26.6℃以上**の暑い日は、疫病菌が活動できなくなります。その日のポイントは「0点」になり、さらに**これまで貯まっていた累積ポイントもすべて「0」にリセット**されます。
    * **④ 乾燥リセット（ポイント0化）**
      まだ累積ポイントが「5点以下」の初期段階で、**過去10日間まったく雨が降らなかった場合**、空気が乾燥してリスクが消滅するため、累積ポイントが「0」にリセットされます。
    """)

# ========== Tab: リアルタイム予察 ==========
with tab1:
    col1, col2 = st.columns(2)
    with col1: em_date = st.date_input("出芽日", datetime.date(datetime.date.today().year, 2, 15))
    with col2: ed_date = st.date_input("予測終了日", datetime.date.today() + datetime.timedelta(days=10))

    if st.button("分析実行", type="primary", key="btn_realtime"):
        with st.spinner("データ取得・計算中..."):
            df = fetch_weather(location_name, lat, lon, em_date.isoformat(), ed_date.isoformat())
            if not df.empty:
                res = calculate_flabs_nagasaki(df, em_date)
                
                alert_df = res[res["FLABS_Index"] >= threshold]
                if not alert_df.empty:
                    target_date = alert_df.iloc[0]["Date"].strftime('%Y-%m-%d')
                    st.error(f"⚠️ **防除アラート**: {target_date} に累積指数 {threshold} 到達予測。直ちに防除を検討してください。")
                else:
                    st.success(f"✅ 予測期間内に閾値 {threshold} への到達はありません。")
                
                st.plotly_chart(build_realtime_chart(res, threshold, spray_lead), use_container_width=True)
                
                with st.expander("📊 計算データ詳細"):
                    st.dataframe(res[["Date", "Temp_Avg", "Temp_Min", "Precipitation", "Precip_5d", "Daily_Index", "FLABS_Index"]], use_container_width=True)

# ========== Tab: 年次比較 ==========
with tab2:
    st.subheader("過去数年との比較・リスク判定")
    
    c1, c2 = st.columns(2)
    with c1:
        base_em_date = st.date_input("出芽日（比較の起点）", datetime.date(datetime.date.today().year, 2, 15), format="YYYY/MM/DD")
    with c2:
        base_hv_date = st.date_input("収穫日（比較の終点）", datetime.date(datetime.date.today().year, 5, 31), format="YYYY/MM/DD")

    target_years = st.multiselect("比較対象年", range(2026, 2019, -1), default=[2026, 2025, 2024], help="収穫日（終点）が属する作年を指定します。")
    
    if st.button("比較実行", key="btn_compare"):
        with st.spinner("過去データの取得・計算中..."):
            year_results = {}
            period_labels = {} 
            
            # 入力された「出芽日」と「収穫日」の年の差分を計算
            year_diff = base_hv_date.year - base_em_date.year

            for yr in target_years:
                e_year = yr
                s_year = yr - year_diff
                
                # 安全対策：入力ミスで終了日の方が過去になってしまう場合
                if e_year < s_year or (e_year == s_year and (base_hv_date.month, base_hv_date.day) < (base_em_date.month, base_em_date.day)):
                    s_year = yr
                    e_year = yr + 1

                s = datetime.date(s_year, base_em_date.month, base_em_date.day)
                e = datetime.date(e_year, base_hv_date.month, base_hv_date.day)
                
                max_forecast = datetime.date.today() + datetime.timedelta(days=14)
                
                if s > max_forecast:
                    st.warning(f"【{yr}年作】出芽日（{s.strftime('%Y/%m/%d')}）が予報範囲外（未来）のためスキップしました。")
                    continue
                
                actual_e = e if e <= max_forecast else max_forecast

                raw = fetch_weather(location_name, lat, lon, s.isoformat(), actual_e.isoformat())
                if not raw.empty:
                    year_results[yr] = calculate_flabs_nagasaki(raw, s)
                    period_labels[yr] = f"{s.strftime('%Y/%m/%d')} 〜 {actual_e.strftime('%Y/%m/%d')}"
            
            if year_results:
                st.plotly_chart(build_comparison_chart(year_results, threshold, datetime.date.today().year), use_container_width=True)

                st.subheader("🗓️ 年次別 疫病リスク判定サマリー")
                st.markdown("※ 出芽日から防除閾値に到達するまでの早さや、感染好適日数の多さから、その年の疫病リスクを判定します。")
                
                summary_data = []

                for yr in sorted(year_results.keys(), reverse=True):
                    df_yr = year_results[yr]
                    hit_rows = df_yr[df_yr["FLABS_Index"] >= threshold]
                    
                    total_favorable_days = df_yr["Favorable"].sum()
                    max_flabs = df_yr["FLABS_Index"].max()
                    
                    if not hit_rows.empty:
                        days_to_threshold = hit_rows.iloc[0]["Days_From_Emergence"]
                        if days_to_threshold <= 40:
                            risk_eval = "🔴 激発年 (極高リスク)"
                        elif days_to_threshold <= 60:
                            risk_eval = "🟠 多発生年 (高リスク)"
                        else:
                            risk_eval = "🟡 警戒年 (中リスク)"
                        days_text = f"出芽後 {days_to_threshold} 日"
                        
                    else:
                        actual_end_date = df_yr["Date"].max().date()
                        if actual_end_date >= datetime.date.today() - datetime.timedelta(days=2):
                            risk_eval = "⏳ 判定中 (未到達)"
                            days_text = "未到達"
                        else:
                            risk_eval = "🔵 少発生年 (低リスク)"
                            days_text = "到達せず"
                    
                    summary_data.append({
                        "対象年": f"{yr}年作",
                        "計算対象期間": period_labels[yr],
                        "リスク判定": risk_eval,
                        "閾値到達タイミング": days_text,
                        "感染好適日数": f"{total_favorable_days} 日",
                        "期間内最大指数": max_flabs
                    })

                summary_df = pd.DataFrame(summary_data)
                
                st.dataframe(
                    summary_df,
                    column_config={
                        "期間内最大指数": st.column_config.ProgressColumn(
                            "期間内最大指数",
                            format="%d",
                            min_value=0,
                            max_value=int(summary_df["期間内最大指数"].max() + 5),
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
