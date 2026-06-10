import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 設定網頁標題與寬度
st.set_page_config(page_title="台股 100 大持股 AI 預測大盤", layout="wide")

# 100檔股票清單設定 (已剔除重複的宏碁)
MY_STOCKS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2881.TW", "2882.TW", "2382.TW", "2891.TW", "3711.TW", "2412.TW",
    "2303.TW", "2886.TW", "3008.TW", "2884.TW", "1216.TW", "2892.TW", "5880.TW", "2357.TW", "3231.TW", "2880.TW",
    "2395.TW", "3034.TW", "2603.TW", "2885.TW", "2912.TW", "2379.TW", "3045.TW", "4904.TW", "2002.TW", "5871.TW",
    "2324.TW", "2356.TW", "2890.TW", "6669.TW", "1101.TW", "2301.TW", "2409.TW", "3481.TW", "2327.TW", "2883.TW",
    "3017.TW", "2618.TW", "2610.TW", "6505.TW", "1301.TW", "1303.TW", "1326.TW", "2105.TW", "2207.TW", "9904.TW",
    "1402.TW", "2801.TW", "2834.TW", "5876.TW", "9945.TW", "2345.TW", "3037.TW", "2383.TW", "2059.TW", "2368.TW",
    "3653.TW", "3443.TW", "3665.TW", "4958.TW", "8046.TW", "2408.TW", "2344.TW", "2449.TW", "5274.TWO", "3529.TWO",
    "6515.TW", "2360.TW", "7769.TW", "6223.TWO", "2609.TW", "2615.TW", "2377.TW", "2353.TW", "4966.TWO", "6415.TW",
    "8454.TW", "9921.TW", "1504.TW", "1513.TW", "1519.TW", "1605.TW", "1722.TW", "2206.TW", "2313.TW", "2371.TW",
    "2606.TW", "2887.TW", "3532.TW", "3702.TW", "5347.TWO", "6239.TW", "8996.TW", "9917.TW", "9941.TW"
]

MY_STOCKS_NAME = [
    "台積電", "鴻海", "聯發科", "台達電", "富邦金", "國泰金", "廣達", "中信金", "日月光投控", "中華電",
    "聯電", "兆豐金", "大立光", "玉山金", "統一", "第一金", "合庫金", "華碩", "緯創", "華南金",
    "研華", "聯詠", "長榮", "元大金", "統一超", "瑞昱", "台灣大", "遠傳", "中鋼", "中租-KY",
    "仁寶", "英業達", "永豐金", "緯穎", "台泥", "光寶科", "友達", "群創", "國巨", "凱基金",
    "奇鋐", "長榮航", "華航", "台塑化", "台塑", "南亞", "台化", "正新", "和泰車", "寶成",
    "遠東新", "彰銀", "臺企銀", "上海商銀", "潤泰新", "智邦", "欣興", "台光電", "川湖", "金像電",
    "健策", "創意", "貿聯-KY", "臻鼎-KY", "南電", "南亞科", "華邦電", "京元電子", "信驊", "力旺",
    "穎崴", "致茂", "鴻勁", "旺矽", "陽明", "萬海", "微星", "宏碁", "譜瑞-KY", "矽力*-KY",
    "富邦媒", "巨大", "東元", "中興電", "華城", "華新", "台肥", "三陽工業", "華通", "大同",
    "裕民", "台新金", "台勝科", "大聯大", "世界先進", "力成", "高力", "中保科", "裕融"
]

# ==========================================
# 🔄 幾何布朗運動 (GBM) 蒙地卡羅核心邏輯
# ==========================================
def calculate_gbm_parameters(df):
    """計算 GBM 所需的年化報酬率與波動度"""
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    days = len(df)
    if days < 2:
        return 0.0, 0.0, float(df['Close'].iloc[-1]) if not df.empty else 0.0
    
    mu = (df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (round(252/days, 1)) - 1
    Std = returns.std() * np.sqrt(252)
    S0 = float(df['Close'].iloc[-1])
    return mu, Std, S0

def run_monte_carlo_table(df, simulations=10000):
    """專門給大盤總表使用的快速模擬 (1萬次即足夠精準收斂平均值)"""
    mu, Std, S0 = calculate_gbm_parameters(df)
    if S0 == 0.0:
        return [0.0] * 10

    time_steps = {'7': 5/252.0, '15': 10/252.0, '30': 21/252.0, '180': 126/252.0, '365': 252/252.0}
    results = []
    
    for dt in time_steps.values():
        Z = np.random.standard_normal(simulations)
        ST = S0 * np.exp((mu - 0.5 * Std**2) * dt + Std * np.sqrt(dt) * Z)
        expected_price = float(np.mean(ST))
        growth_rate = ((expected_price - S0) / S0) * 100
        results.append(round(expected_price, 2))
        results.append(round(growth_rate, 2))
    return [round(S0, 2)] + results

# 💡 加上快取，避免重複下載
@st.cache_data(show_spinner="正在加載 100 大持股數據並運行大盤預測...")
def calculate_all_trends():
    rows = []
    for idx, name in zip(MY_STOCKS, MY_STOCKS_NAME):
        try:
            pure_number_id = idx.split('.')[0]
            df = yf.download(idx, period="10y", progress=False)
            if df.empty:
                rows.append([pure_number_id, name] + [0.0]*11)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            gbm_results = run_monte_carlo_table(df)
            rows.append([pure_number_id, name] + gbm_results)
        except:
            rows.append([pure_number_id, name] + [0.0]*11)
            
    columns = [
        "股票代碼", "股票名稱", "當前現價",
        "一周預測價", "一周(%)", "半月預測價", "半月(%)", 
        "一月預測價", "一月(%)", "半年預測價", "半年(%)", "一年預測價", "一年(%)"
    ]
    return pd.DataFrame(rows, columns=columns)

# ==========================================
# 🎨 Streamlit 網頁前端介面
# ==========================================
st.title("📈 台灣 100 大持股 — AI 多時段預測與個股模擬儀表板")
st.markdown("本系統採用**幾何布朗運動 (GBM) 蒙地卡羅模型**進行資產未來價格演化預測。")

# ------------------------------------------
# 🔍 區塊一：任意股票搜尋與未來路徑圖表 (新功能)
# ------------------------------------------
st.subheader("🔍 個股動態搜尋與未來走勢模擬路徑圖")
search_ticker = st.text_input("請輸入任意股票代號（台股請加 .TW 或 .TWO，美股直接輸入代號如 AAPL）：", value="2330.TW")

if search_ticker:
    with st.spinner(f"正在擷取 {search_ticker} 資料並進行 100,000 次高精細路徑模擬..."):
        try:
            stock_df = yf.download(search_ticker, period="5y", progress=False)
            if stock_df.empty:
                st.error("找不到該股票數據，請檢查代號是否正確！")
            else:
                if isinstance(stock_df.columns, pd.MultiIndex):
                    stock_df.columns = stock_df.columns.get_level_values(0)
                
                # 計算參數
                mu, Std, S0 = calculate_gbm_parameters(stock_df)
                
                # 模擬未來 252 個交易日 (1年) 的走勢路徑做圖
                future_days = 252
                sim_paths = 50 # 畫面上繪製 50 条隨機可能走勢路徑
                dt = 1 / 252.0
                
                # 矩陣型蒙地卡羅路徑生成
                gbm_paths = np.zeros((future_days + 1, sim_paths))
                gbm_paths[0] = S0
                for t in range(1, future_days + 1):
                    Z = np.random.standard_normal(sim_paths)
                    gbm_paths[t] = gbm_paths[t-1] * np.exp((mu - 0.5 * Std**2) * dt + Std * np.sqrt(dt) * Z)
                
                # 計算特定時間點的期望價格與報酬率 (高精細度 50 萬次)
                Z_future = np.random.standard_normal(500000)
                time_horizons = {'未來一週': 5/252.0, '未來一個月': 21/252.0, '未來一年': 252/252.0}
                
                # 顯示數據指標卡
                cols = st.columns(4)
                cols[0].metric("當前現價", f"${S0:.2f}")
                
                for idx, (label, t_target) in enumerate(time_horizons.items(), start=1):
                    ST_target = S0 * np.exp((mu - 0.5 * Std**2) * t_target + Std * np.sqrt(t_target) * Z_future)
                    exp_p = float(np.mean(ST_target))
                    exp_r = ((exp_p - S0) / S0) * 100
                    cols[idx].metric(f"{label}預測", f"${exp_p:.2f}", f"{exp_r:+.2f}%")
                
                # 繪製圖表：歷史股價 + 未來模擬路徑
                st.markdown(f"##### 📊 {search_ticker} 歷史股價與未來 1 年蒙地卡羅模擬路徑走勢")
                
                # 整理歷史資料
                hist_series = stock_df['Close'].tail(252).values # 拿過去一年的歷史當背景
                total_len = len(hist_series) + future_days
                
                chart_data = pd.DataFrame()
                chart_data['歷史股價'] = np.append(hist_series, [None] * future_days)
                
                # 放入 50 條模擬線
                for i in range(sim_paths):
                    chart_data[f'模擬路徑 {i+1}'] = np.append([None] * (len(hist_series) - 1), gbm_paths[:, i])
                
                st.line_chart(chart_data, height=400)
                st.caption("註：灰色/彩色虛線代表模型隨機模擬出的 50 種未來可能走勢路徑，核心指標卡為 50 萬次模擬之統計期望值。")
                
        except Exception as e:
            st.error(f"預測執行失敗: {e}")

st.markdown("---")

# ------------------------------------------
# 📊 區塊二：100 大持股大盤看板
# ------------------------------------------
st.subheader("📊 台灣 100 大持股 AI 預測大盤大看板")
st.markdown("表格依據台灣股市習慣：**紅色代表預期上漲，綠色代表預期下跌**。")

master_df = calculate_all_trends()

# 設定欄位上色邏輯 (只針對帶有 % 符號的漲跌幅欄位上色)
def style_dataframe(df_slice):
    pct_cols = [c for c in master_df.columns if "(%)" in c or c in ["一周(%)", "半月(%)", "一月(%)", "半年(%)", "一年(%)"]]
    return df_slice.style.map(
        lambda val: 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if isinstance(val, (int, float)) and val > 0
               else ('background-color: #d4edda; color: #155724; font-weight: bold;' if isinstance(val, (int, float)) and val < 0 else ''),
        subset=pct_cols
    ).format({c: "{:.2f}" for c in master_df.columns[2:]}) # 格式化所有數字欄位留到小數後兩位

# 呈現 Streamlit 表格
st.dataframe(
    style_dataframe(master_df),
    use_container_width=True,
    height=550
)

st.success("✨ 數據加載完成！您可以點擊大盤總表任意欄位（例如：點擊『一年(%)』）進行全台股大排行排序。")
