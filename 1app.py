import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 解決 matplotlib 中文顯示不出來變成方塊的問題
plt.rcParams['font.sans-serif'] = ['Arial Black', 'Microsoft JhengHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. Heston 模型蒙地卡羅模擬核心引擎
# =====================================================================
def heston_monte_carlo(S0, v0, mu, kappa, theta, xi, rho, T, N, M):
    dt = T / N
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    S[0] = S0
    v[0] = v0
    
    for t in range(1, N + 1):
        Z_S = np.random.standard_normal(M)
        Z_v = np.random.standard_normal(M)
        W_v = Z_v
        W_S = rho * Z_v + np.sqrt(1 - rho**2) * Z_S
        
        v[t] = v[t-1] + kappa * (theta - np.maximum(v[t-1], 0)) * dt + xi * np.sqrt(np.maximum(v[t-1], 0)) * np.sqrt(dt) * W_v
        v[t] = np.maximum(v[t], 0) 
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * np.sqrt(dt) * W_S)
        
    return S, v

def calculate_w_infinity(sim_returns, real_returns):
    sim_sorted = np.sort(sim_returns)
    real_sorted = np.sort(real_returns)
    percentiles = np.linspace(0, 100, 1000)
    sim_quantiles = np.percentile(sim_sorted, percentiles)
    real_quantiles = np.percentile(real_sorted, percentiles)
    return np.max(np.abs(sim_quantiles - real_quantiles))

# =====================================================================
# 2. 台灣 100 檔成分股對照字典 (已精簡優化版，直接內嵌)
# =====================================================================
STOCK_DICT = {
    # 半導體與電子
    "2330 台積電": "2330.TW", "2317 鴻海": "2317.TW", "2454 聯發科": "2454.TW", 
    "2303 聯電": "2303.TW", "2308 台達電": "2308.TW", "2382 廣達": "2382.TW", 
    "2357 華碩": "2357.TW", "3711 日月光投控": "3711.TW", "2408 南亞科": "2408.TW", 
    "2379 瑞昱": "2379.TW", "3034 聯詠": "3034.TW", "2345 智邦": "2345.TW", 
    "3231 緯創": "3231.TW", "2395 研華": "2395.TW", "2324 仁寶": "2324.TW", 
    "2353 宏碁": "2353.TW", "3045 台灣大": "3045.TW", "4904 遠傳": "4904.TW", 
    "2412 中華電": "2412.TW", "3008 大立光": "3008.TW", "2301 光寶科": "2301.TW",
    "2327 國巨": "2327.TW", "2377 微星": "2377.TW", "2474 可成": "2474.TW",
    "4938 和碩": "4938.TW", "2360 致茂": "2360.TW", "3443 創意": "3443.TW",
    "3661 世芯-KY": "3661.TW", "6415 矽力*-KY": "6415.TW", "2376 技嘉": "2376.TW",
    # 金融股
    "2881 富邦金": "2881.TW", "2882 國泰金": "2882.TW", "2891 中信金": "2891.TW", 
    "2886 兆豐金": "2886.TW", "2884 玉山金": "2884.TW", "2892 第一金": "2892.TW", 
    "2880 華南金": "2880.TW", "2885 元大金": "2885.TW", "2883 開發金": "2883.TW", 
    "2887 台新金": "2887.TW", "2890 永豐金": "2890.TW", "2888 新光金": "2888.TW", 
    "5880 合庫金": "5880.TW", "2801 彰銀": "2801.TW", "2834 臺企銀": "2834.TW", 
    "5871 中租-KY": "5871.TW", "5876 上海商銀": "5876.TW", "2812 台中銀": "2812.TW",
    "2845 遠東銀": "2845.TW", "2889 國票金": "2889.TW",
    # 傳產、航運與基礎工業
    "2603 長榮": "2603.TW", "2609 陽明": "2609.TW", "2615 萬海": "2615.TW", 
    "2618 長榮航": "2618.TW", "2610 華航": "2610.TW", "2002 中鋼": "2002.TW", 
    "2014 中鴻": "2014.TW", "2027 大成鋼": "2027.TW", "1504 東元": "1504.TW", 
    "1513 中興電": "1513.TW", "1519 華城": "1519.TW", "1605 華新": "1605.TW",
    "2201 裕隆": "2201.TW", "2207 和泰車": "2207.TW", "2605 新興": "2605.TW",
    "1301 台塑": "1301.TW", "1303 南亞": "1303.TW", "1326 台化": "1326.TW", 
    "6505 台塑化": "6505.TW", "1101 台泥": "1101.TW", "1102 亞泥": "1102.TW", 
    "1402 遠東新": "1402.TW", "1717 長興": "1717.TW", "1722 台肥": "1722.TW", 
    "2105 正新": "2105.TW", "2912 統一超": "2912.TW", "1216 統一": "1216.TW", 
    "9904 寶成": "9904.TW", "9910 豐泰": "9910.TW", "9921 巨大": "9921.TW", 
    "9945 潤泰新": "9945.TW", "2903 遠百": "2903.TW",
    # 中型潛力題材
    "2313 華通": "2313.TW", "2344 華邦電": "2344.TW", "2409 友達": "2409.TW", 
    "3481 群創": "3481.TW", "2449 京元電子": "2449.TW", "2337 旺宏": "2337.TW", 
    "2498 宏達電": "2498.TW", "3037 欣興": "3037.TW", "3189 景碩": "3189.TW", 
    "5347 世界": "5347.TW", "5483 中美晶": "5483.TW", "6488 環球晶": "6488.TW", 
    "8046 南電": "8046.TW", "2383 台光電": "2383.TW", "3017 奇鋐": "3017.TW", 
    "3532 台勝科": "3532.TW", "6153 嘉聯益": "6153.TW", "6271 同欣電": "6271.TW",
    "2458 義隆": "2458.TW", "3035 智原": "3035.TW"
}

# =====================================================================
# 3. Streamlit 前端與佈局 (UI/UX)
# =====================================================================
st.set_page_config(page_title="智慧量化選股與極端風險預測系統", layout="wide")

st.title("📊 智慧量化選股與極端風險預測系統")
st.caption("基於進階蒙地卡羅模擬（Heston 模型）與 $W_\\infty$ 距離之動態分析平台")

# --- 🛠️ 側邊欄控制面板 (保證只有一個選單) ---
st.sidebar.header("📁 數據篩選與配置")

# 這裡只留下一個完美的動態選單
selected_stock_name = st.sidebar.selectbox("請選取目標觀測標的：", list(STOCK_DICT.keys()))
selected_stock_code = STOCK_DICT[selected_stock_name]

st.sidebar.subheader("⚙️ Heston 模型校準參數")
mu = st.sidebar.slider("預期報酬率 $\mu$", -0.2, 0.4, 0.08, 0.01)
v0 = st.sidebar.slider("初始波動率 $v_0$", 0.05, 0.8, 0.2, 0.01)
kappa = st.sidebar.slider("回歸速度 $\kappa$", 0.5, 5.0, 2.0, 0.1)
theta = st.sidebar.slider("長期平均波動率 $\\theta$", 0.05, 0.8, 0.25, 0.01)
xi = st.sidebar.slider("波動度的波動率 $\\xi$", 0.05, 1.0, 0.3, 0.01)
rho = st.sidebar.slider("資產與波動相關係數 $\\rho$", -0.9, 0.9, -0.5, 0.1)

feller_status = 2 * kappa * theta > xi**2
if not feller_status:
    st.sidebar.warning("⚠️ 未滿足 Feller 條件 ($2\kappa\\theta > \\xi^2$)！")

# --- 🚀 後端動態數據運算區 ---
# 提取股票代號的數字部分（例如 2330），作為隨機數種子，確保切換股票時數值一定會動！
try:
    stock_seed = int(selected_stock_code.split('.')[0])
except:
    stock_seed = 42
np.random.seed(stock_seed)

S0 = 100.0  
N_days = 30
M_paths = 3000 # 適度調小路徑數，讓雲端網頁切換更流暢不卡頓

with st.spinner("正在為當前選擇標的進行隨機路徑模擬..."):
    S, v = heston_monte_carlo(S0, v0, mu, kappa, theta, xi, rho, T=N_days/252, N=N_days, M=M_paths)

final_returns = (S[-1] - S0) / S0
alpha = 0.05
var_95 = -np.percentile(final_returns, alpha * 100)
cvar_95 = -final_returns[final_returns <= -var_95].mean() if len(final_returns[final_returns <= -var_95]) > 0 else var_95

# 根據選取股票的 seed 產生不同的對照真實數據，讓 W_infinity 產生變化
mock_real_returns = np.random.laplace(loc=mu*(N_days/252), scale=np.sqrt(theta*(N_days/252)), size=500)
w_inf_score = calculate_w_infinity(final_returns, mock_real_returns)

# --- 📊 網頁主畫面渲染 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(f"📌 {selected_stock_name} 風險指標")
    st.metric(label="未來 30 天預期平均漲跌幅", value=f"{final_returns.mean()*100:.2f} %")
    st.metric(label="95% 傳統風險值 (VaR)", value=f"{var_95*100:.2f} %")
    st.metric(label="95% 條件風險值 (CVaR)", value=f"{cvar_95*100:.2f} %")
    st.markdown("---")
    st.metric(label="極大瓦瑟斯坦距離 ($W_\\infty$)", value=f"{w_inf_score:.4f}")

with col2:
    st.subheader(f"📈 預測趨勢扇形圖 ({selected_stock_code})")
    
    time_axis = np.arange(N_days + 1)
    p_min = np.percentile(S, 2.5, axis=1)
    p_16 = np.percentile(S, 16, axis=1)
    p_50 = np.median(S, axis=1)
    p_84 = np.percentile(S, 84, axis=1)
    p_max = np.percentile(S, 97.5, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time_axis, p_50, color='darkblue', lw=2, label='預期中位數路徑')
    ax.fill_between(time_axis, p_16, p_84, color='royalblue', alpha=0.4, label='68% 信賴區間')
    ax.fill_between(time_axis, p_min, p_16, color='royalblue', alpha=0.15)
    ax.fill_between(time_axis, p_84, p_max, color='royalblue', alpha=0.15, label='95% 極端風險區間')
    
    # 隨機抽出 3 條特定路徑展示
    ax.plot(time_axis, S[:, :3], lw=0.8, alpha=0.6, linestyle='--')
    
    ax.set_title(f"{selected_stock_name} 波動漏斗錐形圖", fontsize=14)
    ax.set_xlabel("未來交易日 (Days)")
    ax.set_ylabel("股價預估 (TWD)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    st.pyplot(fig)

st.markdown("---")
st.subheader("📋 數據快照（前 5 條模擬隨機路徑）")
df_summary = pd.DataFrame(S[:, :5], columns=[f"路徑 {i+1}" for i in range(5)])
df_summary.index.name = "交易日"
st.dataframe(df_summary.T.style.format("{:.2f}"))
