import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# =====================================================================
# 1. 數學模型核心：Heston 模型蒙地卡羅模擬
# =====================================================================
def heston_monte_carlo(S0, v0, mu, kappa, theta, xi, rho, T, N, M):
    """
    S0: 初始股價, v0: 初始波動率
    mu: 預期報酬率, kappa: 均值回歸速度, theta: 長期平均波動率, xi: 波動率的波動度
    rho: 兩個布朗運動的相關係數
    T: 預測時間(年), N: 時間步數, M: 模擬路徑數量
    """
    dt = T / N
    
    # 初始化矩陣
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    S[0] = S0
    v[0] = v0
    
    # 生成相關聯的布朗運動
    for t in range(1, N + 1):
        # 獨立的標準正態隨機變數
        Z_S = np.random.standard_normal(M)
        Z_v = np.random.standard_normal(M)
        
        # 透過 Cholesky 分解使兩者產生相關性 rho
        W_v = Z_v
        W_S = rho * Z_v + np.sqrt(1 - rho**2) * Z_S
        
        # 波動率路徑更新 (確保加上 Feller 條件防呆，防 v 變成負數)
        v[t] = v[t-1] + kappa * (theta - np.maximum(v[t-1], 0)) * dt + xi * np.sqrt(np.maximum(v[t-1], 0)) * np.sqrt(dt) * W_v
        v[t] = np.maximum(v[t], 0) # 截斷法防負值
        
        # 股價路徑更新
        S[t] = S[t-1] * np.exp((mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1]) * np.sqrt(dt) * W_S)
        
    return S, v

# =====================================================================
# 2. 隨機優化距離：極大瓦瑟斯坦距離 (W_infinity) 計算
# =====================================================================
def calculate_w_infinity(sim_returns, real_returns):
    """
    使用離散分位數對齊法，近似計算模擬收益率與歷史收益率之間的最大 Wasserstein 距離
    """
    # 將兩組數據進行排序（代表經驗分佈函數的逆函數）
    sim_sorted = np.sort(sim_returns)
    real_sorted = np.sort(real_returns)
    
    # 由於長度可能不同，將兩者內插到相同的百分位數點上
    percentiles = np.linspace(0, 100, 1000)
    sim_quantiles = np.percentile(sim_sorted, percentiles)
    real_quantiles = np.percentile(real_sorted, percentiles)
    
    # W_infinity 即為這些分位數點之間的最大絕對偏差
    w_inf = np.max(np.abs(sim_quantiles - real_quantiles))
    return w_inf

# =====================================================================
# 3. Streamlit 網頁 UI 設計
# =====================================================================
st.set_page_config(page_title="智慧量化選股與極端風險預測系統", layout="wide")

st.title("📊 智慧量化選股與極端風險預測系統")
st.caption("基於進階蒙地卡羅模擬（Heston 模型）與 $W_\\infty$ 距離之動態分析平台")

# --- 側邊欄：股票選擇與參數設定 ---
st.sidebar.header("📁 數據篩選與配置")

# 模擬 100 檔股票清單
stock_list = [f"股票 {str(i).zfill(3)}.TW" for i in range(1, 101)]
selected_stock = st.sidebar.selectbox("請選取目標觀測標的：", stock_list)

st.sidebar.subheader("⚙️ Heston 模型校準參數")
mu = st.sidebar.slider("預期報酬率 $\mu$", -0.2, 0.4, 0.08, 0.01)
v0 = st.sidebar.slider("初始波動率 $v_0$", 0.05, 0.8, 0.2, 0.01)
kappa = st.sidebar.slider("回歸速度 $\kappa$", 0.5, 5.0, 2.0, 0.1)
theta = st.sidebar.slider("長期平均波動率 $\\theta$", 0.05, 0.8, 0.25, 0.01)
xi = st.sidebar.slider("波動度的波動率 $\\xi$", 0.05, 1.0, 0.3, 0.01)
rho = st.sidebar.slider("資產與波動相關係數 $\\rho$", -0.9, 0.9, -0.5, 0.1)

# 檢查 Feller 條件並給予警告提示
feller_status = 2 * kappa * theta > xi**2
if not feller_status:
    st.sidebar.warning("⚠️ 未滿足 Feller 條件 ($2\kappa\\theta > \\xi^2$)！隨機波動率路徑可能會頻繁歸零，請微調參數。")

# --- 主要內容區 ---
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📌 標的極端風險指標")
    
    # 執行蒙地卡羅模擬 (設定步數 30 天，模擬 5000 條路徑)
    S0 = 100.0  # 假設目前初始股價為 100
    N_days = 30
    M_paths = 5000
    
    with st.spinner("正在進行萬條隨機路徑模擬與校準..."):
        S, v = heston_monte_carlo(S0, v0, mu, kappa, theta, xi, rho, T=N_days/252, N=N_days, M=M_paths)
    
    # 計算期末收益率
    final_returns = (S[-1] - S0) / S0
    
    # 計算 VaR 與 CVaR (信賴水準 95%)
    alpha = 0.05
    var_95 = -np.percentile(final_returns, alpha * 100)
    cvar_95 = -final_returns[final_returns <= -var_95].mean() if len(final_returns[final_returns <= -var_95]) > 0 else var_95
    
    # 模擬一組歷史真實數據（用於計算 W_infinity）
    np.random.seed(42)
    mock_real_returns = np.random.laplace(loc=mu*(N_days/252), scale=np.sqrt(theta*(N_days/252)), size=500)
    w_inf_score = calculate_w_infinity(final_returns, mock_real_returns)
    
    # 使用 st.metric 呈現
    st.metric(label="未來 30 天預期平均漲跌幅", value=f"{final_returns.mean()*100:.2f} %")
    st.metric(label="95% 傳統風險值 (VaR)", value=f"{var_95*100:.2f} %")
    st.metric(label="95% 條件風險值 (CVaR)", value=f"{cvar_95*100:.2f} %")
    
    st.markdown("---")
    st.metric(label="極大瓦瑟斯坦距離 ($W_\\infty$)", value=f"{w_inf_score:.4f}", help="衡量模擬分佈與真實極端尾端偏差。數值越低代表模型對黑天鵝事件的抓取越精準。")

with col2:
    st.subheader(f"📈 {selected_stock} 未來 30 天預測趨勢扇形圖 (Fan Chart)")
    
    # 計算各分位數路徑用於繪製 Fan Chart
    time_axis = np.arange(N_days + 1)
    p_min = np.percentile(S, 2.5, axis=1)
    p_16 = np.percentile(S, 16, axis=1)
    p_50 = np.median(S, axis=1)
    p_84 = np.percentile(S, 84, axis=1)
    p_max = np.percentile(S, 97.5, axis=1)
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time_axis, p_50, color='darkblue', lw=2, label='預期中位數路徑')
    
    # 填滿 68% 信賴區間
    ax.fill_between(time_axis, p_16, p_84, color='royalblue', alpha=0.4, label='68% 信賴區間 (常態波動)')
    # 填滿 95% 信賴區間
    ax.fill_between(time_axis, p_min, p_16, color='royalblue', alpha=0.15)
    ax.fill_between(time_axis, p_84, p_max, color='royalblue', alpha=0.15, label='95% 信賴區間 (極端風險)')
    
    # 隨機挑選 3 條路徑畫出來增加動態感
    ax.plot(time_axis, S[:, :3], lw=0.8, alpha=0.6, linestyle='--')
    
    ax.set_title(f"{selected_stock} 蒙地卡羅波動漏斗錐形圖", fontsize=12)
    ax.set_xlabel("未來交易日 (Days)")
    ax.set_ylabel("股價預估 (TWD)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    st.pyplot(fig)

st.markdown("---")
st.subheader("📋 模擬資產路徑原始數據（前 5 條路徑摘要）")
df_summary = pd.DataFrame(S[:, :5], columns=[f"路徑 {i+1}" for i in range(5)])
df_summary.index.name = "交易日"
st.dataframe(df_summary.T.style.format("{:.2f}"))
