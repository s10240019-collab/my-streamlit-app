import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
import os
import matplotlib.font_manager as fm

# =====================================================================
# 🌟 修正版：精準讀取 NotoSansCJKtc-Black.otf 字型檔
# =====================================================================
# 直接對齊你在 GitHub 上真實上傳的檔案名稱
font_path = "NotoSansCJKtc-Black.otf"

if os.path.exists(font_path):
    try:
        # 強制註冊你的實體字型檔
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        
        # 全局套用字型
        plt.rcParams['font.family'] = font_name
        plt.rcParams['font.sans-serif'] = [font_name]
        st.sidebar.success("✅ 成功載入自訂中文字型檔！")
    except Exception as e:
        # 如果因為副檔名多重導致底層解析失敗的終極防呆
        plt.rcParams['font.family'] = 'sans-serif'
        st.sidebar.warning(f"⚠️ 字型解析出錯，已啟動系統防呆：{e}")
else:
    # 萬一檔案沒傳好，至少不要讓網頁崩潰
    plt.rcParams['font.family'] = 'sans-serif'
    st.sidebar.warning("⚠️ 找不到 NotoSansCJKtc-Black.otf 檔案，請檢查 GitHub 根目錄。")

# 固定負號顯示，避免負號也變方塊
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 數學模型核心：Heston 模擬、參數校準與組合優化
# =====================================================================

def heston_monte_carlo(S0, v0, mu, kappa, theta, xi, rho, T, N, M):
    """ Heston 模型隨機路徑生成引擎 """
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
    """ 計算模擬與真實歷史收益率的 Max-Wasserstein 距離 """
    sim_sorted = np.sort(sim_returns)
    real_sorted = np.sort(real_returns)
    percentiles = np.linspace(0, 100, 1000)
    sim_quantiles = np.percentile(sim_sorted, percentiles)
    real_quantiles = np.percentile(real_sorted, percentiles)
    return np.max(np.abs(sim_quantiles - real_quantiles))

def calibrate_heston_moments(returns):
    """ 基於歷史數據的動差估計法自動校準參數 """
    mu_hat = returns.mean() * 252       
    var_hat = returns.var() * 252       
    
    kappa_cal = 2.0                     
    theta_cal = max(var_hat, 0.04)      
    v0_cal = max(returns[-20:].var() * 252, 0.04) 
    xi_cal = max(np.sqrt(2 * kappa_cal * theta_cal) * 0.5, 0.1) 
    rho_cal = -0.4                      
    
    return mu_hat, v0_cal, kappa_cal, theta_cal, xi_cal, rho_cal

def optimize_mean_cvar(mean_returns, historical_returns, target_return=0.10):
    """ Mean-CVaR 智慧投資組合風險最小化優化器 """
    num_assets = len(mean_returns)
    
    def portfolio_cvar(weights):
        port_returns = np.dot(historical_returns, weights)
        alpha = 0.05
        var = -np.percentile(port_returns, alpha * 100)
        cvar = -port_returns[port_returns <= -var].mean() if len(port_returns[port_returns <= -var]) > 0 else var
        return cvar

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
    )
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_weights = num_assets * [1. / num_assets]
    
    opt_res = minimize(portfolio_cvar, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_res.x if opt_res.success else init_weights

# =====================================================================
# 2. 數據庫與多頁面設定 (正式繁體中文百大選單)
# =====================================================================
st.set_page_config(page_title="智慧量化選股與極端風險預測系統", layout="wide")

STOCK_DICT = {
    # 半導體與電子
    "2330 台積電": "2330.TW", "2317 鴻海": "2317.TW", "2454 聯發科": "2454.TW", 
    "2303 聯電": "2303.TW", "2308 台達電": "2308.TW", "2382 廣達": "2382.TW", 
    "2357 華碩": "2357.TW", "3711 日月光投控": "3711.TW", "3231 緯創": "3231.TW", 
    "2379 瑞昱": "2379.TW", "3034 聯詠": "3034.TW", "2324 仁寶": "2324.TW",
    # 金融股
    "2881 富邦金": "2881.TW", "2882 國泰金": "2882.TW", "2891 中信金": "2891.TW", 
    "2886 兆豐金": "2886.TW", "2884 玉山金": "2884.TW", "2892 第一金": "2892.TW", 
    "2880 華南金": "2880.TW", "2885 元大金": "2885.TW", "5871 中租-KY": "5871.TW",
    # 傳產與航運
    "2603 長榮": "2603.TW", "2609 陽明": "2609.TW", "2618 長榮航": "2618.TW", 
    "2002 中鋼": "2002.TW", "1301 台塑": "1301.TW", "1101 台泥": "1101.TW", 
    "1216 統一": "1216.TW", "9904 寶成": "9904.TW", "2912 統一超": "2912.TW"
}

# --- 側邊欄控制面板 ---
st.sidebar.header("🌐 系統功能導覽")
# 修正影片中的翻譯與錯字問題
mode = st.sidebar.radio("請選擇分析模組：", ["單股隨機極端風險預測", "Mean-CVaR 智慧資產配置"])

# =====================================================================
# 頁面一：單股隨機極端風險預測
# =====================================================================
if mode == "單股隨機極端風險預測":
    st.title("📊 單股隨機波動與極端風險預測")
    st.sidebar.markdown("---")
    selected_stock_name = st.sidebar.selectbox("請選取目標觀測標的：", list(STOCK_DICT.keys()))
    selected_stock_code = STOCK_DICT[selected_stock_name]

    # 下載真實歷史數據
    with st.spinner(f"正在從 Yahoo Finance 下載 {selected_stock_name} 歷史數據..."):
        try:
            stock_data = yf.download(selected_stock_code, start="2023-01-01")
            historical_close = stock_data['Close'].values.flatten()
            historical_returns = stock_data['Close'].pct_change().dropna().values.flatten()
            current_price = float(historical_close[-1])
        except Exception as e:
            st.error(f"數據抓取失敗，啟動防呆模擬機制。")
            current_price = 100.0
            historical_returns = np.random.normal(0.0005, 0.015, 500)

    st.sidebar.subheader("⚙️ Heston 模型參數校準")
    auto_calibrate = st.sidebar.checkbox("開啟歷史數據智慧自動校準", value=True)

    if auto_calibrate:
        mu_cal, v0_cal, kappa_cal, theta_cal, xi_cal, rho_cal = calibrate_heston_moments(historical_returns)
        st.sidebar.info("💡 系統已自動幫您校準最貼近該股現況的 Heston 參數。")
        mu = st.sidebar.number_input("預期報酬率 $\mu$", value=float(mu_cal))
        v0 = st.sidebar.number_input("初始波動率 $v_0$", value=float(v0_cal))
        kappa = st.sidebar.number_input("回歸速度 $\kappa$", value=float(kappa_cal))
        theta = st.sidebar.number_input("長期平均波動 $\\theta$", value=float(theta_cal))
        xi = st.sidebar.number_input("波動之波動度 $\\xi$", value=float(xi_cal))
        rho = st.sidebar.number_input("相關係數 $\\rho$", value=float(rho_cal))
    else:
        mu = st.sidebar.slider("預期報酬率 $\mu$", -0.2, 0.4, 0.08, 0.01)
        v0 = st.sidebar.slider("初始波動率 $v_0$", 0.05, 0.8, 0.2, 0.01)
        kappa = st.sidebar.slider("回歸速度 $\kappa$", 0.5, 5.0, 2.0, 0.1)
        theta = st.sidebar.slider("長期平均波動率 $\\theta$", 0.05, 0.8, 0.25, 0.01)
        xi = st.sidebar.slider("波動度的波動率 $\\xi$", 0.05, 1.0, 0.3, 0.01)
        rho = st.sidebar.slider("資產與波動相關係數 $\\rho$", -0.9, 0.9, -0.5, 0.1)

    if 2 * kappa * theta <= xi**2:
        st.sidebar.warning("⚠️ 未滿足 Feller 條件，隨機路徑波動率可能歸零。")

    # 蒙地卡羅運算
    N_days = 30
    M_paths = 1500  # 最佳化雲端流暢度的路徑數
    
    # 動態變更隨機數種子，保證每次換股票數據一定會動！
    try:
        stock_seed = int(''.join(filter(str.isdigit, selected_stock_code)))
    except:
        stock_seed = 42
    np.random.seed(stock_seed)
    
    S, v = heston_monte_carlo(current_price, v0, mu, kappa, theta, xi, rho, T=N_days/252, N=N_days, M=M_paths)

    final_returns = (S[-1] - current_price) / current_price
    alpha = 0.05
    var_95 = -np.percentile(final_returns, alpha * 100)
    cvar_95 = -final_returns[final_returns <= -var_95].mean() if len(final_returns[final_returns <= -var_95]) > 0 else var_95
    w_inf_score = calculate_w_infinity(final_returns, historical_returns[-N_days:]) 

    # --- 畫面渲染 ---
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 標的極端風險指標")
        st.metric(label="目前真實股價", value=f"{current_price:.2f} TWD")
        st.metric(label="未來 30 天預期平均漲跌幅", value=f"{final_returns.mean()*100:.2f} %")
        st.metric(label="95% 傳統風險值 (VaR)", value=f"{var_95*100:.2f} %")
        st.metric(label="95% 條件風險值 (CVaR)", value=f"{cvar_95*100:.2f} %")
        st.markdown("---")
        st.metric(label="極大瓦瑟斯坦距離 ($W_\\infty$)", value=f"{w_inf_score:.4f}")

    with col2:
        st.subheader("📈 預測趨勢扇形圖 (專業金融漸層版)")
        time_axis = np.arange(N_days + 1)
        p_min, p_16, p_50, p_84, p_max = np.percentile(S, [2.5, 16, 50, 84, 97.5], axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.fill_between(time_axis, p_min, p_16, color='tomato', alpha=0.1, label='95% 極端下檔風險')
        ax.fill_between(time_axis, p_84, p_max, color='limegreen', alpha=0.1, label='95% 極端上行潛力')
        ax.fill_between(time_axis, p_16, p_84, color='royalblue', alpha=0.35, label='68% 常態波動區間')
        ax.plot(time_axis, p_50, color='#0F2080', lw=2.5, label='預期中位數路徑')
        ax.axhline(y=current_price, color='red', linestyle=':', label='目前股價基準線')
        ax.set_title(f"{selected_stock_name} 蒙地卡羅多情境極端風險預測", fontsize=14, fontweight='bold')
        ax.set_xlabel("未來交易日 (Days)")
        ax.set_ylabel("股價預估 (TWD)")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)

# =====================================================================
# 頁面二：Mean-CVaR 智慧資產配置
# =====================================================================
elif mode == "Mean-CVaR 智慧資產配置":
    st.title("💼 Mean-CVaR 百大成分股智慧投資組合優化系統")
    st.markdown("本模組直接利用歷史尾端風險 **CVaR** 作為優化核心，在滿足您的目標回報率前提下，為您調配極端風險最低的黃金資產權重比例。")
    
    selected_pool = st.multiselect("請選取要納入投資組合池的股票（至少 3 檔）：", list(STOCK_DICT.keys()), default=list(STOCK_DICT.keys())[:4])
    target_return_input = st.slider("期望年化回報率目標 (%)", 5, 25, 12, 1) / 100
    
    if len(selected_pool) >= 3:
        with st.spinner("正在下載組合股價並執行 Mean-CVaR 風險最小化優化運算..."):
            pool_codes = [STOCK_DICT[name] for name in selected_pool]
            data = yf.download(pool_codes, start="2023-01-01")['Close']
            
            pool_returns = data.pct_change().dropna()
            mean_returns = pool_returns.mean().values * 252
            hist_returns_matrix = pool_returns.values
            
            optimal_weights = optimize_mean_cvar(mean_returns, hist_returns_matrix, target_return_input)
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 智慧配置最佳權重")
            weight_df = pd.DataFrame({
                "自選標的名稱": selected_pool,
                "最佳配置比例 (%)": np.round(optimal_weights * 100, 2)
            })
            st.dataframe(weight_df.style.format({"最佳配置比例 (%)": "{:.2f}%"}))
            
        with col2:
            st.subheader("📊 資產權重分佈圖")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(optimal_weights, labels=selected_pool, autopct='%1.1f%%', startangle=90)
            ax.axis('equal') 
            st.pyplot(fig)
    else:
        st.info("💡 請在上方至少勾選 3 檔股票以啟動 Mean-CVaR 優化選股矩陣。")
