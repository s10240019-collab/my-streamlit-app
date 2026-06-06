import yfinance as yf
import pandas as pd
import numpy as np

# ==========================================
# 1. 定義標的與資料下載
# ==========================================
# 選擇 5 檔台灣 0050 成分股作為初期測試 (台積電、鴻海、聯發科、富邦金、台塑)
tickers = ['2330.TW', '2317.TW', '2454.TW', '2881.TW', '1301.TW']

print("開始從 yfinance 獲取歷史資料...")
# 下載過去 3 年的歷史資料，並直接提取 'Adj Close' (還原權息後的收盤價)
# 使用還原收盤價才能正確計算真實報酬率
data = yf.download(tickers, start="2021-01-01", end="2023-12-31")['Adj Close']

# 檢查是否有缺失值並進行填補 (以前一日價格填補)
data = data.ffill().dropna()

print("\n資料下載完成，前五筆資料如下：")
print(data.head())

# ==========================================
# 2. 計算對數報酬率 (Log Returns)
# ==========================================
# 在量化金融與隨機微分方程式中，通常使用對數報酬率而非簡單報酬率
# 公式: R_t = ln(S_t / S_{t-1})
log_returns = np.log(data / data.shift(1)).dropna()

# ==========================================
# 3. 基礎極端風險指標計算 (歷史模擬法)
# ==========================================
def calculate_historical_risk(returns_series, alpha=0.95):
    """
    計算給定信賴水準下的歷史風險值 (VaR) 與條件風險值 (CVaR)
    """
    # 計算 (1 - alpha) 的分位數，即 VaR 的臨界值
    var_threshold = returns_series.quantile(1 - alpha)
    
    # 計算小於等於 VaR 臨界值的報酬率平均，即 CVaR (預期尾端損失)
    cvar_value = returns_series[returns_series <= var_threshold].mean()
    
    return var_threshold, cvar_value

print("\n==========================================")
print(" 基礎極端風險指標 (95% 信賴水準)")
print("==========================================")

for ticker in tickers:
    var, cvar = calculate_historical_risk(log_returns[ticker], alpha=0.95)
    # 風險指標通常以正數（損失的百分比）來表示，故加上負號與百分比轉換
    print(f"[{ticker}] 歷史 VaR: {-var * 100:>5.2f}% | 歷史 CVaR: {-cvar * 100:>5.2f}%")


# ==========================================
# 4. 進階模型擴充區 (預留給學生的實作空間)
# ==========================================

def heston_euler_maruyama(S0, v0, mu, kappa, theta, xi, rho, T, N, num_paths):
    """
    [TODO] 學生專題任務 1：
    在此處實作 Heston 模型的 Euler-Maruyama 離散化模擬。
    需產出 num_paths 條長度為 N 的資產價格路徑。
    """
    pass

def calculate_max_wasserstein(dist_sim, dist_real):
    """
    [TODO] 學生專題任務 2：
    在此處實作 W_infty 距離的計算。
    可用於衡量 Heston 模型模擬出的報酬率分佈，與歷史真實分佈的極端偏差。
    (建議提示學生可研究 scipy.optimize 或 POT (Python Optimal Transport) 套件)
    """
    pass
