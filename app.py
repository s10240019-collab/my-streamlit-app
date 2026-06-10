import streamlit as st
import yfinance as yf

st.title("股票數據儀表板")

ticker = st.text_input("請輸入股票代號（例如 AAPL 或 TSM）：", value="TSM")

if ticker:
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period="1mo")
    
    # 檢查抓出來的資料表是不是空的
    if df.empty:
        st.error("找不到該股票代號，請重新輸入！")
    else:
        # 確保有 'Close' 欄位才畫圖
        if 'Close' in df.columns:
            st.subheader(f"{ticker} 過去一個月的歷史股價")
            st.line_chart(df['Close'])
            st.dataframe(df)
        else:
            st.error("資料格式異常，缺少收盤價數據。")
