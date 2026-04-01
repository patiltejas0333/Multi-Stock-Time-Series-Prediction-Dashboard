import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Title
st.title("📈 Multi-Stock Time Series Prediction Dashboard")

# Sidebar
st.sidebar.header("Stock Settings")

# Multiple stock selection
stocks = st.sidebar.multiselect(
    "Select Stocks",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
    default=["RELIANCE.NS"]
)

start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["AR", "MA", "ARMA", "ARIMA"]
)

# -----------------------
# Load Data
# -----------------------
with st.spinner("Downloading data..."):
    data = yf.download(stocks, start=start, end=end)

if not data.empty:

    st.subheader("📊 Stock Prices Comparison")

    fig, ax = plt.subplots()

    # Plot multiple stocks
    for stock in stocks:
        ax.plot(data["Close"][stock], label=stock)

    ax.legend()
    st.pyplot(fig)

    # -----------------------
    # Select one stock for modeling
    # -----------------------
    selected_stock = st.selectbox("Select Stock for Prediction", stocks)

    close = data["Close"][selected_stock].dropna()

    st.subheader(f"📄 Data for {selected_stock}")
    st.write(close.tail())

    # -----------------------
    # ACF & PACF
    # -----------------------
    st.subheader("📉 ACF & PACF Analysis")

    fig_acf, ax_acf = plt.subplots()
    plot_acf(close, ax=ax_acf, lags=40)
    st.pyplot(fig_acf)

    fig_pacf, ax_pacf = plt.subplots()
    plot_pacf(close, ax=ax_pacf, lags=40)
    st.pyplot(fig_pacf)

    # -----------------------
    # MODEL SELECTION
    # -----------------------
    if model_choice == "AR":
        p = st.slider("p (lags)", 1, 10, 2)
        model = ARIMA(close, order=(p, 0, 0))

    elif model_choice == "MA":
        q = st.slider("q (lags)", 1, 10, 2)
        model = ARIMA(close, order=(0, 0, q))

    elif model_choice == "ARMA":
        p = st.slider("p", 1, 5, 2)
        q = st.slider("q", 1, 5, 2)
        model = ARIMA(close, order=(p, 0, q))

    elif model_choice == "ARIMA":
        p = st.slider("p", 1, 5, 2)
        d = st.slider("d", 0, 2, 1)
        q = st.slider("q", 1, 5, 2)
        model = ARIMA(close, order=(p, d, q))

    # Fit model
    with st.spinner("Training model and forecasting..."):
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=30)

    # -----------------------
    # Forecast Plot
    # -----------------------
    fig2, ax2 = plt.subplots()

    ax2.plot(close, label="Original")
    ax2.plot(range(len(close), len(close)+30),
             forecast, linestyle="dashed", label="Forecast")

    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📅 Next 30 Days Prediction")
    st.write(forecast)

else:
    st.warning("No data found")

st.markdown("---")
st.write("Made with Streamlit and statsmodels. Run with: `streamlit run multi_stock_prediction.py`")