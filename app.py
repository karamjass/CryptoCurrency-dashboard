import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# ==============================
# PAGE SETTINGS + THEME
# ==============================
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Crypto Price Prediction Dashboard")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Settings")

coin = st.sidebar.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "SOL-USD"])

# ==============================
# LOAD LIVE DATA
# ==============================
data = yf.download(coin, period="5y")

# remove ticker multi index
try:
    data.columns = data.columns.droplevel(1)
except:
    pass

st.subheader(f"{coin} Historical Data")

# ==============================
# CLEAN METRICS
# ==============================
current_price = data["Close"].iloc[-1]
highest_price = data["High"].max()
lowest_price = data["Low"].min()

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Highest Price", f"${highest_price:,.2f}")
col3.metric("Lowest Price", f"${lowest_price:,.2f}")

# ==============================
# PRICE GRAPH
# ==============================
st.subheader("📈 Price Trend")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(data['Close'])
ax.set_title(f"{coin} Closing Price")
st.pyplot(fig)

# ==============================
# ARIMA PREDICTION
# ==============================
st.subheader("🔮 Future Prediction")

if st.button("Predict Next 30 Days"):

    series = data['Close']

    model_arima = ARIMA(series, order=(5,1,0))
    model_arima_fit = model_arima.fit()

    arima_pred = model_arima_fit.forecast(steps=30)

    st.success("Prediction Generated!")

    # ==========================
    # Prediction Graph
    # ==========================
    fig2, ax2 = plt.subplots(figsize=(12,5))
    ax2.plot(arima_pred)
    ax2.set_title("Next 30 Days Prediction")
    st.pyplot(fig2)

    # ==========================
    # Prediction Cards
    # ==========================
    st.subheader("📅 Next 5 Days Prediction")

    cols = st.columns(5)
    for i in range(5):
        cols[i].metric(
            label=str(arima_pred.index[i].date()),
            value=f"${arima_pred.iloc[i]:,.2f}"
        )

    # ==========================
    # FORMATTED TABLE
    # ==========================
    st.subheader("📊 All Predicted Prices")

    pred_df = pd.DataFrame(arima_pred)
    pred_df.columns = ["Predicted Price"]

    pred_df["Predicted Price"] = pred_df["Predicted Price"].map(lambda x: f"${x:,.2f}")
    pred_df.index = pred_df.index.date

    st.dataframe(pred_df, use_container_width=True, height=350)