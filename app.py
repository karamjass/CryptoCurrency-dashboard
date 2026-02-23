import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline

# ==============================
# PAGE CONFIG
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

st.title("🚀 Crypto AI Trading Dashboard")

# ==============================
# SIDEBAR SETTINGS
# ==============================
st.sidebar.header("⚙ Settings")

coin = st.sidebar.selectbox("Select Crypto", ["BTC-USD", "ETH-USD", "SOL-USD"])

# ==============================
# LOAD LIVE DATA
# ==============================
data = yf.download(coin, period="5y")
data.columns = data.columns.get_level_values(0)

# ==============================
# FEATURE ENGINEERING
# ==============================
data['Returns'] = data['Close'].pct_change()
data['MA_7'] = data['Close'].rolling(7).mean()
data['MA_30'] = data['Close'].rolling(30).mean()
data['Volatility'] = data['Returns'].rolling(7).std()

data.dropna(inplace=True)

# ==============================
# METRICS
# ==============================
col1, col2, col3 = st.columns(3)

current_price = data['Close'].iloc[-1]
highest_price = data['High'].max()
lowest_price = data['Low'].min()

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

series = data['Close']

model_arima = ARIMA(series, order=(5,1,0))
model_fit = model_arima.fit()

arima_pred = model_fit.forecast(steps=30)

st.success("Prediction Generated!")

# =======================
# PREDICTION GRAPH
# =======================
fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(arima_pred)
ax2.set_title("Next 30 Days Forecast")
st.pyplot(fig2)

# =======================
# SHOW PREDICTED VALUES
# =======================

next_day_price = arima_pred.iloc[0]
last_day_price = arima_pred.iloc[-1]

colA, colB = st.columns(2)

colA.metric("Next Day Predicted Price", f"${next_day_price:,.2f}")
colB.metric("30th Day Predicted Price", f"${last_day_price:,.2f}")

# =======================
# FULL TABLE
# =======================

pred_df = pd.DataFrame(arima_pred)
pred_df.columns = ["Predicted Price"]
pred_df.index = pred_df.index.date
pred_df["Predicted Price"] = pred_df["Predicted Price"].map(lambda x: f"${x:,.2f}")

st.subheader("📊 Full 30-Day Prediction")
st.dataframe(pred_df, use_container_width=True, height=300)

# ==============================
# TRADING SIGNAL
# ==============================
st.subheader("📊 Trading Signal")

if data['MA_7'].iloc[-1] > data['MA_30'].iloc[-1]:
    signal = "BUY"
else:
    signal = "SELL"

st.metric("Signal", signal)

# ==============================
# RISK LEVEL
# ==============================
st.subheader("⚠ Risk Meter")

if data['Volatility'].iloc[-1] > 0.02:
    risk = "HIGH"
else:
    risk = "LOW"

st.metric("Market Risk", risk)

# ==============================
# SENTIMENT ANALYSIS
# ==============================
st.subheader("🧠 News Sentiment")

user_news = st.text_input("Enter crypto news headline")

if user_news:

    sentiment_model = pipeline("sentiment-analysis")
    result = sentiment_model(user_news)[0]

    label = result['label']
    score = result['score']

    if label == "POSITIVE":
        sentiment_score = score
    else:
        sentiment_score = -score

    st.metric("Sentiment", label)
    st.metric("Sentiment Score", round(sentiment_score,3))
import requests

st.subheader("📰 Live Crypto News")

api_key = "bcf5ccb3ab7d456c83a7ff782d1ed725"

url = f"https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={api_key}"

news_data = requests.get(url).json()

articles = news_data['articles'][:5]

for article in articles:
    st.write("###", article['title'])
    st.write(article['source']['name'])
    st.write(article['url'])