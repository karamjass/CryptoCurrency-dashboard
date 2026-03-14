import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

FEATURES = ["Open", "High", "Low", "Close", "Volume", "Log_Return"]
LOOKBACK = 90

st.title("BTC Next Day Predictor")

try:
    
    model = load_model("btc_lstm_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")

    btc = yf.download("BTC-USD", period="1y", auto_adjust=False, progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc[["Open", "High", "Low", "Close", "Volume"]]
    btc["Log_Return"] = np.log(btc["Close"]).diff()
    btc = btc.dropna()

    if len(btc) < LOOKBACK:
        st.error(f"Not enough data to predict. Need at least {LOOKBACK} rows.")
    else:
        scaled = scaler.transform(btc[FEATURES].values)
        x_input = scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURES))

        pred_scaled = float(model.predict(x_input, verbose=0)[0][0])
        dummy = np.zeros((1, len(FEATURES)))
        dummy[0, -1] = pred_scaled
        pred_return = float(scaler.inverse_transform(dummy)[0, -1])

        next_price = float(btc["Close"].iloc[-1] * np.exp(pred_return))
        last_close = float(btc["Close"].iloc[-1])
        delta_value = next_price - last_close
        delta_pct = (delta_value / last_close) * 100 if last_close else 0.0

        st.metric(
            "Next Day Predicted Price",
            f"${next_price:,.2f}",
            f"{delta_value:+,.2f} ({delta_pct:+.2f}%) vs last close",
        )

        last_30 = btc["Close"].tail(30)
        next_day = btc.index[-1] + pd.Timedelta(days=1)

        st.subheader("Last 30 Days + Next Day Forecast")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_30.index, last_30.values, label="Actual (Last 30 Days)", linewidth=2)
        ax.scatter([next_day], [next_price], color="orange", label="Next Day Forecast", zorder=3)
        ax.plot([last_30.index[-1], next_day], [last_30.values[-1], next_price], linestyle="--", alpha=0.6)
        ax.set_ylabel("BTC Price (USD)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
except Exception as e:
    st.error(f"Error: {e}")
