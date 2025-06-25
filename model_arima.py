import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv("BTC-USD.csv", parse_dates=["Date"], index_col="Date")

# Convert 'Close' to numeric (clean out strings or garbage)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Drop NaNs in Close column
df = df.dropna(subset=["Close"])

# Split data (90% train, 10% test)
split = int(len(df) * 0.9)
train, test = df[:split], df[split:]

# VERY IMPORTANT: ensure train["Close"] is float, not object
train_close = train["Close"].astype("float64")

# Fit ARIMA model
model = ARIMA(train_close, order=(5, 1, 0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train["Close"], label="Train")
plt.plot(test["Close"], label="Actual")
plt.plot(test.index, forecast, label="Forecast (ARIMA)", linestyle="--")
plt.title("BTC-USD Forecast using ARIMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate
rmse = np.sqrt(mean_squared_error(test["Close"], forecast))
print(f"✅ ARIMA RMSE: {rmse:.2f}")
