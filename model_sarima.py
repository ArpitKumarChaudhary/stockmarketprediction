import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("BTC-USD.csv", parse_dates=["Date"], index_col="Date")

# Ensure Close column is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])

# Train-test split (last 10% for test)
split = int(len(df) * 0.9)
train, test = df.iloc[:split], df.iloc[split:]

# Fit SARIMA model: (p,d,q)(P,D,Q,s)
# s=7 assumes weekly seasonality (7 days)
model = SARIMAX(train["Close"],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# Forecast for test period
forecast = model_fit.forecast(steps=len(test))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(train["Close"], label="Train")
plt.plot(test["Close"], label="Actual")
plt.plot(test.index, forecast, label="Forecast (SARIMA)", linestyle="--")
plt.title("BTC-USD Forecast using SARIMA")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate
rmse = np.sqrt(mean_squared_error(test["Close"], forecast))
print(f"✅ SARIMA RMSE: {rmse:.2f}")
