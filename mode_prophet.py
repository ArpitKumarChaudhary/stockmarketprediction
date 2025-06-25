import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv("BTC-USD.csv")

# Prepare the dataframe for Prophet (columns: ds, y)
df = df[["Date", "Close"]].dropna()
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])
df = df.rename(columns={"Date": "ds", "Close": "y"})

# Initialize and train Prophet model
model = Prophet(daily_seasonality=True)
model.fit(df)

# Create future dataframe for prediction
future = model.make_future_dataframe(periods=30)  # forecast 30 days ahead
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title("BTC-USD Forecast using Prophet")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate last 30 actual vs predicted (if available)
actual = df['y'].iloc[-30:].values
predicted = forecast['yhat'].iloc[-60:-30].values  # last 30 known predictions

mae = mean_absolute_error(actual, predicted)
print(f"✅ Prophet MAE: {mae:.2f}")
