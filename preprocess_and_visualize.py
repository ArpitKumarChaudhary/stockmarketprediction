import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("BTC-USD.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Ensure Close column is numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")  # converts bad data to NaN

# Drop rows where Close is NaN (e.g. due to 'BTC-USD' or blank)
df = df.dropna(subset=["Close"])

# Compute rolling mean
df["Rolling Mean"] = df["Close"].rolling(window=30).mean()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Close")
plt.plot(df["Rolling Mean"], label="30-Day Rolling Mean", linestyle="--")
plt.title("BTC-USD Closing Price with Rolling Average")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
