import yfinance as yf
import pandas as pd

# Download MRF.BO data (adjusted closing is now default)
df = yf.download("MRF.BO", start="2021-01-01", end="2025-06-24", auto_adjust=False)

# Reset index to include 'Date' as a column
df.reset_index(inplace=True)

# Print columns to confirm
print("Downloaded columns:", df.columns)

# Save only the available columns (skip 'Adj Close' if missing)
df.to_csv("MRF.BO.csv", index=False)
print("✅ MRF.BO.csv saved successfully.")
