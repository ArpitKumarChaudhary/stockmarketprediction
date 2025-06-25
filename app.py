from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)

# Load LSTM model once
model = load_model("lstm_model.h5")

@app.route("/")
def index():
    return send_file("frontend.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get("stock", "BTC-USD")
        days = int(data.get("days", 7))

        filename = f"{symbol}.csv"
        if not os.path.exists(filename):
            return jsonify({"error": f"CSV file '{filename}' not found"}), 404

        # Load and clean CSV
        df = pd.read_csv(filename, parse_dates=["Date"])
        df = df[["Date", "Open", "High", "Low", "Close"]]
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
        df.dropna(subset=["Close", "Open", "High", "Low"], inplace=True)

        # Filter by date
        start = data.get("startDate")
        end = data.get("endDate")
        if start:
            df = df[df["Date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["Date"] <= pd.to_datetime(end)]

        if len(df) < 60:
            print(f"[ERROR] After filtering {symbol} from {start} to {end}, only {len(df)} rows found.")
            return jsonify({"error": f"Not enough data after filtering from {start} to {end} (only {len(df)} rows, need at least 60)"}), 400

        # OHLC summary (from filtered data)
        ohlc = {
            "open": float(df.iloc[0]["Open"]),
            "high": float(df["High"].max()),
            "low": float(df["Low"].min()),
            "close": float(df.iloc[-1]["Close"])
        }

        # Normalize close price
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
        last_60 = scaled_data[-60:]
        forecast_input = np.reshape(last_60, (1, 60, 1))

        # Predict future prices
        predictions = []
        for _ in range(days):
            pred = model.predict(forecast_input, verbose=0)[0][0]
            predictions.append(pred)
            forecast_input = np.append(forecast_input[0, 1:], [[pred]], axis=0)
            forecast_input = np.reshape(forecast_input, (1, 60, 1))

        forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actual_prices = scaler.inverse_transform(last_60).flatten()

        # Return raw data between start and end (Date + OHLC)
        filtered_data = df[["Date", "Open", "High", "Low", "Close"]].copy()
        filtered_data["Date"] = filtered_data["Date"].astype(str)  # convert to JSON-serializable

        return jsonify({
            "actual": actual_prices.tolist(),
            "predicted": forecast_prices.tolist(),
            "ohlc": ohlc,
            "raw_data": filtered_data.to_dict(orient="records")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
