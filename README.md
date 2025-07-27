1. Set up environment
(recommended: use virtualenv or conda)
python -m venv venv
source venv/bin/activate       (Linux/macOS)
OR
venv\Scripts\activate          (Windows)


2. Install all dependencies
pip install -r requirements.txt


3. Run the data collection script
Downloads historical stock/crypto data using yFinance

python fetch_data.py

(Inside fetch_data.py)
Define stock ticker (e.g., BTC-USD)
Set date range
Download and save data to CSV


4. Preprocess the data
python preprocess.py

(Inside preprocess.py)
4.1 Load CSV
4.2 Handle missing values
4.3 Feature engineering (moving averages, returns, volatility)


5. Perform Spark-based analysis (optional)
python spark_analysis.py

(Inside spark_analysis.py)
Load data into PySpark
Compute rolling stats, trends, etc.


6. Train the prediction model
python train_model.py

(Inside train_model.py)
Split data (train/test)
Use ML model (e.g., Linear Regression, LSTM)
Train and evaluate

7. Make predictions and plot results
python predict_and_plot.py

(Inside predict_and_plot.py)
Load trained model
Predict next N days
Plot actual vs predicted prices

8. (Optional) Launch Jupyter Notebook
Use `stock_prediction.ipynb` to interactively run steps 3–7
