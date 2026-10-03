# Stock Price Prediction

A Flask web app that predicts the **next trading day's closing price** with machine learning.

## How it works
1. Load daily closing prices (sample data, a CSV file, or a live ticker).
2. Build features: lagged returns, moving-average gaps (5/10/20 days), volatility and RSI.
3. Train **Random Forest** or **Ridge Linear Regression** to predict tomorrow's *return*, then convert it to a price.
4. Evaluate on the **last 20% of dates** (time-ordered split, no shuffling) with MAE, RMSE, R² and directional accuracy.
5. Compare against a **naive baseline** (tomorrow = today). A real model should beat it.

## Setup (Python 3.10 to 3.12 recommended)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

Optional, for live tickers: `pip install -r requirements-optional.txt` (needs internet).

## Using your own data
Upload a CSV with at least `Date` and `Close` columns (at least 120 rows).
Download real history from Yahoo Finance (Historical Data tab) if you prefer not to install yfinance.

## Files
- `app.py`: Flask server
- `data_loader.py`: sample / CSV / ticker loading
- `model.py`: features, training, evaluation
- `templates/index.html`: web page with chart
- `data/sample_stock.csv`: **synthetic** demo data, not a real stock

## Note
Stock prices are noisy and hard to predict. Results on sample data are for demonstration only. This is a learning project, not financial advice.
