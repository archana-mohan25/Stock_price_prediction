"""Load price data from a CSV file, a ticker (yfinance) or the bundled sample."""
import os
import pandas as pd

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample_stock.csv")


def _clean(df):
    cols = {c.lower().strip(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError("CSV needs 'Date' and 'Close' columns.")
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[cols["date"]], errors="coerce"),
        "Close": pd.to_numeric(df[cols["close"]], errors="coerce"),
    }).dropna().sort_values("Date").drop_duplicates("Date")
    return out.set_index("Date")


def from_csv(file_or_path):
    return _clean(pd.read_csv(file_or_path))


def from_sample():
    return from_csv(SAMPLE)


def from_ticker(ticker, period="5y"):
    try:
        import yfinance as yf
    except ImportError:
        raise ValueError("yfinance is not installed. Run: pip install yfinance  (or upload a CSV instead).")
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol and your internet connection.")
    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return _clean(hist[["Date", "Close"]])
