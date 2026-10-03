"""Next-day stock price prediction.

Approach: predict tomorrow's *return* from technical features (lagged returns,
moving-average gaps, volatility, RSI), then convert it to a price.
Evaluation uses a time-ordered split (last 20% = test) and is compared with a
naive baseline that predicts "tomorrow = today".
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 120


def make_features(close):
    f = pd.DataFrame(index=close.index)
    ret = close.pct_change()
    for lag in range(1, 6):
        f[f"ret_lag{lag}"] = ret.shift(lag - 1)
    for w in (5, 10, 20):
        f[f"ma{w}_gap"] = close.rolling(w).mean() / close - 1
    f["vol5"] = ret.rolling(5).std()
    f["vol20"] = ret.rolling(20).std()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100
    return f


def build_model(name):
    if name == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1)


def metrics(actual, pred):
    return {
        "mae": round(float(mean_absolute_error(actual, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, pred))), 4),
        "r2": round(float(r2_score(actual, pred)), 4),
    }


def run_prediction(df, model_name="random_forest"):
    close = df["Close"].astype(float)
    if len(close) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows of data, got {len(close)}.")

    feats = make_features(close).dropna()
    target = close.pct_change().shift(-1)           # tomorrow's return
    train_df = feats.join(target.rename("y")).dropna()

    split = int(len(train_df) * 0.8)
    train, test = train_df.iloc[:split], train_df.iloc[split:]
    cols = [c for c in train_df.columns if c != "y"]

    model = build_model(model_name).fit(train[cols], train["y"])
    pred_ret = model.predict(test[cols])

    today = close.loc[test.index]
    actual_next = today * (1 + test["y"])
    pred_next = today * (1 + pred_ret)
    naive_next = today                               # baseline: no change

    direction = float(np.mean(np.sign(pred_ret) == np.sign(test["y"])) * 100)

    latest = feats.iloc[[-1]]
    next_ret = float(model.predict(latest[cols])[0])
    last_close = float(close.iloc[-1])
    next_date = pd.bdate_range(close.index[-1], periods=2)[1]

    n = min(150, len(test))
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index[-n:]],
        "actual": [round(float(v), 2) for v in actual_next.iloc[-n:]],
        "predicted": [round(float(v), 2) for v in pred_next.iloc[-n:]],
        "model_metrics": metrics(actual_next, pred_next),
        "baseline_metrics": metrics(actual_next, naive_next),
        "direction_accuracy": round(direction, 1),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "last_close": round(last_close, 2),
        "last_date": close.index[-1].strftime("%Y-%m-%d"),
        "forecast": {
            "date": next_date.strftime("%Y-%m-%d"),
            "price": round(last_close * (1 + next_ret), 2),
            "change_pct": round(next_ret * 100, 2),
        },
    }
