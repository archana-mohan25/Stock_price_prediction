import traceback

from flask import Flask, jsonify, render_template, request

import data_loader
from model import run_prediction

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        source = request.form.get("source", "sample")
        model_name = request.form.get("model", "random_forest")
        if source == "csv":
            file = request.files.get("file")
            if not file:
                return jsonify(error="Please choose a CSV file."), 400
            df = data_loader.from_csv(file)
        elif source == "ticker":
            ticker = request.form.get("ticker", "").strip().upper()
            if not ticker:
                return jsonify(error="Please enter a ticker symbol."), 400
            df = data_loader.from_ticker(ticker)
        else:
            df = data_loader.from_sample()
        result = run_prediction(df, model_name)
        result["source"] = source
        return jsonify(result)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify(error=f"Server error: {type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
