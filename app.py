import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Initialize app
app = Flask(__name__)

# Load trained model
model = joblib.load('mobile price/mobile_model.pkl')
# Home route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/options", methods=["GET"])
def get_options():
    try:
        preprocessor = model.named_steps["preprocessing"]

        options = {}

        # Loop through transformers inside ColumnTransformer
        for name, transformer, columns in preprocessor.transformers_:
            
            if name == "encoder":  # this matches your OneHotEncoder
                encoder = transformer
                categories = encoder.categories_

                for col, cats in zip(columns, categories):
                    options[col] = list(cats)

        return jsonify(options)

    except Exception as e:
        return jsonify({"error": str(e)})
    
# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        import math

        # ---- Calculate PPI ----
        ppi = math.sqrt(
            (float(data["resolution_x"])**2 +
            float(data["resolution_y"])**2)**0.5
        ) / float(data["screen_size"])

        # ---- Calculate Performance Score ----
        performance_score = (
            float(data["ram"]) * float(data["clock"])
        )

        # ---- Create dataframe ----
        input_data = pd.DataFrame([{
            "rating": float(data["rating"]),
            "os": data["os"],
            "network_type": data["network_type"],
            "NFC": int(data["NFC"]),
            "chipset": data["chipset"],
            "core_type": data["core_type"],
            "battery_mah": float(data["battery_mah"]),
            "display_type": data["display_type"],
            "rear_camera": float(data["rear_camera"]),
            "company_name": data["company_name"],
            "ppi": ppi,
            "performance_score": performance_score
        }])

        prediction_log = model.predict(input_data)[0]
        prediction = np.expm1(prediction_log)

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)