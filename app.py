
from flask import Flask, request, render_template
import pandas as pd
import joblib

# Create Flask application
app = Flask(__name__)

# Load the saved machine learning model
saved_data = joblib.load("house_price_model.pkl")

model = saved_data["model"]
feature_columns = saved_data["feature_columns"]

print("XGBoost model loaded successfully!")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get numerical values from the HTML form
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])

        waterfront = int(request.form["waterfront"])
        view = int(request.form["view"])
        condition = int(request.form["condition"])

        sqft_above = float(request.form["sqft_above"])
        sqft_basement = float(request.form["sqft_basement"])

        yr_built = int(request.form["yr_built"])
        yr_renovated = int(request.form["yr_renovated"])

        # Get location details
        city = request.form["city"].strip()
        statezip = request.form["statezip"].strip()
        country = request.form["country"].strip()

        # Check location fields
        if not city or not statezip or not country:
            raise ValueError("Please enter all location details.")

        # Create a DataFrame with user input
        input_data = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "city": city,
            "statezip": statezip,
            "country": country
        }])

        # Arrange columns in the same order as training data
        input_data = input_data[feature_columns]

        # Predict house price
        predicted_price = model.predict(input_data)[0]

        # Send prediction to HTML
        return render_template(
            "index.html",
            prediction=predicted_price
        )

    except Exception as e:

        # Display error message in HTML
        return render_template(
            "index.html",
            error=str(e)
        )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)
