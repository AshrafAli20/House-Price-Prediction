
# ==========================================
# HOUSE PRICE PREDICTION USING XGBOOST
# ==========================================

# 1. Import required libraries

import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 2. Load the dataset

df = pd.read_csv("data.csv")

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# 3. Remove date and street columns

df = df.drop(
    columns=["date", "street"],
    errors="ignore"
)

print("\nColumns after removing date and street:")
print(df.columns.tolist())


# 4. Remove invalid house prices

df = df[df["price"] > 0]

print("\nDataset shape after removing invalid prices:")
print(df.shape)

print("Invalid prices remaining:", (df["price"] <= 0).sum())


# 5. Separate input features and target

X = df.drop(columns=["price"])

y = df["price"]


# 6. Identify numerical and categorical columns

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# 7. Prepare numerical preprocessing

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])


# 8. Prepare categorical preprocessing

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 9. Combine preprocessing steps

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# 10. Create XGBoost model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),

    ("regressor", XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    ))
])


# 11. Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])


# 12. Train the model

model.fit(X_train, y_train)

print("\nXGBoost model training completed!")


# 13. Predict house prices

y_pred = model.predict(X_test)


# 14. Evaluate the model

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== XGBOOST MODEL EVALUATION ==========")

print("Mean Absolute Error:", mae)

print("Root Mean Squared Error:", rmse)

print("R2 Score:", r2)


# 15. Save the trained model

joblib.dump(
    {
        "model": model,
        "feature_columns": X.columns.tolist()
    },
    "house_price_model.pkl"
)

print("\nModel saved successfully!")

print("Saved file: house_price_model.pkl")


# 16. Display actual and predicted prices

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n========== ACTUAL VS PREDICTED PRICES ==========")

print(comparison.head(10).round(2).to_string(index=False))


# 17. Dataset summary

print("\n========== DATASET SUMMARY ==========")

print("Total records:", len(df))

print("Total input features:", X.shape[1])

print("Average house price:", round(y.mean(), 2))

print("Minimum house price:", round(y.min(), 2))

print("Maximum house price:", round(y.max(), 2))
