# House Price Prediction - end-to-end
# Run in Jupyter or a Python environment with sklearn, pandas, numpy, matplotlib

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.frame.drop(columns=["MedHouseVal"])  # features
y = data.frame["MedHouseVal"]                 # target (median house value)

# Quick peek
print("Feature columns:", list(X.columns))
print("Rows:", X.shape[0])

# Basic feature engineering
# Add a couple of simple engineered features:
# - rooms_per_household = AveRooms / AveOccup
# - bedrooms_per_room = AveBedrms / AveRooms
def add_features(df):
    df = df.copy()
    # avoid division by zero
    df["rooms_per_household"] = df["AveRooms"] / np.where(df["AveOccup"] == 0, 1, df["AveOccup"])
    df["bedrooms_per_room"] = df["AveBedrms"] / np.where(df["AveRooms"] == 0, 1, df["AveRooms"])
    return df

feat_transformer = FunctionTransformer(add_features)

# Apply engineered features now so column list is updated
X = feat_transformer.transform(X)
# Now list columns
all_columns = list(X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numeric features
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
# There are no categorical features in this dataset, but code kept flexible
categorical_features = []  # none here

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    # if you had categorical: ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder="drop")

# Model pipelines
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Train models
print("\nTraining Linear Regression...")
lr_pipeline.fit(X_train, y_train)

print("Training Ridge Regression...")
ridge_pipeline.fit(X_train, y_train)

print("Training Random Forest (default)...")
rf_pipeline.fit(X_train, y_train)

# Evaluate helper
def evaluate(model_pipeline, X_t, y_t, label="Model"):
    preds = model_pipeline.predict(X_t)
    rmse = mean_squared_error(y_t, preds) ** 0.5
    mae = mean_absolute_error(y_t, preds)
    r2 = r2_score(y_t, preds)
    print(f"\n{label} evaluation:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R^2:  {r2:.4f}")
    return preds, rmse, mae, r2

# Evaluate on test set
lr_preds, lr_rmse, lr_mae, lr_r2 = evaluate(lr_pipeline, X_test, y_test, "LinearRegression")
ridge_preds, ridge_rmse, ridge_mae, ridge_r2 = evaluate(ridge_pipeline, X_test, y_test, "RidgeRegression")
rf_preds, rf_rmse, rf_mae, rf_r2 = evaluate(rf_pipeline, X_test, y_test, "RandomForest (default)")

# Feature importances from Random Forest (after preprocessing)
# We need the preprocessor to transform training features to same order as model expects
preproc = rf_pipeline.named_steps["preprocessor"]
X_train_transformed = preproc.transform(X_train)  # numpy array
# column order coming from numeric_features (we used only numeric)
feature_names = numeric_features  # includes engineered features
# get feature importances
rf = rf_pipeline.named_steps["regressor"]
importances = rf.feature_importances_

# Sort and print top features
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 10 feature importances (Random Forest):")
print(feat_imp.head(10))

# Plot predicted vs actual for the best model (use RF here)
plt.figure(figsize=(7,6))
plt.scatter(y_test, rf_preds, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Random Forest: Predicted vs Actual")
plt.grid(True)
plt.show()

# Residual histogram
residuals = y_test - rf_preds
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=50)
plt.title("RF Residuals Histogram")
plt.xlabel("Residual (actual - predicted)")
plt.show()

# Hyperparameter tuning for Random Forest (quick)
param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [10, 20, None],
    "regressor__min_samples_leaf": [1, 3, 5]
}
gs = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
print("\nStarting GridSearchCV for Random Forest (this can take a while)...")
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV score (neg RMSE):", gs.best_score_)

# Evaluate best estimator on test set
best_rf = gs.best_estimator_
best_preds, best_rmse, best_mae, best_r2 = evaluate(best_rf, X_test, y_test, "RandomForest (best)")

# show a small dataframe of predictions vs actual
df_compare = pd.DataFrame({
    "actual": y_test.values,
    "predicted": best_preds,
    "residual": y_test.values - best_preds
})
print("\nSample predictions (first 10 rows):")
print(df_compare.head(10).to_string(index=False))
