# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/home/desktop-rajesh2006/Projects/Research-oriented lab/datasets/StudentsPerformance.csv')

# Explore the dataset
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:\n", df.head(10))

# Data cleaning
# Convert categorical features to numeric using get_dummies (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df_encoded.drop('math score', axis=1)
y = df_encoded['math score']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize predictions
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Math Scores")
plt.ylabel("Predicted Math Scores")
plt.title("Actual vs Predicted Math Scores")

# Add Average Line 
avg_score = np.mean(y_test)     # Calculate average of actual test scores
plt.axhline(y=avg_score, color='red', linestyle='--', label=f'Average Score = {avg_score:.2f}')

# Optional: Add diagonal line (perfect prediction reference)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='green', linestyle='--', label='Perfect Prediction')
plt.legend()

# Predict a new student's math score (example)
sample_student = X.iloc[[0]]
predicted_score = model.predict(sample_student)
print("\nPredicted math score for sample student:", round(predicted_score[0], 2))

# coefficients for model
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df.sort_values(by='Coefficient', ascending=False))

# show data on graph
plt.show()