import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load the Dataset ---

# Load dataset
df = pd.read_csv('G:\importent doc\private!\logistic regression\StudentPerformance.csv')

# Step 3: Explore the dataset
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:\n", df.head(10))

# Step 4: Data cleaning
# Convert categorical features to numeric using get_dummies (One-Hot Encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 5: Separate features and target
X = df.drop('Performance index', axis=1)
y = df['Performance index']

print("Dataset loaded successfully!")
print(f"Total number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print("-" * 30)


# --- 2. Split the Data ---
# Split the data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size (samples): {X_train.shape[0]}")
print(f"Testing set size (samples): {X_test.shape[0]}")
print("-" * 30)


# --- 3. Train the Logistic Regression Model ---
# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model using the training data
print("Training the Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 30)


# --- 4. Make Predictions ---
# Predict the class labels for the test set
y_pred = model.predict(X_test)
# You can also get probability estimates for each class:
y_proba = model.predict_proba(X_test)


# --- 5. Evaluate the Model ---
# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

print(f"probabilty Score: {y_proba}")