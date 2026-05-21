import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("G:\importent doc\private!\Machine Learning\stardardization\Social_Network_Ads.csv")

# Encode
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

# Use ONLY Age for sigmoid curve
X = df[['Age']]
y = df['Purchased']

# Scale Age
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train model
model = LogisticRegression(solver='liblinear')
model.fit(X_scaled, y)

# Generate smooth curve
age_range = np.linspace(df['Age'].min(), df['Age'].max(), 300).reshape(-1, 1)
age_scaled = sc.transform(age_range)
prob = model.predict_proba(age_scaled)[:, 1]

# Plot
plt.figure(figsize=(8,5))
plt.scatter(df['Age'], y, alpha=0.4, label='Actual')
plt.plot(age_range, prob, linewidth=3, label='Logistic Curve')
plt.xlabel("Age")
plt.ylabel("Probability of Purchase")
plt.title("Logistic Regression - Sigmoid Curve (Age vs Purchase Probability)")
plt.legend()
plt.show()
