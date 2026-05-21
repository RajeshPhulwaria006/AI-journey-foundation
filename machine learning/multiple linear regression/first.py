from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

X, y = make_regression(n_samples=100, n_features=2, n_targets=1, random_state=42)

df = pd.DataFrame({
    'X1': X[:, 0],
    'X2': X[:, 1],
    'target': y
})

# print(df['X1'].shape)
# print(df['X2'].shape)
# print(df['target'].shape)
# print(df.head())

# plt.scatter(x=df[['X1']], y=df['target'], color='red', marker='X')
# plt.show()

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(df[['X1', 'X2']], df['target'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("training completed.")

y_pred = model.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))
