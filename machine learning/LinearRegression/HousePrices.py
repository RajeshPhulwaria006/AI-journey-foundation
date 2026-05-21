import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample dataset
data = {
    'Area': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 15, 20, 8, 12],
    'Price': [400000, 500000, 600000, 650000, 700000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]   # Features
y = df['Price']                        # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example: Predict price for 2800 sq.ft, 4 bedrooms, 10 years old
new_house = pd.DataFrame([[2800, 4, 10]], columns=['Area', 'Bedrooms', 'Age'])
predicted_price = model.predict(new_house)
print("Predicted Price for the new house:", predicted_price[0])
