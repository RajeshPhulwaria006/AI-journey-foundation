import numpy as np
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

# Generate dataset
X, y = make_regression(
    n_samples=100,      # number of data points
    n_features=1,       # number of input features (linear regression → usually 1)
    n_informative=1,
    noise=10,           # noise added to target
    random_state=42     # for reproducibility
)

# Convert to DataFrame (optional but useful)
df = pd.DataFrame(X, columns=['Feature'])
df['Target'] = y

print(df.head())
df.info()

def gradient_descent(x, y, iterats=1000, learningRate=0.01):
    curr_m = curr_b = 0.0
    n = len(x)

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64)   

    # standardization / feature scaling - to reduce scale of data
    mean_x = X.mean()
    std_x = X.std()
    x = (x - mean_x) / std_x

    for i in range(iterats):
        y_pred = curr_m * x + curr_b
        
        # vectorization
        error = y_pred - y
        
        # slope of partial derivatives
        dm = (2/n) * np.sum(x * error)
        db = (2/n) * np.sum(error)
        
        # direction to downward in gradient
        curr_m = curr_m - learningRate * dm
        curr_b = curr_b - learningRate * db
        
        if i % 200 == 0:
            mse = np.mean(error**2) #cost function
            rmse = np.sqrt(mse)
            print(f"iter {i}:\n mse = {mse:.5f}\n rmse = {rmse:.5f} \n")
            
    m = curr_m / std_x
    b = curr_b - (curr_m * mean_x / std_x)    
    return m, b;

m, b = gradient_descent(X.flatten(), y)

print(f"final m : {m :.5f}")
print(f"final b : {b :.5f}")

# import matplotlib.pyplot as plt

# plt.scatter(X, y)
# plt.show()

"""Check how much our GD is correct"""
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print("compared to: ")
print(model.coef_, model.intercept_)

import matplotlib.pyplot as plt

y_pred_GD = m * X + b
y_pred_model = model.coef_[0] * X + model.intercept_

plt.scatter(X, y)
plt.plot(X, y_pred_GD, label="GD")
plt.plot(X, y_pred_model, label="sklearn")
plt.legend()
plt.show()
