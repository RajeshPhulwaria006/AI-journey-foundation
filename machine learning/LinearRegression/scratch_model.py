import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

data = {
    'cgpa': [5.5, 5.7, 4.5, 8.9, 8.6, 7.9, 6.9, 9.6, 4.5, 5.1],
    'package': [4, 5, 3, 12, 8, 5, 5, 9, 5, 5.5]
}
print("Loading dataset...")
df = pd.DataFrame(data)

class linearRegression:
    """
    Linear Regression 
    ---------

    This class implements linear regression from scratch without relying
    on high-level ML libraries. It is intended for educational purposes
    and small-scale experimentation, not production use.

    Parameters
    ----------
    lr [learning rate] : float / by default: 0.01
        Learning rate used for gradient descent.
    epochs [number of iterations] : int / by default: 1000
        Number of optimization iterations.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
        Learned weights after training.
    bias : float
        Bias term of the model.

    Methods
    -------
    fit(X, y)
        Train the model using feature matrix X and target vector y.
    predict(X)
        Predict target values for given input features.

    Examples
    --------
    >>> model = LinearRegression(lr=0.01, epochs=1000)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """
    def __init__(self, lr = 0.01, epochs = 1000):
        self.lr = lr
        self.epochs = epochs + 1
        
        # initialization of starting slope and intercept
        self.m = 0.0
        self.b = 0.0
        self.X_mean = None
        self.X_std = None
    
    # method to train model
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        n = len(X)

        # normalize feature:
        self.X_mean = X.mean()
        self.X_std = X.std()
        X = (X - self.X_mean) / self.X_std

        for i in range(self.epochs):
            # forward pass
            y_pred = self.m * X + self.b
            
            # compute gradients
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)

            # update parameters
            self.m -= self.lr * dm
            self.b -= self.lr * db

            if i % 100 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"    Epochs : {i}, Loss: {loss:.4f}")
                
    # method to make predictions
    def predict(self, X):
        if self.m is None or self.b is None:
            raise Exception("Model not trained. Call fit() first.")

        X = np.array(X)
        # Use TRAINING statistics (never recompute)
        X = float((X - self.X_mean) / self.X_std)
        return self.m * X + self.b


X = np.array(df['cgpa'].values)       # shape: (n,)
y = np.array(df['package'].values)    # shape: (m,)

X = X.reshape(10, 1)
y = y.reshape(10, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = linearRegression()
model.fit(X_train, y_train)

print(f"Prediction for 5.1 CGPA model: {model.predict(5.1):.2f}\n")
print(f"\tM : {model.m}, B : {model.b}")

# visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 2))
plt.scatter(df['cgpa'], df['package'], color='blue')
plt.title("Dataset overview")
plt.xlabel("CGPA")
plt.ylabel("Package (LPA)")
plt.show()