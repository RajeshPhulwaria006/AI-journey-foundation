import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y, lr=0.01, iterations=2000):
    m, b = 0.0, 0.0
    n = len(x)

    cost_history = []
    params_history = []

    for i in range(iterations):
        y_pred = m * x + b
        error = y_pred - y

        # vectorized gradients
        dm = (2/n) * np.sum(error * x)
        db = (2/n) * np.sum(error)

        m -= lr * dm
        b -= lr * db

        cost = np.mean(error ** 2)
        cost_history.append(cost)
        params_history.append((m, b))

    return m, b, cost_history, params_history


def plot_results(x, y, m, b, cost_history):
    # final fitted line
    plt.figure(figsize=(7,5))
    plt.scatter(x, y, label="data", alpha=0.7)
    plt.plot(x, m*x + b, linewidth=2, label="fitted line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    # cost convergence
    plt.figure(figsize=(7,4))
    plt.plot(cost_history)
    plt.yscale("log")
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost Function Convergence")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # sample dataset
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 7, 14, 15, 17], dtype=float)

    m, b, cost_history, params_history = gradient_descent(x, y, lr=0.01, iterations=5000)

    print("Final slope (m):", m)
    print("Final intercept (b):", b)
    print("Final cost:", cost_history[-1])

    plot_results(x, y, m, b, cost_history)
