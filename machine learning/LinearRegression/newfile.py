from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

data = {
    "cgpa": [5.8,6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,8.0,8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,9.0,9.1,9.2,9.3,9.4,9.5],
    "package": [3.2,3.5,3.6,3.8,3.9,4.1,4.3,4.5,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.3,6.5,6.8,7.0,7.3,7.6,7.9,8.2,8.5,8.9,9.2,9.6,10.0,10.5,11.0,11.6,12.2,12.8,13.5,14.2,15.0]
}

df = pd.DataFrame(data)
print(df.head(10))

X = df[['cgpa']]
y = df['package']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute error : ", mean_absolute_error(y_test, y_pred))
print("Mean Squared error : ", mean_squared_error(y_test, y_pred))
print("R2 score : ", r2_score(y_test, y_pred)*100, "%")

plt.scatter(X, y, color='blue', marker='+')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('cgpa')
plt.ylabel('package')
plt.show()