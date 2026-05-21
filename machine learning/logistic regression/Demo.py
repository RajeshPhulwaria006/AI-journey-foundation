from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('G:\importent doc\private!\Machine Learning\logistic regression\Social_Network_Ads.csv')

print(df.head())
df['Gender'] = pd.get_dummies(df['Gender'], drop_first=True)
x = df[['Age', 'Gender', 'EstimatedSalary']]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Training on records : ", x_train.shape[0])
model = LogisticRegression()
model.fit(x_train, y_train)

print("Testing on records : ", x_test.shape[0])
y_pred = model.predict(x_test)
print("Prediction : ", y_pred)

accuracy = model.score(x_test, y_test)
print(f"Accuracy : {accuracy*100}%")

# plt.scatter(x_test, y_test, marker='+', color='red')
# plt.xlabel('Age')
# plt.ylabel('Purchased')
# plt.show()