from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
print(dir(iris))

# print(iris['feature_names'], iris['target_names'])

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
# target = pd.DataFrame(iris.data, columns=iris.target_names)
# print(target.head())
df['Target'] = iris.target
# df['Target_names'] = iris.target_names
print(df)
print(iris.target_names)

setosa = df[df.Target == 0]
versicolor = df[df.Target == 1]
virginica = df[df.Target == 2]

# print(f"setosa : \n{setosa}")
# print(setosa.shape[0])

# print(f"versicolor : \n{versicolor}")
# print(versicolor.shape[0])

# print(f"virginica : \n{virginica}")
# print(virginica.shape[0])
df['Flowers'] = df['Target'].apply(lambda x : iris.target_names[x])

"""Split and training the model : SVC"""
X = df.drop(['Target', 'Flowers'], axis='columns')
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""Training the model"""
model = SVC(C=10, kernel='linear')
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"accuracy : {accuracy * 100:.2f}%")
predict = model.predict(X_test)
print("Prediction : ", predict)

"""classification on graph[visualization]"""
plt.scatter(setosa['sepal length (cm)'], setosa['sepal width (cm)'], color='green', marker='+', label='setosa')
plt.scatter(versicolor['sepal length (cm)'], versicolor['sepal width (cm)'], color='blue', marker='.', label='versicolor')
plt.scatter(virginica['sepal length (cm)'], virginica['sepal width (cm)'], color='red', marker='*', label='virginica')
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.show()