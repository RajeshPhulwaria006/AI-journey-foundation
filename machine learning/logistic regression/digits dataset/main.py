from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sb

digit = load_digits()

# print(digit.data[0])
# print(dir(digit))

# plt.gray()
# for i in range(10):
#     plt.matshow(digit.images[i])
#     plt.show()

""" split the data into target and feature"""
X = digit.data
Y = digit.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

""" training the model """
model = LogisticRegression()
model.fit(X_train, Y_train)

# guess = int(input("Enter number of target : "))

# plt.matshow(digit.images[guess])
# plt.show()

# predicted = model.predict([digit.data[guess]])
# print("\nThe prediction : ", predicted)
# print("\nThe number was : ", digit.target[guess])

# accuracy = model.score(X_test, Y_test)
# print(accuracy*100, "%")

y_pred = model.predict(X_test)

""" get correction matrix """
cm = confusion_matrix(Y_test, y_pred)
print(cm)

""" visualizing correction and faults in model """
plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()
