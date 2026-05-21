from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import seaborn as sb 
import matplotlib.pyplot as plt

digits = load_digits()
df = pd.DataFrame(digits.data)
print(df.head())

X = digits.data
y = digits.target

# spli and train data and then Load model 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(
    kernel='rbf',     # radial basis function kernel
    gamma=0.001,
    C=100
)
model.fit(X_train, y_train)

# plt.matshow(digits.image[1])
# plt.show()

# Prediction and scores
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy : ", acc * 100)

cm = confusion_matrix(y_test, pred)
print("confusion matrix : \n", cm)

print("Classification report : \n", classification_report(y_test, pred))

""" visualizing correction and faults in model """
# plt.figure(figsize=(8,4))
# sb.heatmap(cm, annot=True)
# plt.xlabel('predicted')
# plt.ylabel('Actual')
# plt.show()
plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {pred[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()