from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd

digits = load_digits()
df = pd.DataFrame(digits.data)
print(df.head(10))

df['target'] = digits.target
X = df.drop(['target'], axis='columns')
y = df['target']

# print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy * 100)

cm = confusion_matrix(y_test, y_pred)
print("matrix : \n", cm)

report = classification_report(y_test, y_pred)
print("Report : \n", report)

import seaborn as sb
plt.figure(figsize=(8,4))
sb.heatmap(cm, annot=True)
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()

import pickle 

with open("digit_Trained_model.pkl", "wb")as file:
    pickle.dump(model, file)
    