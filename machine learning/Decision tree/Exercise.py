from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("G:\\importent doc\\private!\\Machine Learning\\Decision tree\\TitanicDataset_cleaned.csv")
columns = df.columns
print(columns)

new_df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
print(new_df.columns)

inputs = new_df.drop(['Survived'], axis='columns')
target = new_df['Survived']

# print(f"inputs : \n {inputs}\n target : \n{target}")

le_Sex = LabelEncoder()
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
df_encoded = inputs.drop(['Sex'], axis='columns')

print(df_encoded)

X = df_encoded # features
Y = target 

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Load model for predictions [Tree classifier]
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
pred = model.predict(X_test)

print(f"\nTesting set :\n{X_test}\n")
print(f"\nPrediction of survival rate : {pred}")

# print(model.predict([[3, 30, 7.2292, 0]]))
# probability = model.predict_proba([[2, 45, 8, 0]])
# print(probability)

"""
fetching the data and getting confusion matrix of how much people died and how much not with the accuracy of predicted data
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb

# print(y_test)
cm = confusion_matrix(y_test, pred)
# print(y_test.shape)
plt.figure(figsize=(10,5))
sb.heatmap(cm, annot=True)
plt.xlabel('predicted')
plt.ylabel('Truth')
plt.show()

# Accuracy :
print(f"Accuracy : {accuracy * 100}%")