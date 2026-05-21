from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd

def getScore(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test) * 100

digits = load_digits()
df = pd.DataFrame(digits.data)
print(df.head(10))

df['target'] = digits.target
X = df.drop(['target'], axis='columns')
y = df['target']

# print(X.head())
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Rfclas = RandomForestClassifier()
logReg = LogisticRegression()
svm = SVC()

# score = getScore(model1, X_train, X_test, y_train, y_test)
# print(score)

"""How k-folds / stratified K-fold consider"""
# kf = KFold(n_splits=3) 
# for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
#    print(train_index, test_index)

# score_rf = []
# score_lreg = []
# score_svm = []

# fold = StratifiedKFold(n_splits=3)
# for train_i, test_i in kf.split(digits.data):
#     # split data 
#     X_train, X_test = digits.data[train_i], digits.data[test_i]
#     y_train, y_test = digits.target[train_i], digits.target[test_i]

#     # Train and store data
#     score_lreg.append(getScore(logReg, X_train, X_test, y_train, y_test))
#     score_rf.append(getScore(Rfclas, X_train, X_test, y_train, y_test))
#     score_svm.append(getScore(svm, X_train, X_test, y_train, y_test))

# print("SVM score : ", score_svm)
# print("random forest score : ", score_rf)
# print("logistic regression score : ", score_lreg)

"""cross validation function to get score [predifined/built-in]"""
score = cross_val_score(Rfclas, X, y, cv=3)
print(score)