import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

""" ---When to use standardization [specific algorithms]
1. K-nn / k-nearest neighbours
2. k-mean
3. Principal component analysis / PCA
4. Artificial neural network . ANN
5. Gradient descent
"""

df = pd.read_csv("G:\importent doc\private!\Machine Learning\standardization\Social_Network_Ads.csv")
df = df.iloc[:, 2:]

print(df.head(10))

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1), df['Purchased'], test_size=0.2, random_state=42
)
# print(X_train, X_test)

scaler = StandardScaler()

# fit the scaler to the train set, learn parameters
scaler.fit(X_train)

# transform train and test sets [it will returns numpy array not dataframe etc]
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\tNumpy array :\n")
print(X_train_scaled)

# get scaler values in dataframe form 
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\n \tdataframe :\n")
print(X_train_scaled)

"""
-> Now main difference is the mean and standard diviation of scaled set would be :
    mean = 0
    std = 1
"""

print(np.round(X_train.describe()))
# print(X_train_scaled.describe())
print(np.round(X_train_scaled.describe(), 1))

# visualize the difference
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Before scaling
ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title('Before scaling')
ax1.set_xlabel('Age')
ax1.set_ylabel('Estimated Salary')

# After scaling
ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'])
ax2.set_title('After scaling')
ax2.set_xlabel('Age')
ax2.set_ylabel('Estimated Salary')

# ---probability density function---
# ax1.set_title("Before scaling")
# sns.kdeplot(X_train['Age'], ax=ax1)
# sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

# ax2.set_title("After scaling")
# sns.kdeplot(X_train_scaled['Age'], ax=ax2)
# sns.kdeplot(X_train_scaled['EstimatedSalary'], ax=ax2)

plt.tight_layout()
plt.show()