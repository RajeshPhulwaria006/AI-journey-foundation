from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('G:\importent doc\private!\Machine Learning\Decision tree\EmployessData.csv')

df['salary_more_than_30k'] = (df['salary'] > 20000).astype(int)

target = df['salary_more_than_30k']
df = df.drop('salary_more_than_30k', axis='columns')

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

# print(df)
# print(target)

df['company_n'] = le_company.fit_transform(df['company'])
df['job_n'] = le_job.fit_transform(df['job'])
df['degree_n'] = le_degree.fit_transform(df['degree'])

df_encoded = df.drop(df[['company', 'job', 'degree', 'salary']], axis='columns')
# print(df)
print(df_encoded)

x = df_encoded
y = target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f" Training dataset : \n {X_train}\n Testing dataset : \n {X_test}")

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

accuracy = (model.score(X_test, y_test)) * 100
prediction = model.predict(X_test)

print(df)
print("\nPrediction [if the salary is more than 30k -> 1 , otherwise its -> 0] : ", prediction)
print(f"\nAccuracy : {accuracy}%")