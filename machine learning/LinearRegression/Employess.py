from word2number import w2n
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


model = LinearRegression()

df = pd.read_csv('G:\importent doc\private!\Machine Learning\LinearRegression\hiring.csv')
print(df.isna().sum())
print("Data before cleaning : \n", df)

# filling null values at test score
test_score = int(df['test_score(out of 10)'].mean())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(test_score)

def convert_exp(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    # If it's a plain word (two, five, seven...)
    try:
        return w2n.word_to_num(x)
    except:
        return np.nan

# apply word to number function 
df['experience'] = df['experience'].apply(convert_exp)
# convert null values to - 0
df['experience'] = df.experience.fillna(0)

print("Data after cleaning : \n", df)

print("Training the model...")
model.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

newEmps = pd.DataFrame({
    "experience": [2, 12],
    "test_score(out of 10)": [9, 10],
    "interview_score(out of 10)": [6, 10]
})  

import pickle 
with open('saved_model', 'wb') as f:
    pickle.dump(model, f)
    print("model saved.")

print("new employee salary : \n")
pred = model.predict(newEmps) # in decimals[XYZ.AB]

# Round to nearest 1000
rounded = (np.round(pred / 1000) * 1000).astype(int)

print(rounded)