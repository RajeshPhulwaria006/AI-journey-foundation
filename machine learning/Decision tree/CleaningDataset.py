import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('G:\importent doc\private!\Machine Learning\Decision tree\TitanicDataset.csv')

# 1. Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# 2. Replace invalid placeholders with NaN
df.replace(['?', ' ', ''], np.nan, inplace=True)

# 3. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 4. Convert numeric-looking columns to numeric
for col in df.columns:
    # Try converting to numeric
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# 5. Handle missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        # Correct way: assign back, no inplace
        df[col] = df[col].fillna(df[col].median())
    else:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

# Show cleaned data preview
print(df.head())

# Save cleaned dataset (optional)
df.to_csv('TitanicDataset_cleaned.csv', index=False)
print("\nCleaned dataset saved as TitanicDataset_cleaned.csv")
