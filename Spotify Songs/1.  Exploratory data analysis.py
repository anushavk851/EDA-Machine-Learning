import pandas as pd
df=pd.read_csv('/content/spotify-2023.csv',encoding='latin1')
df

#DATA UNDERSTANDING

#1.first 3 rows
print(df.head(3))
#2. last 3 rows
print(df.tail(3))
#3. dimension
print(df.ndim)
#4. number of rows and columns
print(df.shape)
#5. column names
print(df.columns)
#6. Categorical columns
print(df.select_dtypes(include=object).columns)
#7. Numerical columns
print(df.select_dtypes(exclude=object).columns)
#8. statistical summary
print(df.describe())
#9. total information
print(df.info())
#10. Unique values
print(df.nunique())
#11.Target column value counts
print(df['mode'].value_counts())


#DATA CLEANING

#1.Duplicate values
print(df.duplicated().sum())

#2.Null Values
print(df.isnull().sum())
df['in_shazam_charts'].value_counts()
df['in_shazam_charts']=df['in_shazam_charts'].fillna(df['in_shazam_charts']).mode()[0]
df['key']=df['key'].fillna(df['key']).mode()[0]
print(df.isnull().sum())

