import pandas as pd
df=pd.read_csv('/content/Salary_Dataset_DataScienceLovers.csv')
df

#DATA UNDERSTANDING

#1.first 3 rows
print(df.head(3))
#2.last 3 rows
print(df.tail(3))
#3.number of rows and columns
print(df.shape)
#4.number of elements
print(df.size)
#5. dimension
print(df.ndim)
#6. statistical summary
print(df.describe())
#7. column names
print(df.columns)
#8.Numerical column names
print(df.select_dtypes(exclude=object).columns)
#9.Object column names
print(df.select_dtypes(include=object).columns)
#10. unique values
print(df.nunique())
#11. datatype of each column'
print(df.dtypes)
#12. Target column maximum  and minimum value
print("Maximum salary:",df['Salary'].max())
print("minimum salary:",df['Salary'].min())


#DATA CLEANING

#1. null values
print(df.isnull().sum())
df['Company Name']=df['Company Name'].fillna(df['Company Name']).mode()[0]
print(df.isnull().sum())

#2. Duplicate values
print(df.duplicated().sum())
df[df.duplicated()]
df=df.drop_duplicates()
print(df.duplicated().sum())
