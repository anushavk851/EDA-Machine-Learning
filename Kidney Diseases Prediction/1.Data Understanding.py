import pandas as pd
import numpy as np
df=pd.read_csv('/content/kidney_disease.csv')
df

#DATA UNDERSTANDING

#first 4 rows
print(df.head(4))
#last 3 rows
print(df.tail(4))
#dimension
print(df.ndim)
#number of rows and columns
print(df.shape)
#statistical summary
print(df.describe())
#unique values
print(df.nunique())
#datatype
print(df.dtypes)
#all column names
print(df.columns)
#numerical column names
print(df.select_dtypes(exclude=object).columns)
#non-numerical column'
print(df.select_dtypes(include=object).columns)
#target column
print(df['classification'].value_counts())








