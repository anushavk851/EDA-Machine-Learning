#DATA CLEANING

#1.REMOVING ID COLUMN
df=df.drop('id',axis=1)

#2.Null value handling
print(df.isnull().sum())
num = df.select_dtypes(include='number').columns
obj= df.drop('classification',axis=1).select_dtypes(include='object').columns
for i in num:
  df[i]=df[i].fillna(df[i].median())
for i in obj:
   df[i]=df[i].fillna(df[i].mode()[0])
print(df.isnull().sum())

#3.target column value having type-error
print(df['classification'].value_counts())
df['classification'] = df['classification'].replace('ckd\t', 'ckd')
print(df['classification'].value_counts())

#4.duplicate values
print(df.duplicated().sum())

#5.OUTLIERS

#1.Boxplot for numerical columns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
df.drop('classification', axis=1).boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot of Numerical Features")
plt.show()

#2.Find the number of outliers
num= df.drop(['classification'], axis=1).select_dtypes(include='number').columns
for i in num:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[i] < lower) | (df[i] > upper)]
    print(i, "outliers:", len(outliers))
  
#3. treating outliers
for i in num:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[i] = df[i].clip(lower, upper)
  
#4. again checking Boxplot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
df.drop('classification', axis=1).boxplot()
plt.xticks(rotation=45)
plt.title("Box Plot of Numerical Features")
plt.show()
