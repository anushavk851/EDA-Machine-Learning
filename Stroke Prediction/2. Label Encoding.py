#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
obj=df.select_dtypes(include=object).columns.to_list()
for i in obj:
  df[i]=le.fit_transform(df[i])
df
