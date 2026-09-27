#INPUT OUTPUT SEPARATION

x=df.drop(['id','stroke'],axis=1)
print(x.ndim)
y=df['stroke']
x
