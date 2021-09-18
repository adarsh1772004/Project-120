import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("income.csv")
X = df[["age", "hours-per-week", "education-num", "capital-gain", "capital-loss"]]
Y=df['income']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
model=GaussianNB()
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(Y_test, y_pred)
print(accuracy)
