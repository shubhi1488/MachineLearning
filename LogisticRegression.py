import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import io
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn import preprocessing
plt.rc("font", size = 14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set(style="white")
sns.set(style="whitegrid", color_codes = True)

# Built in colab with local data upload

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
  # Built in colab with local data upload

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  # Explore data

df = pd.read_csv(io.StringIO(uploaded['Iris.csv'].decode('utf-8')))
df = df[['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']] 	

df.head()

X=df.iloc[:,1:5]
Y=df.iloc[:,5]
print(X)
print(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
print(x_train)
print(x_test)

log=LogisticRegression()
log.fit(x_train,y_train)

y_pred=log.predict(x_test)
print(y_pred)
print(y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(log,x_test,y_test)
confusion_matrix(y_test,y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
