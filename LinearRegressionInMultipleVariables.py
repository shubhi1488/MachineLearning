# Import Libary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
df=pd.read_csv('/headbrain.csv')

df.head()

X=df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values

np.corrcoef(X, Y)

plt.scatter(X, Y, c='green', label='Data points')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

mean_x = np.mean(X)
mean_y = np.mean(Y)

n = len(X)

numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 5.0)
# max_x = np.max(X) + 100
# min_x = np.min(X) - 100

y = b0 + b1 * X

# Ploting Line
plt.plot(X, y, color='blue', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='green', label='Scatter data')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

rmse = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
    
rmse = np.sqrt(rmse/n)
print("Root Mean Square Error is",rmse)

ss_tot = 0
ss_res = 0
for i in range(n):
    y_pred = b0 + b1 * X[i]
    ss_tot += (Y[i] - mean_y) ** 2
    ss_res += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score",r2)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data=pd.read_csv("/headbrain.csv")
data.head()

X=df['Head Size(cm^3)'].values
Y = df['Brain Weight(grams)'].values

m=len(X)
X=X.reshape((m,1))
reg=LinearRegression()
reg.fit(X,Y)

print('m',reg.coef_)
print('c',reg.intercept_)

y_predict=reg.predict(X)
print(y_predict)

rmse=np.sqrt(mean_squared_error(Y,y_predict))
r2=reg.score(X,Y)
print(rmse)
print(r2)
