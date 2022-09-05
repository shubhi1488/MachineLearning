#Linear regression using numpy and matplotlib python module
import numpy as np
x=np.array([171,151,124,134,156])
y=np.array([80,60,45,50,65])
n=len(x)
meanx=sum(x)/n
meany=sum(y)/n
num=0
deno=0
for i in range(n):
  num+=(x[i]-meanx)*(y[i]-meany)
  deno+=(x[i]-meanx)**2
  m=num/deno
  print(m)

  
c=meany-(m*meanx)
print(c)

F=m*130+c
print(F)

import matplotlib.pyplot as plt
plt.scatter(x,y,color='orange')
plt.plot(x,y)
plt.show()

max_x=np.max(x)+100
min_x=np.min(x)-100
x=np.linspace(min_x,max_x,100)
y=[]
for i in range(100):
  y.append(c+m**[i])
  plt.plot(x,y)
