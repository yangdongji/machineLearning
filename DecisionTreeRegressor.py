import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

#create a random dataset
rng = np.random.RandomState(1)
x = np.sort(5*rng.rand(80,1),axis=0)
y = np.sin(x).ravel()
y[: : 5] +=3*(0.5-rng.rand(16)) #add some noise to the data

#train three different deepth of model
clf_1 = DecisionTreeRegressor(max_depth = 2)
clf_2 = DecisionTreeRegressor(max_depth = 4)
clf_3 = DecisionTreeRegressor(max_depth = 6)
clf_1.fit(x,y)
clf_2.fit(x,y)
clf_3.fit(x,y)


x_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]
y_1 = clf_1.predict(x_test)
y_2 = clf_2.predict(x_test)
y_3 = clf_3.predict(x_test)

plt.figure()
plt.scatter(x,y,c='k',label='data')
plt.plot(x_test,y_1,c='g',label='max_deepth=2',linewidth=2)
plt.plot(x_test,y_2,c='r',label='max_deepth=4',linewidth=2)
plt.plot(x_test,y_3,c='b',label='max_deepth=6',linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title('Decision Tree Regressor')
plt.legend()
plt.show()
