import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.forest import RandomForestRegressor
import matplotlib.pyplot as plt
#create a random training set
rng = np.random.RandomState(1)
x = np.sort(5*rng.rand(80,1),axis = 0)
y = np.sin(x).ravel()
y[: : 5] +=3*(0.5 -rng.rand(16))

#create a max deepth 4 model , n_estimator is count 
clf_1 = RandomForestRegressor(n_estimators=400,max_depth=4)
clf_2 = DecisionTreeRegressor(max_depth=4)
clf_1.fit(x,y)
clf_2.fit(x,y)

x_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]
y_1 = clf_1.predict(x_test)
y_2 = clf_2.predict(x_test)

plt.figure()
plt.scatter(x,y,c="k",label ="data")
plt.plot(x_test,y_1,c="g",label="max_depth=4",linewidth=2)
plt.plot(x_test,y_2,c="r",label="max_depth=4",linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
