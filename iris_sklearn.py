import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
clf = neighbors.KNeighborsClassifier(n_neighbors=15,weights='uniform').fit(x,y)
h = 0.2 
x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,z,cmap=cmap_light)
plt.scatter(x[:,0],x[:,1],c=y,cmap=cmap_bold,marker='o')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()
