import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons,make_circles
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
h= 0.2
names = ["Nearest Neighbors","Decision Tree","Naive Bayes"]
classifiers = [
    KNeighborsClassifier(3), DecisionTreeClassifier(max_depth=5), GaussianNB()  ]
x , y =make_classification(n_features=2,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=1 )
rng = np.random.RandomState(2)
x +=2*rng.uniform(size = x.shape)
linearly_separable = (x,y)
datasets=[make_moons(noise=0.3,random_state=0),make_circles(noise=0.2,factor=0.5,random_state=1),linearly_separable]
figure = plt.figure(figsize=(18,6))
i =1 
for ds in datasets:
    x, y =ds
    x = StandardScaler().fit_transform(x)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =.3)
    x_min,x_max = x[:,0].min()- .5,x[:,0].max()+ .5
    y_min,y_max = x[:,0].min()- .5,x[:,0].max()+ .5
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000','#0000FF'])
    ax = plt.subplot(len(datasets),len(classifiers)+1,i)
    ax.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_bright)
    ax.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_bright,alpha = 0.6)

    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i+=1

    for name,clf in zip(names,classifiers):
       ax = plt.subplot(len(datasets),len(classifiers)+1,i)
       clf.fit(x_train,y_train)
       score = clf.score(x_test,y_test)

       z = clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[: , 1]
       z = z.reshape(xx.shape)
       ax.contourf(xx,yy,z,cmap=cm,alpha=.8)

       ax.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_bright)
       ax.scatter(x_test[: , 0],x_test[:, 1],c = y_test,cmap=cm_bright,alpha=0.6)
       ax.set_xlim(xx.min(),xx.max())
       ax.set_ylim(yy.min(),yy.max())
       ax.set_xticks(())
       ax.set_yticks(())
       ax.set_title(name)
       ax.text(xx.max()- .3,yy.min() + .3,('%.3f' %score).lstrip('0'),size=15,horizontalalignment = 'right')
       i+=1
figure.subplots_adjust(left= .02,right=.98)
plt.show()
