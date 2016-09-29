import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
for pairidx,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    x = iris.data[:,pair]
    y = iris.target

    idx = np.arange(x.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]

    mean = x.mean(axis = 0)
    std = x.std(axis = 0)
    x = (x- mean)/std

    clf = DecisionTreeClassifier().fit(x,y)

    plt.subplot(2,3,pairidx + 1)
    x_min,x_max = x[: , 0].min() - 1,x[: , 0].max() +1 
    y_min,y_max = x[: , 0].min() - 1,x[: , 0].max() +1 
    xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    cs = plt.contourf(xx,yy,z ,cmap = plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis('tight')

    for i ,color in [(0,'b'),(1,'r'),(2,'y')]:
        idx =np.where(y==i)
        plt.scatter(x[idx,0],x[idx,1],c=color,label=iris.target_names[i],cmap=plt.cm.Paired)
        plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plt.show()
