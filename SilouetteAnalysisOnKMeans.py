from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

iris =load_iris()
x = iris.data[:,1:3]
y = iris.target

range_n_clusters = [2,3,4]
for n_clusters in range_n_clusters:
    fig,(ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18,7)

    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0,len(x)+(n_clusters+1)*10])

    clusterer = KMeans(n_clusters = n_clusters)
    cluster_labels = clusterer.fit_predict(x)

    silhouette_avg = silhouette_score(x,cluster_labels)
    print("n_clusters = ",n_clusters,"The average silhouette_score is : ",silhouette_avg)

    sample_silhouette_values = silhouette_samples(x,cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values =sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower,y_upper),0,ith_cluster_silhouette_values,facecolor = color,edgecolor = color,alpha =0.7)

        ax1.text(-0.05,y_lower+0.5*size_cluster_i,str(i))
        y_lower = y_upper+10

    ax1.set_title("THe silouette plot for the various clusters.")
    ax1.set_xlabel("The silouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x = silhouette_avg,color = "red",linestyle =" - -")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])

    colors = cm.spectral(cluster_labels.astype(float)/n_clusters)
    ax2.scatter(x[:,0],x[:,1],marker=".",s=30,lw=0,alpha=0.7,c=colors)

    centers =clusterer.cluster_centers_
    ax2.scatter(centers[:,0],centers[:,1],marker='o',c="white",alpha=1,s=200)
    for i,c in enumerate(centers):
        ax2.scatter(c[0],c[1],marker='$%d$'%i,alpha=1,s=50)

    ax2.set_title('The visualization of the clustered data.')
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel('Feature space for the 2nd feature')

    plt.suptitle(("silouette analysis for KMeans clustering on sample data  with n_clusters=%d" %n_clusters),fontsize =14,fontweight ='bold')
    plt.show()
