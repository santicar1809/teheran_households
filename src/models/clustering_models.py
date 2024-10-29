import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import sys
from sklearn.metrics import silhouette_score
import joblib

def clustering_data(data):
    eda_path = './files/modeling_output/figures/'
    output_path = './files/modeling_output/model_fit/'
    # Kmeans
    # Elbow method
    features=data[['area','price(usd)','room']]
    wcss = []
    for i in range(1, 20):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    aa=1
    fige, axe = plt.subplots(figsize=(10, 6))
    axe.plot(range(1, 20), wcss, marker='o', linestyle='--')
    axe.set_title('Método del Codo')
    axe.set_xlabel('Número de Clusters')
    axe.set_ylabel('WCSS')
    fige.savefig(eda_path+'elbow_method.png')
    
    # Aparentemente con 3 clusters seria suficiente clusterizar las casas de teheran

    # Aplicar K-means con 3 clusters a la muestra de datos
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    joblib.dump(kmeans,output_path +f'elbow_kmeans.joblib')
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['area','price(usd)','room'])
    for i in centroids.columns:
        centroids[i]=centroids[i].astype('float')

    # Añade una columna con el número de clúster
    data['cluster'] = kmeans.labels_.astype(str)
    centroids['cluster'] = ['0 centroid', '1 centroid', '2 centroid']

    fig, ax = plt.subplots(3,1,figsize=(10, 20))
    sns.scatterplot(x='area', y='price(usd)', hue='cluster', data=data,ax=ax[0])
    ax[0].scatter(centroids['area'], centroids['price(usd)'], marker='*', s=200, c='red')
    ax[0].set_title('Area vs Price')

    sns.scatterplot(x='area', y='room', hue='cluster', data=data,ax=ax[1])
    ax[1].scatter(centroids['area'], centroids['room'], marker='*', s=200, c='red')
    ax[1].set_title('Area vs Rooms')

    sns.scatterplot(x='room', y='price(usd)', hue='cluster', data=data,ax=ax[2])
    ax[2].scatter(centroids['room'], centroids['price(usd)'], marker='*', s=200, c='red')
    ax[2].set_title('Rooms vs Price')

    plt.tight_layout(h_pad=15.0)

    fig.savefig(eda_path+'clusters_elbow.png')
    # Silhouette method
    sil_score = []
    cluster_list=list(np.arange(2,13))
    for i in cluster_list:
        kmeans = KMeans(n_clusters=i, random_state=42)
        preds=kmeans.fit_predict(features)
        score=silhouette_score(features,preds)
        sil_score.append(score)
    # Graphic
    figs, axs = plt.subplots(figsize=(10, 6))
    axs.plot(cluster_list, sil_score, marker='o', linestyle='--')
    axs.set_title('Silhouette method')
    axs.set_xlabel('Número de Clusters')
    axs.set_ylabel('Sil')
    figs.savefig(eda_path+'silhouette_method.png')

    # Aplicar K-means con 7 clusters a la muestra de datos
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit(features)
    joblib.dump(kmeans,output_path +f'silhouette_kmeans.joblib')
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['area','price(usd)','room'])
    for i in centroids.columns:
        centroids[i]=centroids[i].astype('float')

    # Añade una columna con el número de clúster
    data['cluster'] = kmeans.labels_.astype(str)
    centroids['cluster'] = ['0 centroid', '1 centroid', '2 centroid','3 centroid', '4 centroid', '5 centroid','6 centroid']

    figs1, axs1 = plt.subplots(3,1,figsize=(10, 20))
    sns.scatterplot(x='area', y='price(usd)', hue='cluster', data=data,ax=axs1[0])
    axs1[0].scatter(centroids['area'], centroids['price(usd)'], marker='*', s=200, c='red')
    axs1[0].set_title('Area vs Price')

    sns.scatterplot(x='area', y='room', hue='cluster', data=data,ax=axs1[1])
    axs1[1].scatter(centroids['area'], centroids['room'], marker='*', s=200, c='red')
    axs1[1].set_title('Area vs Rooms')

    sns.scatterplot(x='room', y='price(usd)', hue='cluster', data=data,ax=axs1[2])
    axs1[2].scatter(centroids['room'], centroids['price(usd)'], marker='*', s=200, c='red')
    axs1[2].set_title('Rooms vs Price')

    plt.tight_layout(h_pad=15.0)

    figs1.savefig(eda_path+'clusters_silhouette.png')

    ## DBSCAN
    knn = NearestNeighbors(n_neighbors = 7)
    model = knn.fit(features)
    distances, indices = knn.kneighbors(features)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    figknn, axknn = plt.subplots(figsize=(20, 10))
    axknn.grid()
    axknn.plot(distances)
    axknn.set_xlabel('Points Sorted by Distance')
    axknn.set_ylabel('7-NN Distance')
    axknn.set_title('K-Distance Graph')
    figknn.savefig(eda_path+'knn_clusters.png')

    # Apply DBSCAN
    db = DBSCAN(eps=0.02, min_samples=8)
    features=features.values
    labels = db.fit_predict(features)
    joblib.dump(db,output_path +f'DBSCAN.joblib')
    # Extract core samples (points that are part of a dense cluster)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters (ignoring noise if present)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Plotting the clusters
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    figdb,axdb=plt.subplots(figsize=(10, 7))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black is used for noise.
            color = [0, 0, 0, 1]

        class_member_mask = (labels == label)
        xy = features[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=10)

        xy = features[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

    axdb.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
    axdb.set_xlabel("area")
    axdb.set_ylabel("price")
    plt.tight_layout()
    figdb.savefig(eda_path+'dbscan_clusters_1.png')

    # Plotting the clusters 
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    figdb2,axdb2=plt.subplots(figsize=(10, 7))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black is used for noise.
            color = [0, 0, 0, 1]

        class_member_mask = (labels == label)
        xy = features[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=10)

        xy = features[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

    axdb2.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
    axdb2.set_xlabel("room")
    axdb2.set_ylabel("price")
    plt.tight_layout()
    figdb2.savefig(eda_path+'dbscan_clusters_2.png')

    # Plotting the clusters 
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    figdb3,axdb3=plt.subplots(figsize=(10, 7))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Black is used for noise.
            color = [0, 0, 0, 1]

        class_member_mask = (labels == label)
        xy = features[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 0], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=10)

        xy = features[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 2], xy[:, 0], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=5)

    axdb3.set_title(f"DBSCAN Clustering (Number of clusters: {n_clusters})")
    axdb3.set_xlabel("room")
    axdb3.set_ylabel("area")
    plt.tight_layout()
    figdb3.savefig(eda_path+'dbscan_clusters_3.png')

    # Con DBSCAN hallamos 5 clusteres con los parametros de 0.02 de epsilon y 8 de minimo de core points.

    # Hierachical clustering
    figd,axd=plt.subplots(figsize=(10, 7))
    linkage_data = linkage(features, method = 'ward', metric = 'euclidean')
    dendrogram(linkage_data)
    axd.set_title('Dendogram')
    plt.tight_layout()
    figd.savefig(eda_path+'h_dendogram.png')

    hierarchical_cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
    hierarchical_cluster.fit(features)
    joblib.dump(hierarchical_cluster,output_path +f'hierarchical_cluster.joblib')
    features=data[['area','price(usd)','room']]
    features['Class'] = hierarchical_cluster.labels_
    figh = plt.figure()
    axh = figh.add_subplot(111, projection='3d')
    x = np.array(features['room'])
    y = np.array(features['price(usd)'])
    z = np.array(features['area'])
    c = features['Class']
    sc = axh.scatter(x, y, z, c=c, cmap='viridis', s=50)
    plt.title('Room vs Price vs Area')
    axh.set_xlabel('Room')
    axh.set_ylabel('Price')
    axh.set_zlabel('Area')
    plt.tight_layout()
    figh.savefig(eda_path+'h_clusters.png')