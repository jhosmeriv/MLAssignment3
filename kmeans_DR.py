import data_process
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics # adjusted_mutual_info_score, silhouette_score
from sklearn.random_projection import GaussianRandomProjection
from matplotlib import pyplot as plt


titanic_datasets = data_process.get_titanic()
credit_datasets = data_process.get_credit()


def cluster_score(clusters, y):
    stacked = np.vstack((clusters, y.values)).T
    frame = pd.DataFrame(stacked, columns=['cluster', 'prediction'])

    avg_score = frame.groupby('cluster').mean()['prediction']
    weights = frame.groupby('cluster').count()['prediction']/len(y)

    pred_score = np.array([max([x, 1-x]) for x in avg_score])

    return np.matmul(pred_score, weights.values.reshape(-1, 1))[0]

# Get scores
titanic_frame = None
credit_frame = None
for dataset in [0, 1]:
    if dataset == 0:
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        nc = 5
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets
        nc = 6

    pca = PCA(n_components=nc, random_state=0)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    cluster_counts = [2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 20, 24, 28, 32, 48, 64]
    accs = []
    label_amis = []
    sils = []
    for n_clusters in cluster_counts:

        k_means = KMeans(n_clusters=n_clusters, random_state=0)

        k_means.fit(X_train)
        X_train_clusters = k_means.predict(X_train)
        X_val_clusters = k_means.predict(X_val)

        # Get accuracy
        cluster_accuracy = cluster_score(X_val_clusters, y_val)
        accs.append(cluster_accuracy)

        # How much information about label is in cluster?
        ami = metrics.adjusted_mutual_info_score(X_val_clusters, y_val)
        label_amis.append(ami)

        # Silhouette score
        sil = metrics.silhouette_score(X_train, X_train_clusters)
        sils.append(sil)


    frame = pd.DataFrame(data=np.array([cluster_counts, accs, label_amis, sils]).T,
                         columns=['cluster_counts', 'accs', 'label_amis', 'sils'])

    if dataset == 0:
        titanic_frame = frame
    else:
        credit_frame = frame

# Exploration Plots
dataset_name = None
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plots = ['accs', 'label_amis', 'sils']
names = ['Validation Accuracy', 'Validation Mutual Infoformation', 'Silhouette Score']
for dataset in [0, 1]:
    if dataset == 0:
        frame = titanic_frame
        dataset_name = 'Titanic'
    else:
        frame = credit_frame
        dataset_name = 'Credit'

    for plot in [0, 1, 2]:
        val_type = plots[plot]
        ax[dataset, plot].plot(frame['cluster_counts'], frame[val_type])
        ax[dataset, plot].set_xlabel('K Clusters', fontsize=10)
        ax[dataset, plot].set_ylabel(names[plot], fontsize=10)
        ax[dataset, plot].set_title(dataset_name + ' ' + names[plot], fontsize=14)
fig.suptitle('K Means Parameter Tuning', fontsize=24)
plt.savefig('k_means_DR_tuning_0.png')

# Visualize best (t-sne + clusters, t-sne + labels)
k = 0
dataset_name = None
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
for dataset in [0, 1]:
    if dataset == 0:
        dataset_name = 'Titanic'
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        nc = 5
        k = 10
    else:
        dataset_name = 'Credit'
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets
        nc = 6
        k = 3

    pca = PCA(n_components=nc, random_state=0)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    k_means = KMeans(n_clusters=k, random_state=0)
    X_train_clusters = k_means.fit_predict(X_train)

    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X_train)

    for i in range(k):
        mask = X_train_clusters == i
        X = X_embedded[mask, :]
        ax[dataset, 0].scatter(X[:, 0], X[:, 1], label=i+1)
    ax[dataset, 0].legend(title='Cluster Number')
    ax[dataset, 0].set_title(dataset_name + ' T-SNE Embedding With Cluster', fontsize=14)

    for i in [True, False]:
        mask = y_train == i
        X = X_embedded[mask, :]
        ax[dataset, 1].scatter(X[:, 0], X[:, 1], label=i)
    ax[dataset, 1].legend(title='Response Values')
    ax[dataset, 1].set_title(dataset_name + ' T-SNE Embedding With Label', fontsize=14)
fig.suptitle('T-SNE Embeddings and Labels', fontsize=26)
plt.savefig('k_means_DR_t-sne_0.png')











def cluster_score(clusters, y):
    stacked = np.vstack((clusters, y.values)).T
    frame = pd.DataFrame(stacked, columns=['cluster', 'prediction'])

    avg_score = frame.groupby('cluster').mean()['prediction']
    weights = frame.groupby('cluster').count()['prediction']/len(y)

    pred_score = np.array([max([x, 1-x]) for x in avg_score])

    return np.matmul(pred_score, weights.values.reshape(-1, 1))[0]

# Get scores
titanic_frame = None
credit_frame = None
for dataset in [0, 1]:
    if dataset == 0:
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        nc = 17
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets
        nc = 40

    rp = GaussianRandomProjection(n_components=nc, random_state=0)
    X_train = rp.fit_transform(X_train)
    X_val = rp.transform(X_val)
    X_test = rp.transform(X_test)
    cluster_counts = [2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 16, 20, 24, 28, 32, 48, 64]
    accs = []
    label_amis = []
    sils = []
    for n_clusters in cluster_counts:

        k_means = KMeans(n_clusters=n_clusters, random_state=0)

        k_means.fit(X_train)
        X_train_clusters = k_means.predict(X_train)
        X_val_clusters = k_means.predict(X_val)

        # Get accuracy
        cluster_accuracy = cluster_score(X_val_clusters, y_val)
        accs.append(cluster_accuracy)

        # How much information about label is in cluster?
        ami = metrics.adjusted_mutual_info_score(X_val_clusters, y_val)
        label_amis.append(ami)

        # Silhouette score
        sil = metrics.silhouette_score(X_train, X_train_clusters)
        sils.append(sil)


    frame = pd.DataFrame(data=np.array([cluster_counts, accs, label_amis, sils]).T,
                         columns=['cluster_counts', 'accs', 'label_amis', 'sils'])

    if dataset == 0:
        titanic_frame = frame
    else:
        credit_frame = frame

# Exploration Plots
dataset_name = None
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
plots = ['accs', 'label_amis', 'sils']
names = ['Validation Accuracy', 'Validation Mutual Infoformation', 'Silhouette Score']
for dataset in [0, 1]:
    if dataset == 0:
        frame = titanic_frame
        dataset_name = 'Titanic'
    else:
        frame = credit_frame
        dataset_name = 'Credit'

    for plot in [0, 1, 2]:
        val_type = plots[plot]
        ax[dataset, plot].plot(frame['cluster_counts'], frame[val_type])
        ax[dataset, plot].set_xlabel('K Clusters', fontsize=10)
        ax[dataset, plot].set_ylabel(names[plot], fontsize=10)
        ax[dataset, plot].set_title(dataset_name + ' ' + names[plot], fontsize=14)
fig.suptitle('K Means Parameter Tuning', fontsize=24)
plt.savefig('k_means_DR_tuning_1.png')

# Visualize best (t-sne + clusters, t-sne + labels)
k = 0
dataset_name = None
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
for dataset in [0, 1]:
    if dataset == 0:
        dataset_name = 'Titanic'
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        nc = 17
        k = 5
    else:
        dataset_name = 'Credit'
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets
        nc = 40
        k = 4

    rp = GaussianRandomProjection(n_components=nc, random_state=0)
    X_train = rp.fit_transform(X_train)
    X_val = rp.transform(X_val)
    X_test = rp.transform(X_test)

    k_means = KMeans(n_clusters=k, random_state=0)
    X_train_clusters = k_means.fit_predict(X_train)

    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X_train)

    for i in range(k):
        mask = X_train_clusters == i
        X = X_embedded[mask, :]
        ax[dataset, 0].scatter(X[:, 0], X[:, 1], label=i+1)
    ax[dataset, 0].legend(title='Cluster Number')
    ax[dataset, 0].set_title(dataset_name + ' T-SNE Embedding With Cluster', fontsize=14)

    for i in [True, False]:
        mask = y_train == i
        X = X_embedded[mask, :]
        ax[dataset, 1].scatter(X[:, 0], X[:, 1], label=i)
    ax[dataset, 1].legend(title='Response Values')
    ax[dataset, 1].set_title(dataset_name + ' T-SNE Embedding With Label', fontsize=14)
fig.suptitle('T-SNE Embeddings and Labels', fontsize=26)
plt.savefig('k_means_DR_t-sne_1.png')
