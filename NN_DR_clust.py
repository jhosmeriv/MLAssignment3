import data_process
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def mlp_grid_search(data_sets, prints=True, activation='tanh', static_width=-1):
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    depths = np.arange(5) + 1
    if static_width == -1:
        widths = 10 * (np.arange(5) + 1)
        sizes = [tuple([x] * y) for x in widths for y in depths]
    else:
        sizes = [tuple([static_width] * d) for d in depths]

    best_size = -1
    best_acc = -1
    best_model = None
    acc_train = []
    acc_val = []
    for size in sizes:
        mlp = MLPClassifier(hidden_layer_sizes=size, activation=activation, random_state=0, max_iter=500)
        mlp.fit(X_train, y_train)


        train_score = mlp.score(X_train, y_train)
        acc_train.append(train_score)
        val_score = mlp.score(X_val, y_val)
        acc_val.append(val_score)

        if val_score > best_acc:
            best_model = mlp
            best_size = size
            best_acc = val_score

    if prints:
        print("best size: " + str(best_size))
        print("score: " + str(best_acc))

    return best_model, depths, acc_train, acc_val


titanic_datasets = data_process.get_titanic()
credit_datasets = data_process.get_credit()

pca = PCA(n_components=5, random_state=0)
ica = FastICA(n_components=4, random_state=0)
rp = GaussianRandomProjection(n_components=17, random_state=0)
selector = SelectFromModel(LogisticRegression(random_state=0), threshold='mean')

k_means = KMeans(n_clusters=8, random_state=0)
em = GaussianMixture(n_components=8, random_state=0)

for model in [pca, ica, rp, selector]:
    print(model)
    for clusterer in [k_means, em]:
        print(clusterer)
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        model.fit(X_train, y_train)

        X_train = model.transform(X_train)
        X_val = model.transform(X_val)
        X_test = model.transform(X_test)

        X_train = pd.get_dummies(clusterer.fit_predict(X_train))
        X_val = pd.get_dummies(clusterer.predict(X_val))
        X_test = pd.get_dummies(clusterer.predict(X_test))

        new_ds = (X_train, X_val, X_test, y_train, y_val, y_test)

        best_model, depths, acc_train, acc_val = mlp_grid_search(new_ds)

        #print(best_model.score(X_test, y_test))





