import data_process
import pandas as pd
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from matplotlib import pyplot as plt


titanic_datasets = data_process.get_titanic()
credit_datasets = data_process.get_credit()

# Get scores
titanic_frame = None
credit_frame = None
for dataset in [0, 1]:
    if dataset == 0:
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets

    component_count = np.arange(48) + 1
    recon_errors = []
    for n_components in component_count:

        rp = GaussianRandomProjection(n_components=n_components, random_state=0)
        X_train_rp = rp.fit_transform(X_train)

        # Reconstruction Error
        inverse = np.linalg.pinv(rp.components_.T)
        X_projected = X_train_rp.dot(inverse)
        recon_error = (((X_train.values - X_projected) ** 2) ** .5).mean()
        recon_errors.append(recon_error)


    frame = pd.DataFrame(data=np.array([component_count, recon_errors]).T,
                         columns=['component_counts', 'recon_errors'])

    if dataset == 0:
        titanic_frame = frame
    else:
        credit_frame = frame



# Exploration Plots
dataset_name = None
fig, ax = plt.subplots(2, 1, figsize=(12, 12))
plots = 'recon_errors'
names = ['Training Reconstruction Error']
for dataset in [0, 1]:
    if dataset == 0:
        frame = titanic_frame
        dataset_name = 'Titanic'
    else:
        frame = credit_frame
        dataset_name = 'Credit'


    ax[dataset].plot(frame['component_counts'], frame['recon_errors'])
    ax[dataset].set_xlabel('N Componenets', fontsize=10)
    ax[dataset].set_ylabel('Reconstruction Error', fontsize=10)
    ax[dataset].set_title(dataset_name + ' Reconstruction Error', fontsize=14)
fig.suptitle('RP Parameter Tuning', fontsize=24)
plt.savefig('RP_tuning.png')

