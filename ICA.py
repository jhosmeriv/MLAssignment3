import data_process
from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
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

    component_count = np.arange(12) + 1
    recon_errors = []
    kurts = []
    for n_components in component_count:

        ica = FastICA(n_components=n_components, random_state=0)
        X_train_ica = ica.fit_transform(X_train)

        # Reconstruction Error
        X_projected = ica.inverse_transform(X_train_ica)
        recon_error = (((X_train.values - X_projected) ** 2) ** .5).mean()
        recon_errors.append(recon_error)

        # Kurtosis
        frame = pd.DataFrame(X_train_ica)
        kurt = abs(frame.kurt(axis=0)).mean()
        kurts.append(kurt)



    frame = pd.DataFrame(data=np.array([component_count, recon_errors, kurts]).T,
                         columns=['component_counts', 'recon_errors', 'kurts'])

    if dataset == 0:
        titanic_frame = frame
    else:
        credit_frame = frame



# Exploration Plots
dataset_name = None
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
plots = ['recon_errors', 'kurts']
names = ['Training Reconstruction Error', 'Kurtosis']
for dataset in [0, 1]:
    if dataset == 0:
        frame = titanic_frame
        dataset_name = 'Titanic'
    else:
        frame = credit_frame
        dataset_name = 'Credit'

    for plot in [0, 1]:
        val_type = plots[plot]
        ax[dataset, plot].plot(frame['component_counts'], frame[val_type])
        ax[dataset, plot].set_xlabel('N Components', fontsize=10)
        ax[dataset, plot].set_ylabel(names[plot], fontsize=10)
        ax[dataset, plot].set_title(dataset_name + ' ' + names[plot], fontsize=14)
fig.suptitle('ICA Parameter Tuning', fontsize=24)
plt.savefig('ICA_tuning.png')

# Describe First Component
k = 1
dataset_name = None
for dataset in [0, 1]:
    if dataset == 0:
        k = 4
        dataset_name = 'Titanic'
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
    else:
        k = 8
        dataset_name = 'Credit'
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets

    ica = FastICA(n_components=k, random_state=0)
    ica.fit(X_train)

    df = pd.DataFrame(ica.components_, columns= X_train.columns)
    df.loc[1] = abs(df.iloc[0])
    df = df.sort_values(1, ascending=False, axis=1)
    print(dataset_name)
    print(df.iloc[0, :10])

