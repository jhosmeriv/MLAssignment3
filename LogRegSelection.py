from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import data_process
import pandas as pd


titanic_datasets = data_process.get_titanic()
credit_datasets = data_process.get_credit()

# Get scores
titanic_frame = None
credit_frame = None
dataset_name = None
threshold = 'mean'
for dataset in [0, 1]:
    if dataset == 0:
        X_train, X_val, X_test, y_train, y_val, y_test = titanic_datasets
        threshold = 'mean'
        dataset_name = 'Titanic'
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = credit_datasets
        threshold = '1.5*mean'
        dataset_name = 'Credit'

    recon_errors = []

    selector = SelectFromModel(LogisticRegression(random_state=0), threshold=threshold)
    selector.fit(X_train, y_train)
    df = pd.DataFrame(selector.estimator_.coef_, columns=X_train.columns)
    df.loc[1] = abs(df.iloc[0])
    df = df.sort_values(1, ascending=False, axis=1)
    print(dataset_name)
    print(selector.threshold_)
    print(df.loc[0, abs(df.loc[0]) > selector.threshold_])
