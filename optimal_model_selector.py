# A Python utility to select and use the most optimal model by hyper-parameter tuning over a params grid.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd


input_df = pd.read_csv('testData.csv')
X, y = input_df.loc[:, input_df.columns != 'target'], input_df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Choose the type of classifier: random forest
rf_estimator = RandomForestClassifier(random_state=1, bootstrap=True, oob_score=True)

# Grid of parameters to choose from
parameters = {
    "n_estimators": [101, 201, 301],
    "min_samples_leaf": np.arange(1, 6, 1),
    "max_features": [0.6, 0.7, 0.8, 'log2', 'auto'],
    "max_samples": [0.7, 0.8, 0.9, None]
}


# Run the grid search

# Multiple scores are to be computed during the grid search
scoring = {'Accuracy': 'accuracy', 'Precision': 'precision_micro', 'Recall': 'recall_micro'}

# Use recall (at this time) to find the best parameters for refitting the estimator at the end
gs = GridSearchCV(rf_estimator, param_grid=parameters, scoring=scoring, refit='Recall', cv=5)
gs.fit(X_train, y_train)  # Run fit with all sets of parameters.

# Retrieve the best estimator, i.e., the combination of parameters that gave best score (highest recall)
# on the left out data during cross-validation
rf_estimator = gs.best_estimator_


# Fit the best random forest estimator to the training data
rf_estimator.fit(X_train, y_train)


print("Accuracy on the test set : ", rf_estimator.score(X_test, y_test))

pred_test = rf_estimator.predict(X_test)
print("Precision on the test set : ", metrics.precision_score(y_test, pred_test, average='micro'))
print("Recall on the test set : ", metrics.recall_score(y_test, pred_test, average='micro'))
