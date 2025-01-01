import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
import warnings

# Suppress convergence warnings and sklearn neural network logs
warnings.simplefilter("ignore", ConvergenceWarning)
logging.getLogger("sklearn.neural_network").setLevel(logging.ERROR)


COLS = [
    "Pos",
    "Age",
    "Tm",
    "G",
    "GS",
    "MP",
    "3P",
    "3PA",
    "FG",
    "FGA",
    "2P",
    "2PA",
    "FT",
    "FTA",
]

nba = pd.read_csv("nba_stats.csv")

# mapp position to numeric values
positional_mapping = {
    position: index for index, position in enumerate(nba["Pos"].unique())
}
nba["Pos"] = nba["Pos"].map(positional_mapping)


# separate features and target data
target = nba["Pos"]
train = nba.drop(columns=COLS)

print("\nFeature Selection")
print(f"\nSelected Features :: {train.columns}")

# split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    train, target, test_size=0.2, random_state=0, stratify=target
)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

grid_search = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=200,
    activation="tanh",
    random_state=0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    learning_rate="adaptive",
    alpha=0.1,
)

# mlp_clf.fit(X_train, Y_train)
# Y_pred_train = mlp_clf.predict(X_train)
# Y_pred_test = mlp_clf.predict(X_test)
# print(
# f"Train Accuracy with Early Stopping and Regularization: {accuracy_score(Y_pred_train, Y_train):.4f}"
# )
# print(
# f"Test Accuracy with Early Stopping and Regularization: {accuracy_score(Y_pred_test, Y_test):.4f}"
# )


# Hyperparameter tuning
# param_grid = {
#     "hidden_layer_sizes": [(32, 5), (64, 32), (64, 5), (64, 48, 32, 5)],
#     "activation": ["relu", "tanh"],
#     "alpha": [0.0001, 0.001, 0.01, 0.1],
#     "n_iter_no_change": [2, 10],
#     "learning_rate": ["constant", "adaptive"],
# }

# grid_search = GridSearchCV(
#     estimator=MLPClassifier(
#         random_state=0,
#         max_iter=300,
#         early_stopping=True,
#         validation_fraction=0.1,
#         solver="adam",
#     ),
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
# )

grid_search.fit(X_train, Y_train)

# print(f"\nBest parameters from Grid Search: {grid_search.best_params_}")

# best_mlp_clf = grid_search.best_estimator_
best_mlp_clf = grid_search
Y_pred_train_best = best_mlp_clf.predict(X_train)
Y_pred_test_best = best_mlp_clf.predict(X_test)

print(
    f"\nTrain Accuracy with Best Model: {accuracy_score(Y_pred_train_best, Y_train):.4f}"
)
print(
    f"\nTest Accuracy with Best Model: {accuracy_score(Y_pred_test_best, Y_test):.4f}"
)

# 10 fold cross validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(best_mlp_clf, X_train, Y_train, cv=stratified_kfold)
print(f"\nStratified 10-fold Cross-validation scores (Train Data):\n {cv_scores}")
print(
    f"\nStratified Cross-validation Accuracy (Train Data): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)


# Dummy test dataset evaluation
print("\n\n--------------------------------")
print("Dummy Dataset")
print("--------------------------------")

dummy_test = pd.read_csv("dummy_test.csv")
dummy_test["Pos"] = dummy_test["Pos"].map(positional_mapping)

Y_dummy_test = dummy_test["Pos"]
X_dummy_test = dummy_test.drop(columns=COLS)

X_dummy_test_scaled = scaler.transform(X_dummy_test)
Y_dummy_pred = best_mlp_clf.predict(X_dummy_test_scaled)

print(
    f"Dummy Accuracy with Best Model: {accuracy_score(Y_dummy_pred, Y_dummy_test):.4f}"
)


# OUTPUT

# Feature Selection

# Selected Features :: Index(['FG%', '3P%', '2P%', 'eFG%', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL',
#        'BLK', 'TOV', 'PF', 'PTS'],
#       dtype='object')

# Train Accuracy with Best Model: 0.6023

# Test Accuracy with Best Model: 0.5380

# Stratified 10-fold Cross-validation scores (Train Data):
#  [0.44927536 0.5942029  0.53623188 0.5942029  0.63235294 0.38235294
#  0.51470588 0.58823529 0.52941176 0.52941176]

# Stratified Cross-validation Accuracy (Train Data): 0.5350 ± 0.0710


# --------------------------------
# Dummy Dataset
# --------------------------------
# Dummy Accuracy with Best Model: 0.5825
