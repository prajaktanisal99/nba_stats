import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the data
nba = pd.read_csv("nba_stats.csv")
print(f"Position counts :: {nba['Pos'].value_counts()}")

# Map position labels to numeric values
positional_mapping = {
    position: index for index, position in enumerate(nba["Pos"].unique())
}
nba["Pos"] = nba["Pos"].map(positional_mapping)

# Define target and features
target = nba["Pos"]
train = nba.drop(
    columns=[
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
)
print(f"Features:\n{train.columns}")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    train, target, test_size=0.2, random_state=0, stratify=target
)

# Random Forest for feature importance
rf_clf = RandomForestClassifier(n_estimators=120, random_state=0, max_depth=8)
rf_clf.fit(X_train, Y_train)

# Feature importance
importances = rf_clf.feature_importances_
feature_names = train.columns
feature_importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": importances}
)
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)
print("Feature Importance:")
print(feature_importance_df)

# Select top 12 features based on importance
top_features = feature_importance_df.head(12)["Feature"].values
print(f"Top 12 Features: {top_features}")

# Use only top 12 features for training and testing
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Define parameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt", "log2"],
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=0),
    param_distributions=param_dist,
    n_iter=50,  # Number of random combinations to try
    cv=5,  # 5-fold CV for parameter tuning
    random_state=0,
    n_jobs=-1,  # Use all available cores
)

# Fit RandomizedSearchCV on the top features
random_search.fit(X_train_top, Y_train)

# Best parameters and score from Randomized Search
print("Best parameters from Randomized Search:", random_search.best_params_)
print("Best cross-validation score from Randomized Search:", random_search.best_score_)

# Use the best estimator from RandomizedSearchCV
best_rf_clf = random_search.best_estimator_
Y_pred_top = best_rf_clf.predict(X_test_top)

# Classification Report with Best Model
print(
    f"Classification Report with Best Model:\n{classification_report(Y_test, Y_pred_top)}"
)

# Stratified 10-fold cross-validation with best model
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
cv_scores = cross_val_score(best_rf_clf, X_train_top, Y_train, cv=stratified_kfold)

# Cross-validation results
print("Stratified 10-fold Cross-validation scores with Best Model:")
print(cv_scores)
print(
    f"Stratified Cross-validation Accuracy with Best Model: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)
