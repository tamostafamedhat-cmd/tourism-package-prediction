import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ---------------------------------------------------------
# 1. Load prepared train and test datasets
# ---------------------------------------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = "ProdTaken"

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# ---------------------------------------------------------
# 2. Identify numeric and categorical columns
# ---------------------------------------------------------
numeric_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------------------------------------------------
# 4. Decision Tree model
# ---------------------------------------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=42))
    ]
)

# ---------------------------------------------------------
# 5. Hyperparameter tuning
# ---------------------------------------------------------
param_grid = {
    "clf__max_depth": [4, 6, 8, 10, 12],
    "clf__min_samples_leaf": [2, 5, 10, 20]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ---------------------------------------------------------
# 6. Evaluate best model
# ---------------------------------------------------------
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

metrics = {
    "model": "Decision Tree",
    "best_params": grid_search.best_params_,
    "cv_roc_auc": float(grid_search.best_score_),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, predictions
    ).tolist()
}

# Record every tuning experiment
experiments = []

cv_results = grid_search.cv_results_

for i in range(len(cv_results["params"])):
    experiments.append(
        {
            "parameters": cv_results["params"][i],
            "mean_cv_roc_auc": float(
                cv_results["mean_test_score"][i]
            ),
            "std_cv_roc_auc": float(
                cv_results["std_test_score"][i]
            )
        }
    )

results = {
    "best_model": metrics,
    "experiments": experiments
}

# ---------------------------------------------------------
# 7. Save trained model and experiment results
# ---------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

joblib.dump(
    best_model,
    "outputs/best_model.joblib"
)

with open("outputs/model_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Model training completed successfully.")
print("Best parameters:", grid_search.best_params_)
print("Test ROC-AUC:", metrics["test_roc_auc"])
print("Accuracy:", metrics["accuracy"])
print("F1 score:", metrics["f1"])
print("Model saved to outputs/best_model.joblib")
