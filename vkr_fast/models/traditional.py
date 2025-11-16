from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


RF_PARAM: Dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 12,
    "n_jobs": -1,
    "random_state": 42,
}

CB_PARAM: Dict[str, Any] = {
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3,
    "iterations": 300,
    "loss_function": "MAE",
    "thread_count": 4,
    "random_seed": 42,
    "verbose": False,
}


def fit_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(**RF_PARAM)
    model.fit(X_train, y_train)
    return model


def fit_catboost(X_train: np.ndarray, y_train: np.ndarray):
    # Lazy import to allow environments without catboost
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(**CB_PARAM)
    model.fit(X_train, y_train)
    return model
