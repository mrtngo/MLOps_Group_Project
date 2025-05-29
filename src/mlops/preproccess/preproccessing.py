from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import Dict
from src.mlops.data_validation.data_validation import load_config

config = load_config("config.yaml")
params = config.get("feature_engineering", {}).get("feature_selection", {}).get("params", {})

def select_features(df, feature_cols):
    X = df[feature_cols]
    y = df[config.get("target")]
    rf = RandomForestRegressor(n_estimators=params["n_estimators"], random_state=params["random_state"])
    rf.fit(X, y)
    importance = rf.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    selected_cols = [x[0] for x in ranked[:config.get("feature_engineering", {}).get("feature_selection", {}).get("top_n")]]
    print(f"Selected Features: {selected_cols}")
    return selected_cols


def scale_features(df, selected_cols):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[selected_cols])
    return X, scaler

def smote_oversample(X, y):
    # Custom oversample based on ratio like in PySpark
    class_counts = pd.Series(y).value_counts().to_dict()
    maj = max(class_counts, key=class_counts.get)
    min_ = min(class_counts, key=class_counts.get)
    ratio = class_counts[maj] / class_counts[min_]

    if ratio > 1.5:
        print(f"Class counts: 0={class_counts.get(0,0)}, 1={class_counts.get(1,0)}")
        print("Applied SMOTE-like oversampling.")
        sm = SMOTE(sampling_strategy='auto', random_state=params["random_state"])
        X_res, y_res = sm.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    return X_res, y_res

