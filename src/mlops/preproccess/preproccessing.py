from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import Dict
from data_validation.data_validation import load_config

config = load_config("config.yaml")
params = config.get("preprocessing", {})
data_split = config.get('data_split')


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

