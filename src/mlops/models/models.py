<<<<<<< HEAD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score



def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Linear Regression RMSE: {rmse}")
    return model, preds

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    roc = roc_auc_score(y_test, preds)
    print(f"Logistic Regression ROC AUC: {roc}")
    return model, preds

=======
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, classification_report


def select_features(df, feature_cols, label_col="BTCUSDT_price", top_n=8):
    X = df[feature_cols]
    y = df[label_col]
    rf = RandomForestRegressor(n_estimators=20, random_state=42)
    rf.fit(X, y)
    importance = rf.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    selected_cols = [x[0] for x in ranked[:top_n]]
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
        sm = SMOTE(sampling_strategy='auto', random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    return X_res, y_res

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Linear Regression RMSE: {rmse}")
    return model, preds

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    roc = roc_auc_score(y_test, preds)
    print(f"Logistic Regression ROC AUC: {roc}")
    return model, preds
>>>>>>> 669c4776e2c99a6e3e2870f50e8723130cd6af39
