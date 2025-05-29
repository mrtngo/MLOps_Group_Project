# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import mean_squared_error, roc_auc_score

# import pandas as pd

# def load_data(file_path):
#     """
#     Loads data from a CSV file and returns a pandas DataFrame.
    
#     Args:
#         file_path: str, path to the CSV file
        
#     Returns:
#         pandas DataFrame
#     """
#     df = pd.read_csv(file_path)
#     return df

# def define_features_and_label(df, symbols):
#     """
#     Defines the feature columns and target label for regression and classification tasks.
    
#     Args:
#         df: pandas DataFrame
#         symbols: list of symbols
        
#     Returns:
#         tuple: (feature_cols, label_col)
#     """
#     feature_cols = symbols  # Use the symbols list directly as feature columns
#     label_col = "BTCUSDT_price"
#     # print(f"Feature columns: {feature_cols}")
#     # print(f"Label column: {label_col}")
#     return feature_cols, label_col

# def create_price_direction_label(df, label_col):
#     """
#     Creates a binary price direction column based on price changes.
    
#     Args:
#         df: pandas DataFrame
#         label_col: name of the price column
        
#     Returns:
#         pandas DataFrame with price direction column added
#     """
#     df = df.sort_values('timestamp').copy()
#     df['prev_price'] = df[label_col].shift(1)
#     df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
#     df = df.dropna()
#     # print(df)
#     return df

# def prepare_features(df, feature_cols, label_col):
#     """
#     Prepares feature matrix and target variables for machine learning.
    
#     Args:
#         df: pandas DataFrame
#         feature_cols: list of feature column names
#         label_col: name of the label column
        
#     Returns:
#         tuple: (X, y_reg, y_class) where X is features, y_reg is regression target, y_class is classification target
#     """
#     X = df[feature_cols]
#     y_reg = df[label_col]
#     y_class = df['price_direction']
#     # print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape}, Classification target shape: {y_class.shape}")
#     return X, y_reg, y_class

# # Load data once
# df = load_data("./data/processed/futures_data_processed.csv")

# # Define symbols (including both price and funding rate columns)
# symbols = [
#     "ETHUSDT_price", "XRPUSDT_price", "ADAUSDT_price", "SOLUSDT_price", "BNBUSDT_price",
#     "BTCUSDT_funding_rate", "ETHUSDT_funding_rate", "XRPUSDT_funding_rate",
#     "ADAUSDT_funding_rate", "SOLUSDT_funding_rate", "BNBUSDT_funding_rate"
# ]

# # Get feature columns and label
# feature_cols, label_col = define_features_and_label(df, symbols)

# # Create price direction column
# df = create_price_direction_label(df, label_col)

# # Prepare features
# X, y_reg, y_class = prepare_features(df, feature_cols, label_col)


# def split_data(X, y, test_size=0.2):
#     X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     print(X_train, y_train, X_test, y_test)
#     return X_train, y_train, X_test, y_test

# def train_linear_regression(X_train, y_train, X_test, y_test):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)
#     print(f"Linear Regression RMSE: {rmse}")
#     return model, preds

# def train_logistic_regression(X_train, y_train, X_test, y_test):
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     roc = roc_auc_score(y_test, preds)
#     print(f"Logistic Regression ROC AUC: {roc}")
#     return model, preds

# # print("Training Linear Regression Model...")
# # split_data(X, y_reg)


# train_linear_regression(split_data(X, y_reg))



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
import pandas as pd


def load_data(file_path):
    """
    Loads data from a CSV file and returns a pandas DataFrame.
    
    Args:
        file_path: str, path to the CSV file
        
    Returns:
        pandas DataFrame
    """
    df = pd.read_csv(file_path)
    return df


def define_features_and_label(df, symbols):
    """
    Defines the feature columns and target label for regression and classification tasks.

    Args:
        df: pandas DataFrame
        symbols: list of symbols

    Returns:
        tuple: (feature_cols, label_col)
    """
    feature_cols = symbols  # Use the symbols list directly as feature columns
    label_col = "BTCUSDT_price"
    print(f"Feature columns: {feature_cols}")
    print(f"Label column: {label_col}")
    return feature_cols, label_col


def create_price_direction_label(df, label_col):
    """
    Creates a binary price direction column based on price changes.

    Args:
        df: pandas DataFrame
        label_col: name of the price column

    Returns:
        pandas DataFrame with price direction column added
    """
    df = df.sort_values('timestamp').copy()
    df['prev_price'] = df[label_col].shift(1)
    df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
    df = df.dropna()
    print(df)
    return df


def prepare_features(df, feature_cols, label_col):
    """
    Prepares feature matrix and target variables for machine learning.

    Args:
        df: pandas DataFrame
        feature_cols: list of feature column names
        label_col: name of the label column

    Returns:
        tuple: (X, y_reg, y_class) where X is features,
        y_reg is regression target, y_class is classification target
    """
    X = df[feature_cols]
    y_reg = df[label_col]
    y_class = df['price_direction']
    print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape},Classification target shape: {y_class.shape}")
    return X, y_reg, y_class


def split_data(X, y, test_size=0.2):
    """
    Splits data into training and testing sets.

    Args:
        X: feature matrix
        y: target variable
        test_size: float, proportion of dataset to include in the test split

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Trains a linear regression model and evaluates it.

    Args:
        X_train: training features
        y_train: training target
        X_test: testing features
        y_test: testing target

    Returns:
        tuple: (model, predictions)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Linear Regression RMSE: {rmse}")
    return model, preds


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains a logistic regression model and evaluates it.

    Args:
        X_train: training features
        y_train: training target
        X_test: testing features
        y_test: testing target

    Returns:
        tuple: (model, predictions)
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    roc = roc_auc_score(y_test, preds)
    print(f"Logistic Regression ROC AUC: {roc}")
    return model, preds


def train_model():
    # Load data once
    df = load_data("./data/processed/futures_data_processed.csv")

    # Define symbols (including both price and funding rate columns)
    symbols = [
        "ETHUSDT_price", "XRPUSDT_price", "ADAUSDT_price", "SOLUSDT_price", "BNBUSDT_price",
        "BTCUSDT_funding_rate", "ETHUSDT_funding_rate", "XRPUSDT_funding_rate",
        "ADAUSDT_funding_rate", "SOLUSDT_funding_rate", "BNBUSDT_funding_rate"
    ]

    # Get feature columns and label
    feature_cols, label_col = define_features_and_label(df, symbols)

    # Create price direction column
    df = create_price_direction_label(df, label_col)

    # Prepare features
    X, y_reg, y_class = prepare_features(df, feature_cols, label_col)

    # Split data for linear regression
    print("Training Linear Regression Model...")
    X_train, X_test, y_train, y_test = split_data(X, y_reg)

    # Train linear regression model
    model_lr, preds_lr = train_linear_regression(X_train, y_train, X_test, y_test)