# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, classification_report
# from sklearn.model_selection import train_test_split
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
#     print(f"Feature columns: {feature_cols}")
#     print(f"Label column: {label_col}")
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
#     print(df)
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
#     print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape}, Classification target shape: {y_class.shape}")
#     return X, y_reg, y_class

# def split_data(X, y, test_size=0.2):
#     """
#     Splits data into training and testing sets.
    
#     Args:
#         X: feature matrix
#         y: target variable
#         test_size: float, proportion of dataset to include in the test split
        
#     Returns:
#         tuple: (X_train, X_test, y_train, y_test)
#     """
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#     print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
#     return X_train, X_test, y_train, y_test

# def train_linear_regression(X_train, y_train, X_test, y_test):
#     """
#     Trains a linear regression model and evaluates it.
    
#     Args:
#         X_train: training features
#         y_train: training target
#         X_test: testing features
#         y_test: testing target
        
#     Returns:
#         tuple: (model, predictions)
#     """
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)
#     print(f"Linear Regression RMSE: {rmse}")
#     return model, preds

# def train_logistic_regression(X_train, y_train, X_test, y_test):
#     """
#     Trains a logistic regression model and evaluates it.
    
#     Args:
#         X_train: training features
#         y_train: training target
#         X_test: testing features
#         y_test: testing target
        
#     Returns:
#         tuple: (model, predictions)
#     """
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     roc = roc_auc_score(y_test, preds)
#     print(f"Logistic Regression ROC AUC: {roc}")
#     return model, preds

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

# # Split data for linear regression
# print("Training Linear Regression Model...")
# X_train, X_test, y_train, y_test = split_data(X, y_reg)

# # Train linear regression model
# model_lr, preds_lr = train_linear_regression(X_train, y_train, X_test, y_test)

# model_log, preds_log = train_logistic_regression(X_train, y_train, X_test, y_test)



# def plot_confusion_matrix(y_test, preds):
#     cm = confusion_matrix(y_test, preds)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
#     plt.title("Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.tight_layout()
#     plt.savefig("confusion_matrix.png")
#     plt.close()

# def plot_lr_predictions(df, y_true, y_pred, timestamp_col="timestamp"):
#     # Sorting and plotting
#     df_plot = df[[timestamp_col]].iloc[-len(y_true):].copy()
#     df_plot["actual"] = y_true
#     df_plot["predicted"] = y_pred
#     df_plot = df_plot.sort_values(by=timestamp_col)

#     plt.figure(figsize=(14, 7))
#     plt.plot(df_plot[timestamp_col], df_plot["actual"], label="Actual BTC Price", marker='o')
#     plt.plot(df_plot[timestamp_col], df_plot["predicted"], label="Predicted BTC Price", marker='x')
#     plt.xlabel("Timestamp")
#     plt.ylabel("BTC Price (USDT)")
#     plt.title("Actual vs Predicted BTC Prices Over Time")
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig("price_prediction_plot.png")
#     plt.close()

# plot_confusion_matrix(y_test, preds_log)



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
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
    feature_cols = symbols
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
        tuple: (X, y_reg, y_class) where X is features, y_reg is regression target, y_class is classification target
    """
    X = df[feature_cols]
    y_reg = df[label_col]
    y_class = df['price_direction']
    print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape}, Classification target shape: {y_class.shape}")
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
    rmse = root_mean_squared_error(y_test, preds)
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

def plot_confusion_matrix(y_test, preds):
    """
    Plots and saves a confusion matrix for classification predictions.
    
    Args:
        y_test: true labels
        preds: predicted labels
    """
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_lr_predictions(df, y_true, y_pred, timestamp_col="timestamp"):
    """
    Plots and saves actual vs predicted BTC prices over time.
    
    Args:
        df: pandas DataFrame with timestamp column
        y_true: true BTC prices
        y_pred: predicted BTC prices
        timestamp_col: name of timestamp column
    """
    df_plot = df[[timestamp_col]].iloc[-len(y_true):].copy()
    df_plot["actual"] = y_true
    df_plot["predicted"] = y_pred
    df_plot = df_plot.sort_values(by=timestamp_col)

    plt.figure(figsize=(14, 7))
    plt.plot(df_plot[timestamp_col], df_plot["actual"], label="Actual BTC Price", marker='o')
    plt.plot(df_plot[timestamp_col], df_plot["predicted"], label="Predicted BTC Price", marker='x')
    plt.xlabel("Timestamp")
    plt.ylabel("BTC Price (USDT)")
    plt.title("Actual vs Predicted BTC Prices Over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("price_prediction_plot.png")
    plt.close()

# Load data
df = load_data("./data/processed/futures_data_processed.csv")

# Define symbols
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
X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(X, y_reg)

# Train linear regression model
model_lr, preds_lr = train_linear_regression(X_train_reg, y_train_reg, X_test_reg, y_test_reg)

# Plot linear regression predictions
plot_lr_predictions(df, y_test_reg, preds_lr)

# Split data for logistic regression
print("Training Logistic Regression Model...")
X_train_class, X_test_class, y_train_class, y_test_class = split_data(X, y_class)

# Train logistic regression model
model_log, preds_log = train_logistic_regression(X_train_class, y_train_class, X_test_class, y_test_class)

# Plot confusion matrix for logistic regression
plot_confusion_matrix(y_test_class, preds_log)