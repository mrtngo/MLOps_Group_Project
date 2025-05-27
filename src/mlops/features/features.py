def define_features_and_label(df, symbols):
    """
    Defines the feature columns and target label for regression and classification tasks.
    
    Args:
        df: pandas DataFrame
        symbols: list of symbols
        
    Returns:
        tuple: (feature_cols, label_col)
    """
    feature_cols = [f"{symbol}_price" for symbol in symbols[1:]] + \
                   [f"{symbol}_funding_rate" for symbol in symbols]
    label_col = "BTCUSDT_price"
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
    return X, y_reg, y_class