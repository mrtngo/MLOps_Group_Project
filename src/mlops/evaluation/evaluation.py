import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, classification_report

def plot_confusion_matrix(y_test, preds):
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
    # Sorting and plotting
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
