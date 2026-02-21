import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score
from pyod.models.auto_encoder import AutoEncoder

from utils import ensure_dir, save_confusion_matrix, save_roc_curve, save_score_hist

RANDOM_STATE = 42

def parse_args():
    p = argparse.ArgumentParser(description="Fraud Detection using PyOD AutoEncoder (Anomaly Detection)")
    p.add_argument("--data", type=str, default="data/creditcard.csv", help="Path to creditcard.csv")
    p.add_argument("--outdir", type=str, default="outputs", help="Where to write outputs")
    p.add_argument("--test_size", type=float, default=0.25, help="Test set proportion")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=512, help="Batch size")
    p.add_argument("--contamination", type=float, default=0.00172, help="Expected outlier ratio (fraud rate)")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.outdir)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find {args.data}. Download the Kaggle dataset CSV and place it at the data/creditcard.csv"
        )

    # Load dataset
    df = pd.read_csv(args.data)

    # Expected schema: Time, V1..V28, Amount, Class
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column (0=legit, 1=fraud).")

    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class"])

    # Basic cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Split (stratified so fraud proportion is preserved in test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )

    # IMPORTANT (best practice for anomaly detection):
    # Train the AutoEncoder ONLY on normal transactions (Class=0)
    X_train_norm = X_train[y_train == 0]

    # Scale features (fit scaler on normal training data only)
    scaler = StandardScaler()
    X_train_norm_scaled = scaler.fit_transform(X_train_norm)
    X_test_scaled = scaler.transform(X_test)

    # Build AutoEncoder model
    # Outlier score = reconstruction error. :contentReference[oaicite:4]{index=4}
    clf = AutoEncoder(
        contamination=args.contamination,
        preprocessing=False,           # we already scaled
        hidden_neuron_list=[128, 64, 32],
        dropout_rate=0.2,
        batch_norm=True,
        hidden_activation_name="relu",
        epoch_num=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        device=args.device,
        random_state=RANDOM_STATE,
        verbose=1
    )

    clf.fit(X_train_norm_scaled)

    # Predictions + scores
    # decision_function returns raw outlier scores (higher = more anomalous)
    scores_test = clf.decision_function(X_test_scaled)
    y_pred = clf.predict(X_test_scaled)   # 0/1 based on learned threshold

    # Metrics
    report = classification_report(y_test, y_pred, digits=4)
    ap = average_precision_score(y_test, scores_test)  # PR-AUC-ish (good for imbalance)
    roc_auc = save_roc_curve(y_test, scores_test, os.path.join(args.outdir, "roc_curve.png"))
    save_confusion_matrix(y_test, y_pred, os.path.join(args.outdir, "confusion_matrix.png"))
    save_score_hist(scores_test, y_test, os.path.join(args.outdir, "score_histogram.png"))

    # Save metrics to file (for your Word doc screenshot too)
    metrics_path = os.path.join(args.outdir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=== PyOD AutoEncoder Fraud Detection ===\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Contamination: {args.contamination}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write(f"ROC AUC: {roc_auc:.6f}\n")
        f.write(f"Average Precision (AP): {ap:.6f}\n")

    print("\n" + report)
    print(f"ROC AUC: {roc_auc:.6f}")
    print(f"Average Precision (AP): {ap:.6f}")
    print(f"\nSaved outputs to: {args.outdir}")

if __name__ == "__main__":
    main()
