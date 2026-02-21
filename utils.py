import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_confusion_matrix(y_true, y_pred, out_path: str, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit(0)", "Fraud(1)"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_roc_curve(y_true, scores, out_path: str, title: str = "ROC Curve") -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return roc_auc

def save_score_hist(scores, y_true, out_path: str, title: str = "Outlier Score Distribution") -> None:
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots()
    ax.hist(scores[y_true == 0], bins=60, alpha=0.7, label="Legit(0)")
    ax.hist(scores[y_true == 1], bins=60, alpha=0.7, label="Fraud(1)")
    ax.set_title(title)
    ax.set_xlabel("Outlier score (reconstruction error)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
