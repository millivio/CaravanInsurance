from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report
import numpy as np

def evaluate_model(name, y_true, y_scores):
    # 1) Global ranking metrics
    roc = roc_auc_score(y_true, y_scores)
    pra = average_precision_score(y_true, y_scores)
    
    # 2) Choose best threshold (maximize F1)
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    idx = np.nanargmax(f1)
    best_thr = thr[idx]
    
    # Predictions at best threshold
    preds = (y_scores >= best_thr).astype(int)
    
    # 3) Report
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pra:.4f}")
    print(f"Best-F1 threshold: {best_thr:.3f}")
    print(f"At best-F1: Precision={prec[idx]:.3f}, Recall={rec[idx]:.3f}, F1={f1[idx]:.3f}")
    print(classification_report(y_true, preds, zero_division=0))
    
    return {
        "Model": name,
        "ROC-AUC": roc,
        "PR-AUC": pra,
        "Precision(best-F1)": prec[idx],
        "Recall(best-F1)": rec[idx],
        "F1(best)": f1[idx]
    }

