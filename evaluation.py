import torch
import numpy as np
import tqdm
import sklearn.metrics as skm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):

    y_labels = (y_pred > 0.5).astype(int)
    accuracy = sum(y_labels == y_true) / len(y_true) * 100
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_labels).ravel()
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    f1_score = (2 * precision * recall) / (precision + recall)
    roc_auc = skm.roc_auc_score(y_true, y_pred)
    result = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "ROC_AUC": roc_auc
    }
    return result

def evaluate(model, X_test, y_test, batch_size=64, prefixes=None):
    model.eval()

    test_loader = torch.utils.data.DataLoader(
        range(len(X_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    y_pred = []
    y_true = y_test.detach().cpu().numpy()

    for i, indices in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            if prefixes is not None:
                prefixes_slice = prefixes[indices]
            else:
                prefixes_slice = None
            y_pred.extend(model(X_test[indices], prefixes=prefixes_slice).tolist())

    y_pred = np.array(y_pred)

    result = compute_metrics(y_true, y_pred)

    print(result)

    return result
