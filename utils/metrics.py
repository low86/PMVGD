import numpy as np

def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10,  20,  30]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds

def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks