'''
This file contains functions that calculate metrics used
to determine the performance of a classification model.
'''

def calculate_class_precision(l, preds, labels):
    '''
    Calculates precision for `l`, the "positive" label in a dataset. \\
    Precision defined as: true positives / (true positives + false positives)
    
    Parameters:
    - `l`: a valid label that is found in `preds` and `labels`
    - `preds`: NumPy `ndarray` of predicted labels
    - `labels`: NumPy `ndarray` of true labels
    
    '''
    tp = preds[(labels==l) & (preds==l)].shape[0]
    fp = preds[(labels!=l) & (preds==l)].shape[0]

    return tp / (tp + fp) if (tp + fp) != 0 else 0

def calculate_class_recall(l, preds, labels):
    '''
    Calculates recall for `l`, the "positive" label in a dataset. \\
    Recall defined as: true positives / (true positives + false negatives)
    
    Parameters:
    - `l`: a valid label that is found in `preds` and `labels`
    - `preds`: NumPy `ndarray` of predicted labels
    - `labels`: NumPy `ndarray` of true labels
    
    '''
    tp = preds[(labels==l) & (preds==l)].shape[0]
    fn = preds[(labels==l) & (preds!=l)].shape[0]

    return tp / (tp + fn) if (tp + fn) != 0 else 0

def calculate_class_fscore(prec, rec):
    '''
    Calculates the F-Score given a precision and recall. For the returned
    value to be meaningful, the provided precision and recall values must
    have been calculated using the same "positive" label and on the same sets
    of predicted and actual labels. \\
    F-Score defined as 2 * precision * recall / (precision * recall)

    Parameters:
    - `prec`: precision value
    - `rec`: recall value
    '''
    return (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0