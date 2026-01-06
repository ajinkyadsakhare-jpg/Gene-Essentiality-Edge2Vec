import pandas as pd
from sklearn.base import is_classifier, clone
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
import numpy as np
import random

pred_metrics = [roc_auc_score, top_k_accuracy_score, average_precision_score]
scorer =    {'Accuracy'  : (accuracy_score , {}),
             'BA'  :       (balanced_accuracy_score , {}),
             'Precision' : (precision_score, {'pos_label': 1, 'average':'macro'}),
             'Recall'    : (recall_score   , {'pos_label': 1, 'average':'macro'}),
             'MCC' :       (matthews_corrcoef, {}), 
            }

def evaluate_fold(train_x, train_y, test_x, test_y, estimator, indices, test_indices, targets, 
                  predictions, probabilities, fold, scorer=scorer):
    # Initialize classifier
    clf = clone(estimator)
    clf.base_estimator = estimator.base_estimator
    #print(clf.base_estimator)
    clf.fit(train_x, train_y)
    probs = clf.predict_proba(test_x)
    preds = clf.predict(test_x)
    cm = confusion_matrix(test_y, preds)
    #preds = clf.classes_[np.argmax(probs, axis=1)]
    indices = np.concatenate((indices, test_indices))
    targets = np.concatenate((targets, test_y))
    predictions = np.concatenate((predictions, preds))
    probabilities = np.concatenate((probabilities, probs[:, 0]))

    # Calculate and store evaluation metrics for each fold
    metrics = {"index": fold}
    for key in scorer.keys():
        metric, kwargs = scorer[key]
        if metric in pred_metrics:
            metrics[key] = metric(test_y, probs[:, 1], **kwargs) if len(clf.classes_) == 2 else metric(test_y, probs, **kwargs)
        else:
            metrics[key] = metric(test_y, preds, **kwargs)
    metrics['cm'] = cm
    return indices, targets, predictions, probabilities, metrics
    
def skfold_cv(X, Y, estimator, n_splits=10, scorer=scorer, verbose: bool = False, show_progress: bool = False, seed: int = 42, precision:int=3):
    """
    Perform cross-validated predictions using a classifier.

    :param DataFrame X: Features DataFrame.
    :param DataFrame Y: Target variable DataFrame.
    :param int n_splits: Number of folds for cross-validation.
    :param estimator object: Classifier method (must have fit, predict, predict_proba methods)
    :param str or None outfile: File name for saving predictions.
    :param bool show_progress: Verbosity level for printing progress bar (default: False).
    :param bool verbose: Whether to print verbose information.
    :param int or None seed: Random seed for reproducibility.

    :returns: Summary statistics of the cross-validated predictions, single measures and label predictions
    :rtype: Tuple(pd.DataFrame,pd.DataFrame,pd.DataFrame)

    :example:
 
    .. code-block:: python

        # Example usage
        from lightgbm import LGBMClassifier
        X_data = pd.DataFrame(...)
        Y_data = pd.DataFrame(...)
        clf = LGBMClassifier(random_state=0)
        df_scores, scores, predictions = k_fold_cv(df_X, df_y, clf, n_splits=5, verbose=True, display=True, seed=42)
    """
    # check estimator
    assert is_classifier(estimator) and hasattr(estimator, 'fit') and callable(estimator.fit) and hasattr(estimator, 'predict_proba') and callable(estimator.predict_proba), "Bad estimator imput!"

    # get list of indices
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        all_indices = Y.index
        y = Y.values.ravel()
    else:
        y = Y
        all_indices = np.arange(0,len(y))

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Initialize StratifiedKFold
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    nclasses = len(np.unique(y))
    indices = np.array([], dtype=str)
    targets = np.array([], dtype=np.int64)
    predictions = np.array([], dtype=np.int64)
    probabilities = np.array([], dtype=np.int64)
    scores = pd.DataFrame()

    if verbose:
        print(f'Classification with {estimator.__class__.__name__}...')

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(np.arange(len(X)), y), total=kf.get_n_splits(), 
                                                   desc=f"{n_splits}-fold", disable=not show_progress)):
        X_train, y_train = X[train_idx], y[train_idx]
        indices, targets, predictions, probabilities, metrics = evaluate_fold(X_train, y_train, X[test_idx], y[test_idx], 
                                                                            estimator, indices, all_indices[test_idx], targets, 
                                                                            predictions, probabilities, fold, scorer)
        dfs = pd.DataFrame.from_dict(metrics, orient='index').T.set_index('index')
        scores = pd.concat([scores, dfs], axis=0)

    # Calculate mean and standard deviation of evaluation metrics
    fmt = '{:.%df}' % precision
    df_scores = pd.DataFrame([f'{fmt.format(val)}±{fmt.format(err)}' for val, err in zip(scores.loc[:, scores.columns != "cm"].mean(axis=0).values,
                                                                     scores.loc[:, scores.columns != "cm"].std(axis=0))] +
                             [(scores[['cm']].sum()).values[0].tolist()],
                             columns=['measure'], index=scores.columns)

    # Create DataFrame for storing detailed predictions
    df_predictions = pd.DataFrame({'indices': indices, 'label': targets, 'prediction': predictions, 'probabilities': probabilities}).set_index(['indices'])

    # Save detailed predictions to a CSV file if requested
    if in_notebook():
        ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(df_predictions['label'].values,df_predictions['prediction'].values), display_labels=np.unique(y)).plot()

    # Return the summary statistics of cross-validated predictions, the single measures and the prediction results
    return df_scores, scores, df_predictions