#!/usr/bin/env python3
"""
Phase 1: HELP Pipeline Verification (Speed Optimized)
======================================================
Replicate HELP pipeline exactly using their Zenodo data.
Target: BA ~ 0.89 (validates our implementation before Phase 2)

Speed optimizations:
- float32 data types
- LightGBM with early stopping
- Unbuffered output
- Progress reporting
"""

import sys
import os
import gc

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import warnings
warnings.filterwarnings("ignore")

# Configuration
SEED = 42
N_FOLDS = 5
N_REPEATS = 10
N_VOTERS = 13
LEARNING_RATE = 0.1
N_ESTIMATORS = 500  # Higher with early stopping
BOOSTING_TYPE = 'gbdt'
EARLY_STOPPING_ROUNDS = 50

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/raw/help_features')


class SplittingVotingEnsembleLGBM:
    """sveLGBM with early stopping for faster training."""

    def __init__(self, n_voters=-1, learning_rate=0.1, n_estimators=500,
                 boosting_type='gbdt', random_state=42, early_stopping=50):
        self.n_voters = n_voters
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.boosting_type = boosting_type
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)

        unique, counts = np.unique(y, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        majority_count = counts.max()
        minority_count = counts.min()

        majority_idx = np.where(y == majority_label)[0]
        minority_idx = np.where(y != majority_label)[0]

        if self.n_voters <= 0:
            self.n_voters = max(1, round(majority_count / minority_count))

        rng = np.random.RandomState(self.random_state)
        shuffled_majority = rng.permutation(majority_idx)
        majority_splits = np.array_split(shuffled_majority, self.n_voters)

        self.estimators_ = []
        for i, split in enumerate(majority_splits):
            X_subset = np.vstack([X[split], X[minority_idx]])
            y_subset = np.concatenate([y[split], y[minority_idx]])

            # Split for early stopping validation
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_subset, y_subset, test_size=0.15, random_state=self.random_state, stratify=y_subset
            )

            clf = LGBMClassifier(
                verbose=-1,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                boosting_type=self.boosting_type,
                random_state=self.random_state,
                n_jobs=2,
                force_col_wise=True
            )

            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(self.early_stopping, verbose=False)]
            )
            self.estimators_.append(clf)

            del X_subset, y_subset, X_tr, X_val, y_tr, y_val
            gc.collect()

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        sum_proba = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)
        for est in self.estimators_:
            sum_proba += est.predict_proba(X).astype(np.float32)
        return sum_proba / self.n_voters

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def load_and_preprocess():
    """Load and preprocess data."""
    print("Loading data...")

    bio = pd.read_csv(os.path.join(DATA_DIR, 'Kidney_BIO.csv'), index_col=0).astype(np.float32)
    ccfs = pd.read_csv(os.path.join(DATA_DIR, 'Kidney_CCcfs.csv'), index_col=0).astype(np.float32)
    n2v = pd.read_csv(os.path.join(DATA_DIR, 'Kidney_EmbN2V_128.csv'), index_col=0).astype(np.float32)
    labels = pd.read_csv(os.path.join(DATA_DIR, 'Kidney_HELP.csv'), index_col=0)

    print(f"  Bio: {bio.shape}, CCcfs: {ccfs.shape}, N2V: {n2v.shape}, Labels: {labels.shape}")

    # Remove constants and normalize
    print("Preprocessing...")
    bio = bio.loc[:, bio.nunique() > 1]
    ccfs = ccfs.loc[:, ccfs.nunique() > 1]

    bio_scaled = pd.DataFrame(
        StandardScaler().fit_transform(bio).astype(np.float32),
        index=bio.index, columns=bio.columns
    )
    ccfs_scaled = pd.DataFrame(
        StandardScaler().fit_transform(ccfs).astype(np.float32),
        index=ccfs.index, columns=ccfs.columns
    )

    del bio, ccfs
    gc.collect()

    # Common genes
    common = sorted(set(bio_scaled.index) & set(ccfs_scaled.index) & set(n2v.index) & set(labels.index))
    print(f"  Common genes: {len(common)}")

    X = pd.concat([bio_scaled.loc[common], n2v.loc[common], ccfs_scaled.loc[common]], axis=1)
    y = labels.loc[common].replace({'E': 1, 'aE': 0, 'sNE': 0}).values.ravel()

    del bio_scaled, ccfs_scaled, n2v, labels
    gc.collect()

    print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"  Class distribution: E={np.sum(y==1)}, NE={np.sum(y==0)}")

    return X.values.astype(np.float32), y


def run_cv(X, y):
    """Run CV with progress reporting."""
    print(f"\nRunning {N_REPEATS} x {N_FOLDS}-fold CV...")

    all_ba = []
    all_auroc = []

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
        rep_ba = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = SplittingVotingEnsembleLGBM(
                n_voters=N_VOTERS,
                learning_rate=LEARNING_RATE,
                n_estimators=N_ESTIMATORS,
                boosting_type=BOOSTING_TYPE,
                random_state=SEED,
                early_stopping=EARLY_STOPPING_ROUNDS
            )
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            ba = balanced_accuracy_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_proba[:, 1])

            rep_ba.append(ba)
            all_ba.append(ba)
            all_auroc.append(auroc)

            print(f"  Rep {rep+1}/{N_REPEATS} Fold {fold+1}/{N_FOLDS}: BA={ba:.4f}")

            del clf, X_train, X_test, y_train, y_test
            gc.collect()

        print(f"  Iteration {rep+1} mean BA: {np.mean(rep_ba):.4f}")

    return np.array(all_ba), np.array(all_auroc)


def main():
    print("=" * 60)
    print("Phase 1: HELP Pipeline Verification")
    print("=" * 60)
    print(f"Target: BA ~ 0.89")
    print(f"Config: {N_FOLDS}-fold x {N_REPEATS} iter, n_voters={N_VOTERS}")
    print()

    X, y = load_and_preprocess()
    ba_scores, auroc_scores = run_cv(X, y)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    mean_ba = ba_scores.mean()
    std_ba = ba_scores.std()
    mean_auroc = auroc_scores.mean()
    std_auroc = auroc_scores.std()

    print(f"Balanced Accuracy: {mean_ba:.4f} +/- {std_ba:.4f}")
    print(f"AUROC:             {mean_auroc:.4f} +/- {std_auroc:.4f}")

    target = 0.89
    diff = abs(mean_ba - target)
    status = "SUCCESS" if diff < 0.02 else "NEEDS_REVIEW"
    print(f"\nStatus: {status} (diff from target: {diff:.4f})")

    # Save
    results_dir = os.path.dirname(__file__)
    pd.DataFrame({'BA': ba_scores, 'AUROC': auroc_scores}).to_csv(
        os.path.join(results_dir, 'phase1_results.csv'), index=False
    )
    pd.DataFrame([{
        'phase': '1', 'method': 'HELP_exact',
        'BA_mean': mean_ba, 'BA_std': std_ba,
        'AUROC_mean': mean_auroc, 'AUROC_std': std_auroc,
        'target_BA': target, 'status': status
    }]).to_csv(os.path.join(results_dir, 'phase1_summary.csv'), index=False)

    print(f"\nResults saved.")
    return mean_ba, status


if __name__ == "__main__":
    main()
