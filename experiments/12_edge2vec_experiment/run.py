#!/usr/bin/env python3
"""
Phase 2b: Edge2Vec on Raw IID (Memory Optimized)
=================================================
Train Edge2Vec on filtered IID network (kidney tissue, HELP genes).
Uses edge type information (exp/pred/ortho) in random walks.

Memory optimizations:
- float32 data types
- Aggressive garbage collection
- Sequential voter training with early stopping
- Unbuffered output
"""

import sys
import os
import gc

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier, early_stopping
import warnings
warnings.filterwarnings("ignore")

from embedding.edge2vec import edge2vec_embed, read_graph

# Configuration
SEED = 42
N_FOLDS = 5
N_REPEATS = 10
N_VOTERS = 13
LEARNING_RATE = 0.1
N_ESTIMATORS = 500
BOOSTING_TYPE = 'gbdt'
EARLY_STOPPING_ROUNDS = 50

# Edge2Vec params (matching Node2Vec where applicable)
E2V_DIM = 128
E2V_WALK_LENGTH = 80
E2V_NUM_WALKS = 10
E2V_WINDOW = 5
E2V_EPOCHS = 1  # Match Node2Vec
E2V_P = 1.0
E2V_Q = 1.0
E2V_EM_ITERATIONS = 5  # EM iterations for transition matrix
E2V_EM_WALKS = 2
E2V_EM_WALK_LENGTH = 3

# Paths
DATA_DIR = os.path.join(project_root, 'data/raw')
PROCESSED_DIR = os.path.join(project_root, 'data/processed')
RESULTS_DIR = os.path.dirname(__file__)


class SplittingVotingEnsembleLGBM:
    """sveLGBM with early stopping (memory optimized)."""

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


def step1_ensure_iid_processed():
    """Step 1: Ensure IID network is processed (reuse from Phase 2a if exists)."""
    print("\n" + "=" * 60)
    print("Step 1: Checking IID Network Processing")
    print("=" * 60)

    graph_dir = os.path.join(PROCESSED_DIR, 'graphs/kidney')
    edge2vec_input = os.path.join(graph_dir, 'kidney_edge2vec_input.txt')
    mapping_file = os.path.join(graph_dir, 'kidney_edge_type_mapping.csv')

    if os.path.exists(edge2vec_input) and os.path.exists(mapping_file):
        print(f"  Using existing processed network: {edge2vec_input}")
        mapping = pd.read_csv(mapping_file)
        n_types = len(mapping)
        print(f"  Edge types: {n_types}")
        return edge2vec_input, n_types, graph_dir

    # Need to process IID - this should have been done by Phase 2a
    raise FileNotFoundError(f"Edge2Vec input not found: {edge2vec_input}. Run Phase 2a first.")


def step2_train_edge2vec(edge2vec_input: str, n_types: int, output_dir: str):
    """Step 2: Train Edge2Vec on processed network."""
    print("\n" + "=" * 60)
    print("Step 2: Training Edge2Vec")
    print("=" * 60)
    print(f"  Edge types: {n_types}")
    print(f"  EM iterations: {E2V_EM_ITERATIONS}")
    print(f"  Walk length: {E2V_WALK_LENGTH}, Num walks: {E2V_NUM_WALKS}")

    emb_dir = os.path.join(output_dir, '../embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, 'kidney_edge2vec_iid.txt')

    # Run Edge2Vec
    embeddings = edge2vec_embed(
        edge_list_path=edge2vec_input,
        output_path=emb_path,
        n_edge_types=n_types,
        dimensions=E2V_DIM,
        walk_length=E2V_WALK_LENGTH,
        num_walks=E2V_NUM_WALKS,
        window_size=E2V_WINDOW,
        p=E2V_P,
        q=E2V_Q,
        em_iterations=E2V_EM_ITERATIONS,
        em_walks=E2V_EM_WALKS,
        em_walk_length=E2V_EM_WALK_LENGTH,
        workers=4,  # Parallel walk generation
        epochs=E2V_EPOCHS,
        seed=SEED,
        verbose=True
    )

    # Convert to DataFrame
    data = []
    for node, emb in embeddings.items():
        row = {'node': node}
        for i, val in enumerate(emb):
            row[f'Edge2Vec_{i}'] = float(val)
        data.append(row)

    e2v_df = pd.DataFrame(data).set_index('node').astype(np.float32)
    print(f"  Embeddings: {e2v_df.shape}")

    # Save as CSV
    csv_path = os.path.join(emb_dir, 'kidney_edge2vec_iid.csv')
    e2v_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    del embeddings, data
    gc.collect()

    return e2v_df


def step3_build_features(e2v_df):
    """Build feature matrix (Bio + CCcfs + E2V) with memory optimization."""
    print("\n" + "=" * 60)
    print("Step 3: Building Feature Matrix")
    print("=" * 60)

    features_dir = os.path.join(DATA_DIR, 'help_features')

    # Load Bio (small file)
    print("Loading Bio features...")
    bio = pd.read_csv(os.path.join(features_dir, 'Kidney_BIO.csv'), index_col=0).astype(np.float32)
    bio = bio.loc[:, bio.nunique() > 1]
    print(f"  Bio: {bio.shape}")

    # Load Labels
    print("Loading labels...")
    labels = pd.read_csv(os.path.join(features_dir, 'Kidney_HELP.csv'), index_col=0)
    print(f"  Labels: {labels.shape}")

    # Find common genes (before loading CCcfs)
    common_genes = sorted(set(bio.index) & set(e2v_df.index) & set(labels.index))
    print(f"  Common genes: {len(common_genes)}")

    # Subset to common genes
    bio = bio.loc[common_genes]
    e2v_df = e2v_df.loc[common_genes]
    labels = labels.loc[common_genes]

    # Scale Bio
    print("Scaling Bio...")
    bio_scaled = pd.DataFrame(
        StandardScaler().fit_transform(bio).astype(np.float32),
        index=bio.index, columns=bio.columns
    )
    del bio
    gc.collect()

    # Load CCcfs (largest file)
    print("Loading CCcfs features...")
    ccfs_path = os.path.join(features_dir, 'Kidney_CCcfs.csv')
    ccfs = pd.read_csv(ccfs_path, index_col=0)
    ccfs = ccfs.loc[ccfs.index.intersection(common_genes)].astype(np.float32)
    ccfs = ccfs.loc[:, ccfs.nunique() > 1]
    print(f"  CCcfs: {ccfs.shape}")

    # Scale CCcfs
    print("Scaling CCcfs...")
    ccfs_scaled = pd.DataFrame(
        StandardScaler().fit_transform(ccfs).astype(np.float32),
        index=ccfs.index, columns=ccfs.columns
    )
    del ccfs
    gc.collect()

    # Ensure same index order
    common_genes = sorted(set(bio_scaled.index) & set(ccfs_scaled.index) & set(e2v_df.index))
    print(f"  Final common genes: {len(common_genes)}")

    # Concatenate features
    print("Concatenating features...")
    X = pd.concat([
        bio_scaled.loc[common_genes],
        e2v_df.loc[common_genes],
        ccfs_scaled.loc[common_genes]
    ], axis=1)

    y = labels.loc[common_genes].replace({'E': 1, 'aE': 0, 'sNE': 0}).values.ravel()

    print(f"  Final X: {X.shape}")
    print(f"  Class distribution: E={np.sum(y==1)}, NE={np.sum(y==0)}")

    del bio_scaled, ccfs_scaled, e2v_df, labels
    gc.collect()

    return X.values.astype(np.float32), y, len(common_genes)


def step4_run_cv(X, y):
    """Run CV with progress reporting."""
    print("\n" + "=" * 60)
    print("Step 4: Running Cross-Validation")
    print("=" * 60)
    print(f"Config: {N_FOLDS}-fold x {N_REPEATS} iterations, n_voters={N_VOTERS}")

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
    print("Phase 2b: Edge2Vec on Raw IID")
    print("=" * 60)
    print(f"Goal: Test if edge type info improves prediction (Y > X?)")

    # Step 1: Ensure IID processed
    edge2vec_input, n_types, graph_dir = step1_ensure_iid_processed()

    # Step 2: Train Edge2Vec
    e2v_df = step2_train_edge2vec(edge2vec_input, n_types, graph_dir)

    # Step 3: Build features
    X, y, n_genes = step3_build_features(e2v_df)
    n_features = X.shape[1]

    # Step 4: Run CV
    ba_scores, auroc_scores = step4_run_cv(X, y)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS - Phase 2b (Edge2Vec)")
    print("=" * 60)

    mean_ba = ba_scores.mean()
    std_ba = ba_scores.std()
    mean_auroc = auroc_scores.mean()
    std_auroc = auroc_scores.std()

    print(f"Balanced Accuracy: {mean_ba:.4f} +/- {std_ba:.4f}")
    print(f"AUROC:             {mean_auroc:.4f} +/- {std_auroc:.4f}")
    print()
    print(f"This is result Y for comparison with baseline X (Phase 2a)")

    # Save results
    pd.DataFrame({'BA': ba_scores, 'AUROC': auroc_scores}).to_csv(
        os.path.join(RESULTS_DIR, 'phase2b_results.csv'), index=False
    )

    summary = {
        'phase': '2b',
        'method': 'Edge2Vec',
        'network': 'IID (kidney >= 1) with edge types',
        'n_edge_types': n_types,
        'BA_mean': mean_ba,
        'BA_std': std_ba,
        'AUROC_mean': mean_auroc,
        'AUROC_std': std_auroc,
        'n_genes': n_genes,
        'n_features': n_features
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_DIR, 'phase2b_summary.csv'), index=False
    )

    # Load Phase 2a results for comparison
    phase2a_summary = os.path.join(project_root, 'experiments/11_node2vec_baseline/phase2a_summary.csv')
    if os.path.exists(phase2a_summary):
        baseline = pd.read_csv(phase2a_summary).iloc[0]
        print("\n" + "=" * 60)
        print("COMPARISON: Edge2Vec (Y) vs Node2Vec (X)")
        print("=" * 60)
        print(f"Phase 2a (Node2Vec): BA = {baseline['BA_mean']:.4f} +/- {baseline['BA_std']:.4f}")
        print(f"Phase 2b (Edge2Vec): BA = {mean_ba:.4f} +/- {std_ba:.4f}")
        diff = mean_ba - baseline['BA_mean']
        print(f"Difference (Y - X):  {diff:+.4f}")
        if diff > 0.005:
            print("=> Edge type information IMPROVES prediction!")
        elif diff < -0.005:
            print("=> Edge type information HURTS prediction")
        else:
            print("=> Results are comparable (difference < 0.5%)")

    print(f"\nResults saved.")
    return mean_ba


if __name__ == "__main__":
    main()
