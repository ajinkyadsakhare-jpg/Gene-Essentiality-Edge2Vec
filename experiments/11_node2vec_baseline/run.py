#!/usr/bin/env python3
"""
Phase 2a: Node2Vec Baseline on Raw IID (Memory Optimized)
==========================================================
Train Node2Vec on filtered IID network (kidney tissue, HELP genes).
Compare to Phase 1 to measure effect of different PPI source.

Memory optimizations:
- float32 data types
- Lazy loading of large files
- Aggressive garbage collection
- Sequential voter training (not parallel)
- Efficient graph representation
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
import networkx as nx
from gensim.models import Word2Vec
import random
import warnings
warnings.filterwarnings("ignore")

# Configuration
SEED = 42
N_FOLDS = 5
N_REPEATS = 10
N_VOTERS = 13
LEARNING_RATE = 0.1
N_ESTIMATORS = 500
BOOSTING_TYPE = 'gbdt'
EARLY_STOPPING_ROUNDS = 50

# Node2Vec params (matching HELP)
N2V_P = 1.0
N2V_Q = 1.0
N2V_DIM = 128
N2V_WALK_LENGTH = 80
N2V_NUM_WALKS = 10
N2V_WINDOW = 5
N2V_EPOCHS = 1

# Paths
DATA_DIR = os.path.join(project_root, 'data/raw')
PROCESSED_DIR = os.path.join(project_root, 'data/processed')
RESULTS_DIR = os.path.dirname(__file__)


class SplittingVotingEnsembleLGBM:
    """sveLGBM with early stopping for faster training (memory optimized)."""

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


def get_help_genes_minimal():
    """Get HELP genes by only loading index columns (memory efficient)."""
    features_dir = os.path.join(DATA_DIR, 'help_features')

    # Only load index columns
    bio_genes = set(pd.read_csv(os.path.join(features_dir, 'Kidney_BIO.csv'), usecols=[0]).iloc[:, 0])
    labels_genes = set(pd.read_csv(os.path.join(features_dir, 'Kidney_HELP.csv'), usecols=[0]).iloc[:, 0])
    n2v_genes = set(pd.read_csv(os.path.join(features_dir, 'Kidney_EmbN2V_128.csv'), usecols=[0]).iloc[:, 0])

    # CCcfs is large - just load first column
    ccfs_genes = set(pd.read_csv(os.path.join(features_dir, 'Kidney_CCcfs.csv'), usecols=[0], nrows=None).iloc[:, 0])

    common = bio_genes & ccfs_genes & n2v_genes & labels_genes
    print(f"  HELP gene set: {len(common)} genes")

    del bio_genes, ccfs_genes, n2v_genes, labels_genes
    gc.collect()

    return common


def step1_process_iid(gene_set):
    """Process IID network for kidney tissue (memory optimized)."""
    print("\n" + "=" * 60)
    print("Step 1: Processing IID Network")
    print("=" * 60)

    iid_path = os.path.join(DATA_DIR, 'iid/human_annotated_PPIs.txt')
    output_dir = os.path.join(PROCESSED_DIR, 'graphs/kidney')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading IID (only needed columns)...")
    df = pd.read_csv(iid_path, sep='\t',
                     usecols=['symbol1', 'symbol2', 'evidence_type', 'kidney'],
                     low_memory=True)
    print(f"  Total interactions: {len(df)}")

    # Filter by kidney
    df = df[df['kidney'] >= 1.0].copy()
    print(f"  After kidney filter (>=1): {len(df)}")

    # Filter by genes
    mask = df['symbol1'].isin(gene_set) & df['symbol2'].isin(gene_set)
    df = df[mask].copy()
    genes_in_network = set(df['symbol1']) | set(df['symbol2'])
    print(f"  After gene filter: {len(df)} edges, {len(genes_in_network)} genes")

    # Remove duplicates
    df = df.drop_duplicates(subset=['symbol1', 'symbol2'])
    print(f"  After dedup: {len(df)} edges")

    # Save processed network
    network_path = os.path.join(output_dir, 'kidney_network.csv')
    df[['symbol1', 'symbol2']].to_csv(network_path, index=False)
    print(f"  Saved: {network_path}")

    return df, network_path


def step2_train_node2vec(network_df):
    """Train Node2Vec with memory-efficient implementation."""
    print("\n" + "=" * 60)
    print("Step 2: Training Node2Vec")
    print("=" * 60)

    # Build graph efficiently
    print("Building graph...")
    G = nx.Graph()
    for _, row in network_df.iterrows():
        G.add_edge(str(row['symbol1']), str(row['symbol2']))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Memory-efficient Node2Vec
    random.seed(SEED)
    np.random.seed(SEED)

    nodes = list(G.nodes())

    # Precompute neighbors (more memory efficient than full alias tables)
    print("Precomputing neighbors...")
    neighbors = {node: list(G.neighbors(node)) for node in nodes}

    # Generate walks
    print(f"Generating walks ({N2V_NUM_WALKS} per node, length {N2V_WALK_LENGTH})...")
    walks = []

    for walk_iter in range(N2V_NUM_WALKS):
        print(f"  Walk iteration {walk_iter + 1}/{N2V_NUM_WALKS}")
        random.shuffle(nodes)

        for node in nodes:
            walk = [node]
            while len(walk) < N2V_WALK_LENGTH:
                cur = walk[-1]
                cur_neighbors = neighbors.get(cur, [])
                if not cur_neighbors:
                    break
                # For p=1, q=1, uniform random walk
                next_node = random.choice(cur_neighbors)
                walk.append(next_node)
            walks.append(walk)

    print(f"  Generated {len(walks)} walks")

    # Free graph memory
    del G, neighbors
    gc.collect()

    # Train Word2Vec
    print("Training Word2Vec...")
    model = Word2Vec(
        sentences=walks,
        vector_size=N2V_DIM,
        window=N2V_WINDOW,
        min_count=1,
        sg=1,
        workers=2,  # Reduced for memory
        epochs=N2V_EPOCHS,
        seed=SEED
    )

    # Free walks
    del walks
    gc.collect()

    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = {}
    for node in model.wv.key_to_index:
        embeddings[node] = model.wv[node].astype(np.float32)

    # Convert to DataFrame
    data = []
    for node, emb in embeddings.items():
        row = {'node': node}
        for i, val in enumerate(emb):
            row[f'Node2Vec_{i}'] = val
        data.append(row)

    n2v_df = pd.DataFrame(data).set_index('node')
    print(f"  Embeddings: {n2v_df.shape}")

    # Save
    output_dir = os.path.join(PROCESSED_DIR, 'embeddings')
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, 'kidney_node2vec_iid.csv')
    n2v_df.to_csv(emb_path)
    print(f"  Saved: {emb_path}")

    del model, embeddings, data
    gc.collect()

    return n2v_df


def step3_build_features(n2v_df):
    """Build feature matrix (Bio + CCcfs + N2V) with memory optimization."""
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
    common_genes = sorted(set(bio.index) & set(n2v_df.index) & set(labels.index))
    print(f"  Common genes: {len(common_genes)}")

    # Subset to common genes
    bio = bio.loc[common_genes]
    n2v_df = n2v_df.loc[common_genes]
    labels = labels.loc[common_genes]

    # Scale Bio
    print("Scaling Bio...")
    bio_scaled = pd.DataFrame(
        StandardScaler().fit_transform(bio).astype(np.float32),
        index=bio.index, columns=bio.columns
    )
    del bio
    gc.collect()

    # Load CCcfs in chunks (largest file)
    print("Loading CCcfs features...")
    ccfs_path = os.path.join(features_dir, 'Kidney_CCcfs.csv')

    # Load without dtype spec (index column is strings)
    ccfs = pd.read_csv(ccfs_path, index_col=0)
    ccfs = ccfs.loc[ccfs.index.intersection(common_genes)].astype(np.float32)

    # Remove constant columns
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
    common_genes = sorted(set(bio_scaled.index) & set(ccfs_scaled.index) & set(n2v_df.index))
    print(f"  Final common genes: {len(common_genes)}")

    # Concatenate features
    print("Concatenating features...")
    X = pd.concat([
        bio_scaled.loc[common_genes],
        n2v_df.loc[common_genes],
        ccfs_scaled.loc[common_genes]
    ], axis=1)

    y = labels.loc[common_genes].replace({'E': 1, 'aE': 0, 'sNE': 0}).values.ravel()

    print(f"  Final X: {X.shape}")
    print(f"  Class distribution: E={np.sum(y==1)}, NE={np.sum(y==0)}")

    del bio_scaled, ccfs_scaled, n2v_df, labels
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
    print("Phase 2a: Node2Vec Baseline on Raw IID")
    print("=" * 60)
    print(f"Goal: Establish baseline X for Edge2Vec comparison")

    # Get HELP genes (minimal memory)
    print("\nGetting HELP gene set...")
    gene_set = get_help_genes_minimal()

    # Step 1: Process IID
    network_df, network_path = step1_process_iid(gene_set)
    n_edges = len(network_df)

    del gene_set
    gc.collect()

    # Step 2: Train Node2Vec
    n2v_df = step2_train_node2vec(network_df)

    del network_df
    gc.collect()

    # Step 3: Build features
    X, y, n_genes = step3_build_features(n2v_df)
    n_features = X.shape[1]

    # Step 4: Run CV
    ba_scores, auroc_scores = step4_run_cv(X, y)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS - Phase 2a (Node2Vec Baseline)")
    print("=" * 60)

    mean_ba = ba_scores.mean()
    std_ba = ba_scores.std()
    mean_auroc = auroc_scores.mean()
    std_auroc = auroc_scores.std()

    print(f"Balanced Accuracy: {mean_ba:.4f} +/- {std_ba:.4f}")
    print(f"AUROC:             {mean_auroc:.4f} +/- {std_auroc:.4f}")
    print()
    print(f"This is baseline X for Edge2Vec comparison (Phase 2b)")

    # Save results
    pd.DataFrame({'BA': ba_scores, 'AUROC': auroc_scores}).to_csv(
        os.path.join(RESULTS_DIR, 'phase2a_results.csv'), index=False
    )

    summary = {
        'phase': '2a',
        'method': 'Node2Vec',
        'network': 'IID (kidney >= 1)',
        'BA_mean': mean_ba,
        'BA_std': std_ba,
        'AUROC_mean': mean_auroc,
        'AUROC_std': std_auroc,
        'n_genes': n_genes,
        'n_edges': n_edges,
        'n_features': n_features
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_DIR, 'phase2a_summary.csv'), index=False
    )

    print(f"\nResults saved.")
    return mean_ba


if __name__ == "__main__":
    main()
