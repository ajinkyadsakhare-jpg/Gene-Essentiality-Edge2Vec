#!/usr/bin/env python3
"""
Edge2Vec - Edge Type Aware Random Walks
========================================
Python 3 port of the original Edge2Vec implementation.

Original: https://github.com/RoyZhengGao/edge2vec

This module implements edge-type-aware random walks that use a learned
transition matrix to weight transitions between different edge types.

Two main components:
1. transition.py: Learn transition matrix via EM algorithm
2. edge2vec.py: Generate random walks using transition matrix + Word2Vec

Reference:
    Gao et al., "Edge2vec: Representation learning using edge semantics
    for biomedical knowledge discovery" BMC Bioinformatics (2019)
"""

import argparse
import os
import random
import math
import numpy as np
import networkx as nx
from scipy import stats
from gensim.models import Word2Vec
from typing import List, Dict, Tuple, Optional


# ============================================================================
# TRANSITION MATRIX LEARNING (EM Algorithm)
# ============================================================================

def initialize_transition_matrix(n_types: int) -> np.ndarray:
    """Initialize uniform transition matrix."""
    init_val = 1.0 / (n_types * n_types)
    return np.full((n_types, n_types), init_val)


def simulate_walks_for_em(G: nx.Graph,
                          num_walks: int,
                          walk_length: int,
                          matrix: np.ndarray,
                          p: float = 1.0,
                          q: float = 1.0,
                          verbose: bool = False) -> List[List[str]]:
    """
    Generate random walks for EM transition matrix learning.

    Returns walks as sequences of edge types (for computing co-occurrence).
    """
    walks = []
    links = list(G.edges(data=True))

    for walk_iter in range(num_walks):
        if verbose:
            print(f'  Walk iteration: {walk_iter + 1}/{num_walks}')

        random.shuffle(links)
        count = min(1000, len(links))  # Limit for efficiency

        for link in links[:count]:
            walk = edge2vec_walk_em(G, walk_length, link, matrix, p, q)
            walks.append(walk)

    return walks


def edge2vec_walk_em(G: nx.Graph,
                     walk_length: int,
                     start_link: Tuple,
                     matrix: np.ndarray,
                     p: float,
                     q: float) -> List[str]:
    """
    Single edge-type-aware random walk for EM (returns edge types).
    """
    walk = [start_link]
    result = [str(start_link[2]['type'])]

    while len(walk) < walk_length:
        cur = walk[-1]
        start_node, end_node = cur[0], cur[1]
        cur_edge_type = cur[2]['type']

        # Determine walk direction based on node degrees (hub avoidance)
        start_prob = 1.0 / G.degree(start_node)
        end_prob = 1.0 / G.degree(end_node)
        prob = start_prob / (start_prob + end_prob)

        if np.random.rand() < prob:
            direction_node = start_node
            left_node = end_node
        else:
            direction_node = end_node
            left_node = start_node

        # Get neighbors and compute transition weights
        neighbors = list(G.neighbors(direction_node))
        if not neighbors:
            break

        # Calculate weighted probabilities
        distance_sum = 0
        neighbor_weights = []

        for neighbor in neighbors:
            neighbor_link = G[direction_node][neighbor]
            neighbor_type = neighbor_link['type']
            neighbor_weight = neighbor_link.get('weight', 1.0)
            trans_weight = matrix[cur_edge_type - 1][neighbor_type - 1]

            # p/q biasing (similar to Node2Vec)
            if G.has_edge(neighbor, left_node):
                weight = trans_weight * neighbor_weight / p  # Return
            elif neighbor == left_node:
                weight = trans_weight * neighbor_weight  # Stay
            else:
                weight = trans_weight * neighbor_weight / q  # Explore

            neighbor_weights.append((neighbor, neighbor_link, weight))
            distance_sum += weight

        if distance_sum <= 0:
            break

        # Sample next neighbor
        rand = np.random.rand() * distance_sum
        threshold = 0
        next_neighbor = None
        next_link = None

        for neighbor, link, weight in neighbor_weights:
            threshold += weight
            if threshold >= rand:
                next_neighbor = neighbor
                next_link = link
                break

        if next_neighbor is None:
            break

        # Add to walk
        walk.append((direction_node, next_neighbor, next_link))
        result.append(str(next_link['type']))

    return result


def update_transition_matrix(walks: List[List[str]],
                             n_types: int,
                             metric: int = 1) -> np.ndarray:
    """
    E-step: Update transition matrix based on walk statistics.

    Metrics:
    1 = Wilcoxon test (default)
    2 = Entropy
    3 = Spearman correlation
    4 = Pearson correlation
    """
    matrix = np.zeros((n_types, n_types))
    repo = {i: [] for i in range(n_types)}

    # Collect type counts per walk
    for walk in walks:
        curr_repo = {}
        for edge in walk:
            edge_id = int(edge) - 1
            curr_repo[edge_id] = curr_repo.get(edge_id, 0) + 1

        for i in range(n_types):
            repo[i].append(curr_repo.get(i, 0))

    # Compute similarity matrix
    for i in range(n_types):
        for j in range(n_types):
            if metric == 1:
                matrix[i][j] = _wilcoxon_similarity(repo[i], repo[j])
            elif metric == 2:
                matrix[i][j] = _entropy_similarity(repo[i], repo[j])
            elif metric == 3:
                matrix[i][j] = _spearman_similarity(repo[i], repo[j])
            elif metric == 4:
                matrix[i][j] = _pearson_similarity(repo[i], repo[j])
            else:
                raise ValueError(f"Invalid metric: {metric}")

    return matrix


def _wilcoxon_similarity(v1: List[int], v2: List[int]) -> float:
    """Wilcoxon test similarity (smaller = more similar)."""
    try:
        result = stats.wilcoxon(v1, v2).statistic
        if np.isnan(result):
            result = 0
        return 1 / (math.sqrt(result) + 1)
    except:
        return 0


def _entropy_similarity(v1: List[int], v2: List[int]) -> float:
    """Entropy-based similarity."""
    try:
        result = stats.entropy(v1, v2)
        if np.isnan(result):
            result = 0
        return result
    except:
        return 0


def _spearman_similarity(v1: List[int], v2: List[int]) -> float:
    """Spearman correlation similarity."""
    try:
        result = stats.spearmanr(v1, v2).correlation
        if np.isnan(result):
            result = -1
        return 1 / (1 + math.exp(-result))  # Sigmoid
    except:
        return 0.5


def _pearson_similarity(v1: List[int], v2: List[int]) -> float:
    """Pearson correlation similarity."""
    try:
        result = stats.pearsonr(v1, v2)[0]
        if np.isnan(result):
            result = -1
        return 1 / (1 + math.exp(-result))  # Sigmoid
    except:
        return 0.5


def learn_transition_matrix(G: nx.Graph,
                            n_types: int,
                            em_iterations: int = 5,
                            num_walks: int = 2,
                            walk_length: int = 3,
                            p: float = 1.0,
                            q: float = 1.0,
                            metric: int = 1,
                            verbose: bool = True) -> np.ndarray:
    """
    Learn transition matrix via EM algorithm.

    Args:
        G: NetworkX graph with 'type' edge attribute
        n_types: Number of edge types
        em_iterations: Number of EM iterations
        num_walks: Walks per edge for E-step
        walk_length: Length of each walk
        p: Return parameter
        q: In-out parameter
        metric: Similarity metric (1=Wilcoxon, 2=Entropy, 3=Spearman, 4=Pearson)
        verbose: Print progress

    Returns:
        Learned transition matrix (n_types x n_types)
    """
    if verbose:
        print(f"Learning transition matrix ({em_iterations} EM iterations)...")

    matrix = initialize_transition_matrix(n_types)

    for i in range(em_iterations):
        if verbose:
            print(f"\nEM Iteration {i + 1}/{em_iterations}")

        # M-step: Generate walks with current matrix
        walks = simulate_walks_for_em(G, num_walks, walk_length, matrix, p, q, verbose)

        # E-step: Update matrix based on walks
        matrix = update_transition_matrix(walks, n_types, metric)

        if verbose:
            print(f"  Updated transition matrix:\n{matrix}")

    return matrix


# ============================================================================
# EDGE2VEC EMBEDDING (Random Walks + Word2Vec)
# ============================================================================

def read_graph(edge_list_path: str,
               weighted: bool = False,
               directed: bool = False) -> nx.Graph:
    """
    Read graph from edge list file.

    Format (space-separated):
    - Unweighted: node1 node2 type edge_id
    - Weighted: node1 node2 type weight edge_id
    """
    if weighted:
        G = nx.read_edgelist(
            edge_list_path,
            nodetype=str,
            data=(('type', int), ('weight', float), ('id', int)),
            create_using=nx.DiGraph()
        )
    else:
        G = nx.read_edgelist(
            edge_list_path,
            nodetype=str,
            data=(('type', int), ('id', int)),
            create_using=nx.DiGraph()
        )
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0

    if not directed:
        G = G.to_undirected()

    return G


def simulate_walks(G: nx.Graph,
                   num_walks: int,
                   walk_length: int,
                   matrix: np.ndarray,
                   p: float = 1.0,
                   q: float = 1.0,
                   verbose: bool = True,
                   workers: int = 4) -> List[List[str]]:
    """
    Generate node walks for Word2Vec embedding.
    Memory-optimized single-threaded version with pre-computed neighbor lookup.

    Returns walks as sequences of node IDs.
    """
    import gc

    # Pre-compute neighbor information for fast lookup
    if verbose:
        print("  Pre-computing neighbor structure...")

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Build neighbor arrays (lists of (neighbor_idx, edge_type, weight))
    neighbors = [[] for _ in range(n_nodes)]
    neighbor_set = [set() for _ in range(n_nodes)]  # For fast has_edge check

    for u, v, d in G.edges(data=True):
        u_idx, v_idx = node_to_idx[u], node_to_idx[v]
        edge_type = d['type']
        weight = d.get('weight', 1.0)

        neighbors[u_idx].append((v_idx, edge_type, weight))
        neighbors[v_idx].append((u_idx, edge_type, weight))
        neighbor_set[u_idx].add(v_idx)
        neighbor_set[v_idx].add(u_idx)

    if verbose:
        print(f"  Structure built. Starting walks...")

    walks = []
    for walk_iter in range(num_walks):
        if verbose:
            print(f'  Walk iteration: {walk_iter + 1}/{num_walks}')

        random.shuffle(nodes)

        for start_node in nodes:
            walk = _fast_edge2vec_walk(
                start_node, node_to_idx, nodes, neighbors, neighbor_set,
                walk_length, matrix, p, q
            )
            walks.append(walk)

        # Periodic garbage collection
        if walk_iter % 2 == 0:
            gc.collect()

    return walks


def _fast_edge2vec_walk(start_node: str,
                        node_to_idx: Dict,
                        nodes: List[str],
                        neighbors: List[List],
                        neighbor_set: List[set],
                        walk_length: int,
                        matrix: np.ndarray,
                        p: float,
                        q: float) -> List[str]:
    """
    Fast single random walk using pre-computed neighbor structure.
    """
    walk = [start_node]
    cur_idx = node_to_idx[start_node]

    while len(walk) < walk_length:
        cur_nbrs = neighbors[cur_idx]
        if not cur_nbrs:
            break

        if len(walk) == 1:
            # First step: uniform random
            next_idx, _, _ = cur_nbrs[int(np.random.rand() * len(cur_nbrs))]
            walk.append(nodes[next_idx])
            cur_idx = next_idx
        else:
            prev_idx = node_to_idx[walk[-2]]
            prev_edge_type = None
            # Find edge type from prev to cur
            for nbr_idx, et, _ in neighbors[prev_idx]:
                if nbr_idx == cur_idx:
                    prev_edge_type = et
                    break

            if prev_edge_type is None:
                break

            # Calculate weighted probabilities
            weights = []
            total_weight = 0.0

            for nbr_idx, nbr_type, nbr_weight in cur_nbrs:
                trans_weight = matrix[prev_edge_type - 1][nbr_type - 1]

                if nbr_idx in neighbor_set[prev_idx]:  # Can return
                    w = trans_weight * nbr_weight / p
                elif nbr_idx == prev_idx:  # Stay
                    w = trans_weight * nbr_weight
                else:  # Explore
                    w = trans_weight * nbr_weight / q

                weights.append((nbr_idx, w))
                total_weight += w

            if total_weight <= 0:
                break

            # Sample next node
            rand = np.random.rand() * total_weight
            threshold = 0
            next_idx = None

            for nbr_idx, w in weights:
                threshold += w
                if threshold >= rand:
                    next_idx = nbr_idx
                    break

            if next_idx is None:
                break

            walk.append(nodes[next_idx])
            cur_idx = next_idx

    return walk


def edge2vec_walk(G: nx.Graph,
                  walk_length: int,
                  start_node: str,
                  matrix: np.ndarray,
                  p: float,
                  q: float) -> List[str]:
    """
    Single edge-type-aware random walk from a starting node.

    Returns sequence of node IDs.
    """
    walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))
        random.shuffle(cur_nbrs)

        if not cur_nbrs:
            break

        if len(walk) == 1:
            # First step: uniform random
            next_node = cur_nbrs[int(np.random.rand() * len(cur_nbrs))]
            walk.append(next_node)
        else:
            prev = walk[-2]
            prev_edge_type = G[prev][cur]['type']

            # Calculate weighted probabilities
            distance_sum = 0
            for neighbor in cur_nbrs:
                neighbor_link = G[cur][neighbor]
                neighbor_type = neighbor_link['type']
                neighbor_weight = neighbor_link.get('weight', 1.0)
                trans_weight = matrix[prev_edge_type - 1][neighbor_type - 1]

                # p/q biasing
                if G.has_edge(neighbor, prev):
                    distance_sum += trans_weight * neighbor_weight / p
                elif neighbor == prev:
                    distance_sum += trans_weight * neighbor_weight
                else:
                    distance_sum += trans_weight * neighbor_weight / q

            if distance_sum <= 0:
                break

            # Sample next node
            rand = np.random.rand() * distance_sum
            threshold = 0

            for neighbor in cur_nbrs:
                neighbor_link = G[cur][neighbor]
                neighbor_type = neighbor_link['type']
                neighbor_weight = neighbor_link.get('weight', 1.0)
                trans_weight = matrix[prev_edge_type - 1][neighbor_type - 1]

                if G.has_edge(neighbor, prev):
                    threshold += trans_weight * neighbor_weight / p
                elif neighbor == prev:
                    threshold += trans_weight * neighbor_weight
                else:
                    threshold += trans_weight * neighbor_weight / q

                if threshold >= rand:
                    walk.append(neighbor)
                    break

    return walk


def train_embedding(walks: List[List[str]],
                    dimensions: int = 128,
                    window_size: int = 5,
                    min_count: int = 0,
                    sg: int = 1,
                    workers: int = 4,
                    epochs: int = 5) -> Word2Vec:
    """
    Train Word2Vec on random walks.

    Args:
        walks: List of node sequences
        dimensions: Embedding dimension
        window_size: Context window size
        min_count: Minimum word frequency
        sg: Skip-gram (1) or CBOW (0)
        workers: Number of workers
        epochs: Training epochs

    Returns:
        Trained Word2Vec model
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=dimensions,
        window=window_size,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs
    )
    return model


def edge2vec_embed(edge_list_path: str,
                   output_path: str,
                   n_edge_types: int,
                   dimensions: int = 128,
                   walk_length: int = 80,
                   num_walks: int = 10,
                   window_size: int = 5,
                   p: float = 1.0,
                   q: float = 1.0,
                   em_iterations: int = 5,
                   em_walks: int = 2,
                   em_walk_length: int = 3,
                   weighted: bool = False,
                   directed: bool = False,
                   workers: int = 4,
                   epochs: int = 5,
                   seed: int = 42,
                   verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Full Edge2Vec pipeline: Learn matrix -> Generate walks -> Train Word2Vec.

    Args:
        edge_list_path: Path to edge list file
        output_path: Path to save embeddings
        n_edge_types: Number of distinct edge types
        dimensions: Embedding dimension
        walk_length: Random walk length
        num_walks: Number of walks per node
        window_size: Word2Vec window size
        p: Return parameter
        q: In-out parameter
        em_iterations: EM iterations for transition matrix
        em_walks: Walks per EM iteration
        em_walk_length: Walk length for EM
        weighted: Use edge weights
        directed: Directed graph
        workers: Parallel workers
        epochs: Word2Vec epochs
        seed: Random seed
        verbose: Print progress

    Returns:
        Dict mapping node ID -> embedding vector
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"\nEdge2Vec Embedding Pipeline")
        print("=" * 60)
        print(f"Input: {edge_list_path}")
        print(f"Edge types: {n_edge_types}")
        print(f"Dimensions: {dimensions}")
        print(f"Walk length: {walk_length}, Num walks: {num_walks}")

    # Step 1: Read graph
    if verbose:
        print(f"\n1. Reading graph...")
    G = read_graph(edge_list_path, weighted, directed)
    if verbose:
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Step 2: Learn transition matrix
    if verbose:
        print(f"\n2. Learning transition matrix...")
    matrix = learn_transition_matrix(
        G, n_edge_types,
        em_iterations=em_iterations,
        num_walks=em_walks,
        walk_length=em_walk_length,
        p=p, q=q,
        verbose=verbose
    )

    # Step 3: Generate walks
    if verbose:
        print(f"\n3. Generating random walks...")
    walks = simulate_walks(G, num_walks, walk_length, matrix, p, q, verbose, workers)
    if verbose:
        print(f"   Generated {len(walks)} walks")

    # Step 4: Train Word2Vec
    if verbose:
        print(f"\n4. Training Word2Vec...")
    model = train_embedding(
        walks,
        dimensions=dimensions,
        window_size=window_size,
        workers=workers,
        epochs=epochs
    )

    # Step 5: Save embeddings
    if verbose:
        print(f"\n5. Saving embeddings to {output_path}...")
    model.wv.save_word2vec_format(output_path)

    # Return as dict
    embeddings = {node: model.wv[node] for node in model.wv.key_to_index}

    if verbose:
        print(f"\nDone! Embedded {len(embeddings)} nodes.")

    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge2Vec: Edge-type-aware embeddings")
    parser.add_argument('--input', required=True, help='Input edge list path')
    parser.add_argument('--output', required=True, help='Output embeddings path')
    parser.add_argument('--n-types', type=int, required=True, help='Number of edge types')
    parser.add_argument('--dimensions', type=int, default=128, help='Embedding dimensions')
    parser.add_argument('--walk-length', type=int, default=80, help='Walk length')
    parser.add_argument('--num-walks', type=int, default=10, help='Walks per node')
    parser.add_argument('--window-size', type=int, default=5, help='Word2Vec window')
    parser.add_argument('--p', type=float, default=1.0, help='Return parameter')
    parser.add_argument('--q', type=float, default=1.0, help='In-out parameter')
    parser.add_argument('--em-iterations', type=int, default=5, help='EM iterations')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--epochs', type=int, default=5, help='Word2Vec epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--weighted', action='store_true', help='Use edge weights')
    parser.add_argument('--directed', action='store_true', help='Directed graph')

    args = parser.parse_args()

    edge2vec_embed(
        edge_list_path=args.input,
        output_path=args.output,
        n_edge_types=args.n_types,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        p=args.p,
        q=args.q,
        em_iterations=args.em_iterations,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.seed,
        weighted=args.weighted,
        directed=args.directed
    )
