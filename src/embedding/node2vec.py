#!/usr/bin/env python3
"""
Node2Vec Implementation
=======================
Node2Vec embeddings matching HELP paper parameters.

Parameters from HELP code (embedding.py):
- p = 1.0
- q = 1.0
- dimensions = 128
- walk_number = 10
- walk_length = 80
- window_size = 5
- min_count = 1
- epochs = 1

Reference:
    Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks"
    KDD 2016
"""

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class Node2Vec:
    """
    Node2Vec implementation with p,q biased random walks.

    Matches HELP pipeline parameters for reproducibility.
    """

    def __init__(self,
                 p: float = 1.0,
                 q: float = 1.0,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 window_size: int = 5,
                 min_count: int = 1,
                 epochs: int = 1,
                 workers: int = 4,
                 seed: int = 42):
        """
        Initialize Node2Vec.

        Args:
            p: Return parameter (1.0 = no bias toward returning)
            q: In-out parameter (1.0 = no bias toward BFS/DFS)
            dimensions: Embedding dimension
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            window_size: Word2Vec context window
            min_count: Minimum word frequency in Word2Vec
            epochs: Word2Vec training epochs
            workers: Parallel workers
            seed: Random seed
        """
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed

        self.graph = None
        self.model = None
        self.embeddings = None

    def fit(self, graph: nx.Graph, verbose: bool = True) -> 'Node2Vec':
        """
        Fit Node2Vec on the graph.

        Args:
            graph: NetworkX graph (undirected)
            verbose: Print progress

        Returns:
            self
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.graph = graph

        if verbose:
            print(f"Fitting Node2Vec...")
            print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
            print(f"  Parameters: p={self.p}, q={self.q}, dim={self.dimensions}")
            print(f"  Walk length: {self.walk_length}, Num walks: {self.num_walks}")

        # Precompute transition probabilities
        if verbose:
            print("  Precomputing transition probabilities...")
        self._precompute_transition_probs()

        # Generate walks
        if verbose:
            print("  Generating random walks...")
        walks = self._generate_walks(verbose)

        if verbose:
            print(f"  Generated {len(walks)} walks")

        # Train Word2Vec
        if verbose:
            print("  Training Word2Vec...")

        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            sg=1,  # Skip-gram
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )

        # Extract embeddings
        self.embeddings = {node: self.model.wv[node] for node in self.model.wv.key_to_index}

        if verbose:
            print(f"  Embedded {len(self.embeddings)} nodes")

        return self

    def _precompute_transition_probs(self):
        """Precompute alias sampling tables for efficient biased walks."""
        self.alias_nodes = {}
        self.alias_edges = {}

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                probs = [self.graph[node][nbr].get('weight', 1.0) for nbr in neighbors]
                norm = sum(probs)
                probs = [p / norm for p in probs]
                self.alias_nodes[node] = (neighbors, probs)

        for edge in self.graph.edges():
            self._get_edge_alias(edge[0], edge[1])
            self._get_edge_alias(edge[1], edge[0])

    def _get_edge_alias(self, src, dst):
        """Compute alias table for edge (src, dst)."""
        neighbors = list(self.graph.neighbors(dst))
        if not neighbors:
            return

        probs = []
        for nbr in neighbors:
            weight = self.graph[dst][nbr].get('weight', 1.0)

            if nbr == src:
                # Return to previous node
                probs.append(weight / self.p)
            elif self.graph.has_edge(nbr, src):
                # Stay at same distance
                probs.append(weight)
            else:
                # Move further away
                probs.append(weight / self.q)

        norm = sum(probs)
        probs = [p / norm for p in probs]
        self.alias_edges[(src, dst)] = (neighbors, probs)

    def _generate_walks(self, verbose: bool = True) -> List[List[str]]:
        """Generate random walks from all nodes."""
        walks = []
        nodes = list(self.graph.nodes())

        for walk_iter in range(self.num_walks):
            if verbose:
                print(f"    Walk iteration {walk_iter + 1}/{self.num_walks}")

            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(node)
                walks.append(walk)

        return walks

    def _random_walk(self, start_node) -> List[str]:
        """Perform a single biased random walk."""
        walk = [str(start_node)]

        while len(walk) < self.walk_length:
            cur = walk[-1]

            if cur not in self.alias_nodes:
                break

            neighbors, probs = self.alias_nodes[cur]

            if len(walk) == 1:
                # First step: sample from node distribution
                next_node = np.random.choice(neighbors, p=probs)
            else:
                # Subsequent steps: use edge-biased sampling
                prev = walk[-2]
                if (prev, cur) in self.alias_edges:
                    neighbors, probs = self.alias_edges[(prev, cur)]
                    next_node = np.random.choice(neighbors, p=probs)
                else:
                    next_node = np.random.choice(neighbors, p=probs)

            walk.append(str(next_node))

        return walk

    def get_embedding(self, node: str) -> Optional[np.ndarray]:
        """Get embedding for a single node."""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.embeddings.get(str(node))

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all embeddings as dictionary."""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.embeddings

    def get_embeddings_df(self) -> pd.DataFrame:
        """Get embeddings as DataFrame (index=node, columns=dim_0...dim_n)."""
        if self.embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        data = []
        for node, emb in self.embeddings.items():
            row = {'node': node}
            for i, val in enumerate(emb):
                row[f'Node2Vec_{i}'] = val
            data.append(row)

        df = pd.DataFrame(data)
        df = df.set_index('node')
        return df

    def save(self, path: str):
        """Save embeddings to file (Word2Vec format)."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        self.model.wv.save_word2vec_format(path)

    def save_csv(self, path: str):
        """Save embeddings to CSV (matching HELP format)."""
        df = self.get_embeddings_df()
        df.to_csv(path)
        print(f"Saved embeddings to {path}")


def load_ppi_as_graph(edge_file: str,
                      symbol1_col: str = 'symbol1',
                      symbol2_col: str = 'symbol2',
                      weight_col: Optional[str] = None) -> nx.Graph:
    """
    Load PPI network as NetworkX graph.

    Args:
        edge_file: Path to edge list CSV
        symbol1_col: Column name for source node
        symbol2_col: Column name for target node
        weight_col: Optional column for edge weight

    Returns:
        NetworkX undirected graph
    """
    df = pd.read_csv(edge_file)

    G = nx.Graph()

    for _, row in df.iterrows():
        src = str(row[symbol1_col])
        dst = str(row[symbol2_col])
        weight = row[weight_col] if weight_col else 1.0
        G.add_edge(src, dst, weight=weight)

    return G


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Node2Vec embeddings")
    parser.add_argument('--input', required=True, help='Input edge list CSV')
    parser.add_argument('--output', required=True, help='Output embeddings CSV')
    parser.add_argument('--p', type=float, default=1.0, help='Return parameter')
    parser.add_argument('--q', type=float, default=1.0, help='In-out parameter')
    parser.add_argument('--dimensions', type=int, default=128, help='Dimensions')
    parser.add_argument('--walk-length', type=int, default=80, help='Walk length')
    parser.add_argument('--num-walks', type=int, default=10, help='Walks per node')
    parser.add_argument('--window-size', type=int, default=5, help='Window size')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')

    args = parser.parse_args()

    # Load graph
    G = load_ppi_as_graph(args.input)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Train Node2Vec
    n2v = Node2Vec(
        p=args.p,
        q=args.q,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window_size=args.window_size,
        epochs=args.epochs,
        seed=args.seed,
        workers=args.workers
    )

    n2v.fit(G)
    n2v.save_csv(args.output)
