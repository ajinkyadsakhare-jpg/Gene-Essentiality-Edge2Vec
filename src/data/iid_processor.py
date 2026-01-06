#!/usr/bin/env python3
"""
IID Database Processor
======================
Processes IID (Integrated Interactions Database) for tissue-specific PPI networks.

Functions:
- Filter by tissue (e.g., kidney >= 1)
- Filter by gene set (e.g., HELP genes)
- Extract edge types from evidence_type column
- Convert to Edge2Vec format
"""

import os
import pandas as pd
import numpy as np
from typing import Set, Optional, Tuple, Dict


def load_iid(path: str, usecols: Optional[list] = None) -> pd.DataFrame:
    """
    Load IID human_annotated_PPIs.txt file.

    Args:
        path: Path to human_annotated_PPIs.txt
        usecols: Columns to load (None = all)

    Returns:
        DataFrame with IID interactions
    """
    default_cols = ['symbol1', 'symbol2', 'evidence_type', 'kidney']
    cols = usecols or default_cols

    df = pd.read_csv(path, sep='\t', usecols=cols, low_memory=False)
    return df


def filter_by_tissue(df: pd.DataFrame, tissue: str = 'kidney',
                     min_score: float = 1.0) -> pd.DataFrame:
    """
    Filter IID interactions by tissue expression score.

    Args:
        df: IID DataFrame
        tissue: Tissue column name (lowercase)
        min_score: Minimum tissue score threshold

    Returns:
        Filtered DataFrame
    """
    if tissue not in df.columns:
        raise ValueError(f"Tissue '{tissue}' not found in columns: {df.columns.tolist()}")

    mask = df[tissue] >= min_score
    filtered = df[mask].copy()

    print(f"  Tissue filter ({tissue} >= {min_score}): {len(df)} -> {len(filtered)} edges")
    return filtered


def filter_by_genes(df: pd.DataFrame, gene_set: Set[str],
                    symbol1_col: str = 'symbol1',
                    symbol2_col: str = 'symbol2') -> pd.DataFrame:
    """
    Filter interactions to only include genes in the provided set.
    Both endpoints must be in the gene set.

    Args:
        df: IID DataFrame
        gene_set: Set of valid gene symbols
        symbol1_col: Column name for first gene
        symbol2_col: Column name for second gene

    Returns:
        Filtered DataFrame
    """
    mask = df[symbol1_col].isin(gene_set) & df[symbol2_col].isin(gene_set)
    filtered = df[mask].copy()

    # Count unique genes in filtered network
    genes_in_network = set(filtered[symbol1_col]) | set(filtered[symbol2_col])

    print(f"  Gene filter: {len(df)} -> {len(filtered)} edges")
    print(f"  Unique genes in network: {len(genes_in_network)}")

    return filtered


def parse_evidence_type(evidence_str: str) -> str:
    """
    Parse evidence_type string and return simplified type.

    Evidence types in IID:
    - exp: experimental
    - pred: predicted
    - ortho: orthologous
    - Combinations: exp|pred, exp|ortho, ortho|pred, exp|ortho|pred

    Returns single canonical type or combination.
    """
    if pd.isna(evidence_str):
        return 'unknown'

    evidence_str = str(evidence_str).lower().strip()

    # Parse individual types
    types = set()
    for part in evidence_str.split('|'):
        part = part.strip()
        if 'exp' in part:
            types.add('exp')
        elif 'pred' in part:
            types.add('pred')
        elif 'ortho' in part:
            types.add('ortho')

    if not types:
        return 'unknown'

    # Return canonical form
    return '|'.join(sorted(types))


def add_edge_types(df: pd.DataFrame,
                   evidence_col: str = 'evidence_type') -> pd.DataFrame:
    """
    Add parsed edge type column.

    Args:
        df: IID DataFrame with evidence_type column
        evidence_col: Name of evidence column

    Returns:
        DataFrame with 'edge_type' column added
    """
    df = df.copy()
    df['edge_type'] = df[evidence_col].apply(parse_evidence_type)

    # Show distribution
    type_counts = df['edge_type'].value_counts()
    print(f"  Edge type distribution:")
    for etype, count in type_counts.items():
        pct = 100 * count / len(df)
        print(f"    {etype}: {count} ({pct:.1f}%)")

    return df


def create_edge_type_mapping(df: pd.DataFrame) -> Dict[str, int]:
    """
    Create mapping from edge type strings to integer IDs (1-indexed for Edge2Vec).

    Returns:
        Dict mapping edge_type string -> integer ID
    """
    unique_types = sorted(df['edge_type'].unique())
    mapping = {etype: idx + 1 for idx, etype in enumerate(unique_types)}

    print(f"  Edge type mapping (1-indexed):")
    for etype, idx in mapping.items():
        print(f"    {idx}: {etype}")

    return mapping


def to_edge2vec_format(df: pd.DataFrame,
                       output_path: str,
                       edge_type_mapping: Dict[str, int],
                       symbol1_col: str = 'symbol1',
                       symbol2_col: str = 'symbol2',
                       weighted: bool = False) -> pd.DataFrame:
    """
    Convert IID DataFrame to Edge2Vec input format.

    Edge2Vec format (space-separated):
    - Unweighted: node1 node2 type edge_id
    - Weighted: node1 node2 type weight edge_id

    Args:
        df: Processed IID DataFrame with edge_type column
        output_path: Path to save Edge2Vec input file
        edge_type_mapping: Dict mapping edge_type -> integer
        weighted: Include weight column

    Returns:
        DataFrame in Edge2Vec format
    """
    edge2vec_df = pd.DataFrame({
        'node1': df[symbol1_col].values,
        'node2': df[symbol2_col].values,
        'type': df['edge_type'].map(edge_type_mapping).values,
        'edge_id': range(len(df))
    })

    if weighted:
        edge2vec_df['weight'] = 1.0
        # Reorder columns: node1 node2 type weight edge_id
        edge2vec_df = edge2vec_df[['node1', 'node2', 'type', 'weight', 'edge_id']]

    # Save without header (Edge2Vec format)
    edge2vec_df.to_csv(output_path, sep=' ', header=False, index=False)

    print(f"  Saved Edge2Vec format: {output_path}")
    print(f"  Edges: {len(edge2vec_df)}, Unique edge types: {edge2vec_df['type'].nunique()}")

    return edge2vec_df


def process_iid_for_tissue(iid_path: str,
                           gene_set: Set[str],
                           tissue: str = 'kidney',
                           min_tissue_score: float = 1.0,
                           output_dir: str = '.') -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Full pipeline: Load IID -> Filter -> Add types -> Convert to Edge2Vec.

    Args:
        iid_path: Path to IID human_annotated_PPIs.txt
        gene_set: Set of valid gene symbols (e.g., HELP genes)
        tissue: Tissue name for filtering
        min_tissue_score: Minimum tissue score threshold
        output_dir: Directory for output files

    Returns:
        Tuple of (processed DataFrame, edge type mapping)
    """
    print(f"\nProcessing IID for {tissue} tissue...")

    # Load
    print("\n1. Loading IID...")
    df = load_iid(iid_path, usecols=['symbol1', 'symbol2', 'evidence_type', tissue])
    print(f"  Loaded {len(df)} total interactions")

    # Filter by tissue
    print(f"\n2. Filtering by tissue ({tissue} >= {min_tissue_score})...")
    df = filter_by_tissue(df, tissue, min_tissue_score)

    # Filter by genes
    print(f"\n3. Filtering by gene set ({len(gene_set)} genes)...")
    df = filter_by_genes(df, gene_set)

    # Add edge types
    print("\n4. Parsing edge types...")
    df = add_edge_types(df)

    # Create mapping
    print("\n5. Creating edge type mapping...")
    mapping = create_edge_type_mapping(df)

    # Save processed network
    os.makedirs(output_dir, exist_ok=True)
    network_path = os.path.join(output_dir, f'{tissue}_network.csv')
    df.to_csv(network_path, index=False)
    print(f"\n6. Saved processed network: {network_path}")

    # Convert to Edge2Vec format
    print("\n7. Converting to Edge2Vec format...")
    edge2vec_path = os.path.join(output_dir, f'{tissue}_edge2vec_input.txt')
    to_edge2vec_format(df, edge2vec_path, mapping)

    # Save mapping
    mapping_path = os.path.join(output_dir, f'{tissue}_edge_type_mapping.csv')
    pd.DataFrame([{'edge_type': k, 'type_id': v} for k, v in mapping.items()]).to_csv(
        mapping_path, index=False
    )
    print(f"  Saved edge type mapping: {mapping_path}")

    return df, mapping


def get_help_genes(features_dir: str) -> Set[str]:
    """
    Get the set of genes used in HELP pipeline.
    Intersection of Bio, CCcfs, N2V, and Labels.

    Args:
        features_dir: Path to directory with HELP feature files

    Returns:
        Set of gene symbols
    """
    # Load gene lists from each file
    bio = pd.read_csv(os.path.join(features_dir, 'Kidney_BIO.csv'), index_col=0)
    ccfs = pd.read_csv(os.path.join(features_dir, 'Kidney_CCcfs.csv'), index_col=0)
    n2v = pd.read_csv(os.path.join(features_dir, 'Kidney_EmbN2V_128.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(features_dir, 'Kidney_HELP.csv'), index_col=0)

    # Find intersection
    gene_sets = [set(bio.index), set(ccfs.index), set(n2v.index), set(labels.index)]
    common_genes = gene_sets[0]
    for gs in gene_sets[1:]:
        common_genes = common_genes & gs

    print(f"  HELP gene intersection:")
    print(f"    Bio: {len(bio)}, CCcfs: {len(ccfs)}, N2V: {len(n2v)}, Labels: {len(labels)}")
    print(f"    Common: {len(common_genes)}")

    return common_genes


if __name__ == "__main__":
    # Example usage
    import sys

    # Paths (adjust as needed)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    iid_path = os.path.join(project_root, 'data/raw/iid/human_annotated_PPIs.txt')
    features_dir = os.path.join(project_root, 'data/raw/help_features')
    output_dir = os.path.join(project_root, 'data/processed/graphs/kidney')

    print("=" * 60)
    print("IID Processor - Kidney Tissue")
    print("=" * 60)

    # Get HELP genes
    print("\nLoading HELP gene set...")
    gene_set = get_help_genes(features_dir)

    # Process IID
    df, mapping = process_iid_for_tissue(
        iid_path=iid_path,
        gene_set=gene_set,
        tissue='kidney',
        min_tissue_score=1.0,
        output_dir=output_dir
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
