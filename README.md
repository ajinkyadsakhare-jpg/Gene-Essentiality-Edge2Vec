# HELP-Edge2Vec

Replication of the HELP (Human gene Essentiality Labeling & Prediction) pipeline with Edge2Vec embeddings.

## Overview

This project replicates the HELP pipeline for predicting context-specific essential genes, with a key modification: **Edge2Vec** (edge-type-aware random walks) instead of Node2Vec for PPI network embeddings.

### Why Edge2Vec?

1. **Reproducibility**: The original HELP paper does not specify Node2Vec parameters (p, q, walk_length, etc.)
2. **Richer semantics**: STRING PPI network has multiple evidence channels that Edge2Vec can leverage
3. **Scientific contribution**: Testing whether edge type information improves essentiality prediction

## Project Structure

```
HELP-Edge2Vec/
├── config/                   # Configuration files
├── src/                      # Core reusable modules
│   ├── data/                 # Data loading and processing
│   ├── embedding/            # Node2Vec and Edge2Vec implementations
│   ├── evaluation/           # Model evaluation utilities
│   ├── features/             # Feature engineering
│   ├── labeling/             # Label generation
│   ├── model/                # Machine learning models
│   └── preprocessing/        # Data preprocessing
├── data/                     # Raw and processed data (gitignored)
├── experiments/              # Experiment scripts with results
│   ├── 10_kidney_baseline/   # Phase 1: HELP baseline verification
│   ├── 11_node2vec_baseline/ # Phase 2a: Node2Vec on IID network
│   └── 12_edge2vec_experiment/ # Phase 2b: Edge2Vec with edge types
├── results/                  # Experiment outputs
├── notebooks/                # Exploratory analysis
└── external/                 # External dependencies
```

## Setup

```bash
# Clone repository with submodules
git clone --recurse-submodules <repo-url>
cd HELP-Edge2Vec

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Experiments

Three phases were conducted:

### Phase 1: HELP Baseline Verification
Replicate HELP's kidney tissue results using original HELP PPI network.
```bash
python experiments/10_kidney_baseline/phase1_verify.py
```
**Result**: BA = 0.8821 ± 0.0092 (matches HELP paper)

### Phase 2a: Node2Vec on IID Network
Test Node2Vec embeddings on tissue-specific IID network.
```bash
python experiments/11_node2vec_baseline/run.py
```
**Result**: BA = 0.8839 ± 0.0104

### Phase 2b: Edge2Vec with Edge Types
Test whether edge type information improves prediction.
```bash
python experiments/12_edge2vec_experiment/run.py
```
**Result**: BA = 0.8750 ± 0.0117

### Conclusion
Edge type information does **not** improve essentiality prediction. Edge2Vec underperforms Node2Vec by ~0.9% in balanced accuracy.

## Data Sources

| Source | Description | Version |
|--------|-------------|---------|
| DepMap | CRISPR gene effect scores | v.23Q4 |
| STRING | Generic human PPI | v13 |
| IID | Tissue-specific PPI | - |
| GTEX | Tissue expression | - |
| HPA | Protein expression | - |
| GO | Gene Ontology | - |
| KEGG | Pathways | - |
| REACTOME | Pathways | - |
| BIOGRID | Interactions | - |
| COMPARTMENTS | Subcellular localization | - |

## References

- **HELP**: Surya et al., "HELP: A machine learning framework for context-specific human gene essentiality prediction" (2023)
- **Node2Vec**: Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks", KDD 2016
- **IID Database**: Kotlyar et al., "Integrated Interactions Database", Nucleic Acids Research 2016
- **DepMap**: Dempster et al., "Chronos: a cell population dynamics model of CRISPR experiments", Nature Genetics 2021

## Acknowledgments

This project builds upon the work of:
- **HELP authors** (Granata et al., ICAR-CNR) - Original framework
- **Zheng Gao et al.** - edge2vec algorithm
- **ICAR-CNR** - SVElearn ensemble method
- **Broad Institute** - DepMap data
- **University of Toronto** - IID tissue-specific PPI database

See [CREDITS.md](CREDITS.md) for complete citations and license information.

## Citation

If you use this code, please cite the original HELP paper and this repository:

```bibtex
@article{Granata2024,
  author = {Ilaria Granata and Lucia Maddalena and Mario Manzo and Mario Rosario Guarracino and Maurizio Giordano},
  title = {HELP: A computational framework for labelling and predicting human context-specific essential genes},
  year = {2024},
  doi = {10.1101/2024.04.16.589691},
  journal = {bioRxiv}
}
```

## License

MIT License

Note: `external/icarlearn/` is GPL v3.0 licensed, `external/edge2vec/` is BSD 3-Clause. See [CREDITS.md](CREDITS.md) for details.
