# Credits and Acknowledgments

This project builds upon the work of many researchers, developers, and organizations. We gratefully acknowledge the following contributions:

---

## Primary Research

### HELP Framework
**Authors**: Ilaria Granata, Lucia Maddalena, Mario Manzo, Mario Rosario Guarracino, Maurizio Giordano
**Institution**: High Performance Computing and Networking Institute (ICAR), Italian National Council of Research (CNR)
**Paper**: "HELP: A computational framework for labelling and predicting human context-specific essential genes"
**DOI**: 10.1101/2024.04.16.589691
**Citation**:
```bibtex
@article{Granata2024,
  author = {Ilaria Granata and Lucia Maddalena and Mario Manzo and Mario Rosario Guarracino and Maurizio Giordano},
  title = {HELP: A computational framework for labelling and predicting human context-specific essential genes},
  year = {2024},
  doi = {10.1101/2024.04.16.589691},
  journal = {bioRxiv}
}
```

This project replicates and extends the HELP pipeline with Edge2Vec embeddings.

---

## Software Dependencies

### SVElearn (Splitting Voting Ensemble)
**Location**: `external/icarlearn/`
**Authors**: Maurizio Giordano, Ilaria Granata
**Institution**: ICAR-CNR
**License**: GNU General Public License v3.0
**Zenodo DOI**: 10.5281/zenodo.10964743
**Description**: Machine learning ensemble method for imbalanced datasets, used for gene essentiality classification
**Repository**: https://github.com/giordamaug/SVElearn

### edge2vec
**Location**: `external/edge2vec/`
**Author**: Zheng Gao (gao27@indiana.edu)
**License**: BSD 3-Clause License
**Paper**: Gao et al., "edge2vec: Representation learning using edge semantics for biomedical knowledge discovery", BMC Bioinformatics 2019
**DOI**: 10.1186/s12859-019-2914-2
**Description**: Edge-type-aware graph embedding algorithm
**Repository**: https://github.com/zheng-gao/edge2vec
**Citation**:
```bibtex
@article{Gao2019,
  author = {Gao, Zheng and Fu, Gang and Ouyang, Chunping and Tsutsui, Satoshi and Liu, Xiaozhong and Yang, Jeremy and Gessner, Christopher and others},
  title = {edge2vec: Representation learning using edge semantics for biomedical knowledge discovery},
  journal = {BMC Bioinformatics},
  volume = {20},
  number = {1},
  pages = {306},
  year = {2019},
  doi = {10.1186/s12859-019-2914-2}
}
```

### node2vec
**Authors**: Aditya Grover, Jure Leskovec
**Paper**: "node2vec: Scalable Feature Learning for Networks", KDD 2016
**Description**: Graph embedding using biased random walks
**Citation**:
```bibtex
@inproceedings{Grover2016,
  author = {Grover, Aditya and Leskovec, Jure},
  title = {node2vec: Scalable Feature Learning for Networks},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2016},
  pages = {855--864},
  doi = {10.1145/2939672.2939754}
}
```

---

## Data Sources

### DepMap (Dependency Map)
**Institution**: Broad Institute
**Version**: v23Q4 (assumed)
**Description**: CRISPR gene effect scores for gene essentiality labels
**URL**: https://depmap.org/
**Paper**: Dempster et al., "Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects", Nature Genetics 2021
**Citation**:
```bibtex
@article{Dempster2021,
  author = {Dempster, Joshua M. and others},
  title = {Chronos: a cell population dynamics model of CRISPR experiments that improves inference of gene fitness effects},
  journal = {Nature Genetics},
  year = {2021},
  doi = {10.1038/s41588-021-00944-6}
}
```

### IID (Integrated Interactions Database)
**Authors**: Kotlyar et al.
**Description**: Tissue-specific protein-protein interactions
**URL**: http://iid.ophid.utoronto.ca/
**Paper**: "Integrated Interactions Database: Tissue-specific view of the human and model organism interactomes", Nucleic Acids Research 2016
**Citation**:
```bibtex
@article{Kotlyar2016,
  author = {Kotlyar, Max and Pastrello, Chiara and Malik, Zarko and Jurisica, Igor},
  title = {IID 2018 update: context-specific physical protein--protein interactions in human, model organisms and domesticated species},
  journal = {Nucleic Acids Research},
  volume = {47},
  number = {D1},
  pages = {D581--D589},
  year = {2019},
  doi = {10.1093/nar/gky1037}
}
```

### HELP Zenodo Dataset
**Authors**: HELP authors (Granata et al.)
**Description**: Preprocessed features (Bio, CCcfs, EmbN2V) and PPI networks
**Location**: `data/raw/help_zenodo/`, `data/raw/help_features/`, `data/raw/help_ppi/`
**Note**: Used for Phase 1 baseline verification

### STRING Database
**Description**: Protein-protein interaction database (referenced in methodology)
**Version**: v13
**URL**: https://string-db.org/

### GTEx (Genotype-Tissue Expression)
**Description**: Tissue expression data (referenced for future work)
**URL**: https://gtexportal.org/

### Human Protein Atlas (HPA)
**Description**: Protein expression data (referenced for future work)
**URL**: https://www.proteinatlas.org/

---

## Python Libraries

This project relies on the following open-source Python libraries:

- **NumPy**: Numerical computing (BSD License)
- **pandas**: Data manipulation (BSD License)
- **SciPy**: Scientific computing (BSD License)
- **scikit-learn**: Machine learning utilities (BSD License)
- **LightGBM**: Gradient boosting framework (MIT License)
- **NetworkX**: Graph analysis (BSD License)
- **gensim**: Word2Vec/Skip-gram embeddings (LGPL License)
- **PyYAML**: YAML parsing (MIT License)
- **tqdm**: Progress bars (MIT/MPL License)
- **joblib**: Pipeline persistence (BSD License)
- **Optuna**: Hyperparameter optimization (MIT License)

---

## Acknowledgments

- **HELP authors** for making their data and methodology publicly available
- **ICAR-CNR** for developing the SVElearn ensemble method
- **Zheng Gao and collaborators** for the edge2vec algorithm
- **Broad Institute** for DepMap CRISPR screening data
- **University of Toronto** for the IID tissue-specific PPI database
- All contributors to the open-source scientific Python ecosystem

---

## License Compliance

This project includes code from:
- `external/edge2vec/` - BSD 3-Clause License (Copyright 2019 Zheng Gao)
- `external/icarlearn/` - GNU GPL v3.0 (Copyright ICAR-CNR)

All original code in this repository is released under the MIT License, except where noted. The inclusion of GPL-licensed components (icarlearn) means that any distribution of this software must comply with GPL v3.0 terms.

---

## How to Cite This Work

If you use this repository, please cite:
1. The original HELP paper (Granata et al., 2024)
2. The edge2vec paper (Gao et al., 2019)
3. This repository (with appropriate GitHub citation)

```bibtex
@misc{HELPEdge2Vec2026,
  title = {HELP-Edge2Vec: Testing Edge-Type-Aware Embeddings for Gene Essentiality Prediction},
  author = {[Your Name]},
  year = {2026},
  url = {[Your GitHub URL]}
}
```

---

**Last Updated**: January 2026
