# SVElearn
Machine learning methods for unbalanced datasets

## Authors
- [Maurizio Giordano](https://orcid.org/0000-0001-9917-7591) and [Ilaria Granata](https://orcid.org/0000-0002-3450-4667)
- High Performance Computing and Networking (ICAR), Italian National Council of Research (CNR)

# Documentation
This package is a library of machine learning methods developed by ICAR-CNR for efficient prediction models in unbalanced data domains.

## The Splitting Voting Ensemble

The Splitting Votign Ensemble (SVE) is a meta-model designed to address classification task on unbalanced machine learning datasets.

SVE can be considered a meta-learning algorithm since it uses another learning method as a base model for all members of the ensemble to combine their predictions. 
This algorithm was designed and developed to address binary and multiclass classification tasks in data domains characterized by strong unbalancing of classes, such as Cybersecurity, Bionformatics, etc.

Before training, the method partitions the set of majority class samples into $n$ parts, and it trains each classifier on a subset of training data composed of one of these parts along with the entire set of minority class training samples. 
In multiclass scenarios, the partition ratio considered is that between the samples of the majority class and the second majority class: only the samples of the majority classs are partitioned, and each partition associated with a duplicate of the samples of the remaining classes.

<img src="https://github.com/giordamaug/SVElearn/blob/main/images/softvoting_tr.png" width="300" />

During testing on unseen data, each classifier of the ensemble produces a probability for the label prediction; we compute the final probability response of the ensemble as the average of the probabilities of the n voting classifiers. 
The number $n$ of classifiers is automatically determined by the algorithm according to the class distribution in training data, or user-specified as an input paramter (``n_voters'').

<img src="https://github.com/giordamaug/SVElearn/blob/main/images/softvoting_ts.png" width="400" />


# Credits
The SVElearn package was developed by High Performance Computing and Networking Institute of National Research Council of Italy (ICAR-CNR).
This software is released under the GNU Licence (v.3) 

# Cite
If you use want to reference this software, please use the DOI: doi/10.5281/zenodo.10964743 

[![DOI](https://zenodo.org/badge/821813810.svg)](https://zenodo.org/doi/10.5281/zenodo.12598244)

If you want to cite the work in which this software was first used and described, 
please cite the following article:

```
@article {Granata2024.04.16.589691,
	author = {Ilaria Granata and Lucia Maddalena and Mario Manzo and Mario  Rosario Guarracino and Maurizio Giordano},
	title = {HELP: A computational framework for labelling and predicting human context-specific essential genes},
	elocation-id = {2024.04.16.589691},
	year = {2024},
	doi = {10.1101/2024.04.16.589691},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.16.589691},
	eprint = {https://www.biorxiv.org/content/early/2024/04/20/2024.04.16.589691.full.pdf},
	journal = {bioRxiv}
}
```
