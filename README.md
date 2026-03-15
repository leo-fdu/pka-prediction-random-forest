# pKa Prediction using Morgan Fingerprints and Random Forest

This project implements a machine learning workflow for predicting molecular **pKa values** using **Morgan fingerprints** and a **Random Forest regressor**.  
The goal is to investigate how classical cheminformatics descriptors combined with ensemble learning perform on pKa prediction tasks and how model performance changes under different **data splitting strategies** and **hyperparameter configurations**.

The project includes:

- dataset preprocessing
- scaffold-based and random dataset splitting
- Morgan fingerprint feature generation
- Random Forest training and evaluation
- hyperparameter tuning
- visualization of tuning results


# Project Structure

```
pka_predictor_random_forest
│
├── data
│
├── src
│   ├── preprocess.py
│   ├── random_split.py
│   ├── scaffold_split.py
│   ├── featurize.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune_rf.py
│   └── plot_tuning_results.py
│
├── results
│
├── requirements.txt
│
└── README.md
```


# Workflow

The overall workflow of the project is:

```
Raw dataset
↓
Data cleaning (preprocess.py)
↓
Dataset splitting
  • random split
  • scaffold split
↓
Morgan fingerprint generation
↓
Random Forest training
↓
Hyperparameter tuning
↓
Model evaluation
↓
Visualization of results
```


# Dataset Preprocessing

The preprocessing step performs:

- SMILES validation using RDKit
- removal of invalid structures
- canonicalization of molecules
- aggregation of repeated measurements
- extraction of final dataset containing:

```
SMILES, pKa
```


# Molecular Representation

The model uses **Morgan fingerprints** as molecular descriptors.

Morgan fingerprints are circular fingerprints generated using the Morgan algorithm (similar to ECFP).

Advantages:

- captures local atomic environments
- invariant to atom ordering
- widely used in cheminformatics

Typical parameters:

- radius = 2
- binary bit vector


# Machine Learning Model

The model used is a **Random Forest Regressor**.

Random Forest is an ensemble learning method based on bagging and decision trees.

Advantages:

- robust to overfitting
- works well with sparse fingerprints
- interpretable hyperparameters

Key hyperparameters explored:

- `n_estimators`
- `max_depth`
- `max_features`
- `min_samples_split`
- `min_samples_leaf`


# Dataset Splitting Strategies

Two splitting strategies were compared.


## Random Split

Randomly distributes molecules into:

- training set
- validation set
- test set

Advantages:

- balanced distribution
- easier prediction task

Disadvantages:

- training and test molecules may share similar scaffolds


## Scaffold Split

Uses **Bemis–Murcko scaffolds** to group molecules.

Molecules sharing the same scaffold are placed into the same dataset split.

Advantages:

- better simulation of real-world generalization
- prevents scaffold leakage

Disadvantages:

- harder prediction task


# Model Performance Comparison

Figure 1 compares the average performance of models trained under random and scaffold splitting strategies.

```
results/figure_1_metric_means_random_vs_scaffold.png
```

Observations:

- Random split produces better validation performance.
- Scaffold split results in higher prediction error.
- This indicates scaffold-based evaluation is a more challenging and realistic task.


# Hyperparameter Tuning

Hyperparameter tuning was performed by grid search across several parameters.

The effect of each parameter on validation MAE is shown below.


## Effect of max_depth

```
results/figure_2_max_depth_val_mae_comparison.png
```

Observation:

Increasing tree depth generally improves model performance until saturation.


## Effect of max_features

```
results/figure_2_max_features_val_mae_comparison.png
```

Observation:

Using larger feature subsets significantly improves performance compared to `sqrt` or `log2`.


## Effect of min_samples_leaf

```
results/figure_2_min_samples_leaf_val_mae_comparison.png
```


## Effect of min_samples_split

```
results/figure_2_min_samples_split_val_mae_comparison.png
```


## Effect of n_estimators

```
results/figure_2_n_estimators_val_mae_comparison.png
```


# Best Hyperparameter Combinations

The best hyperparameter combinations based on validation MAE are summarized below.

```
results/table_1_best_hyperparameters.png
```


# Example Best Model Performance

Random split best model:

```
Validation MAE  = 0.866
Validation RMSE = 1.225
Validation R²   = 0.850
```

Scaffold split best model:

```
Validation MAE  = 1.263
Validation RMSE = 1.637
Validation R²   = 0.621
```


# Installation

Install dependencies using:

```
pip install -r requirements.txt
```

or using conda:

```
conda install -c conda-forge rdkit scikit-learn pandas matplotlib numpy
```


# Running the Project


## 1 Preprocess dataset

```
python src/preprocess.py
```


## 2 Split dataset

Random split

```
python src/random_split.py
```

Scaffold split

```
python src/scaffold_split.py
```


## 3 Train model

```
python src/train.py
```


## 4 Hyperparameter tuning

```
python src/tune_rf.py
```


## 5 Plot results

```
python src/plot_tuning_results.py
```


# Key Takeaways

1. Morgan fingerprints combined with Random Forest provide strong baseline performance for pKa prediction.
2. Scaffold-based splitting reveals the true generalization ability of the model.
3. Increasing `max_features` significantly improves performance.
4. Increasing tree depth improves performance but with diminishing returns.
5. Increasing the number of trees improves stability but increases computation cost.


# Future Improvements

Possible extensions include:

- graph neural networks
- molecular descriptors beyond fingerprints
- pKa site prediction
- conformer-aware features
- larger datasets


# License

This project is intended for educational and research purposes.