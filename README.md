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
# pKa Prediction using Morgan Fingerprints and Random Forest

This project implements a machine learning workflow for predicting molecular **pKa values** using **Morgan fingerprints** and a **Random Forest regressor**.

The project focuses on two key aspects:

1. **Model optimization** (Random Forest hyperparameter tuning)
2. **Molecular representation analysis** (Morgan fingerprint parameter tuning)

The goal is not only to achieve good predictive performance, but also to understand:

> **What kind of molecular information is actually important for pKa prediction?**

---

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
│   ├── tune_morgan_rf.py
│   ├── plot_tuning_results.py
│   └── plot_morgan_tuning_results.py
│
├── outputs
├── figures
│
├── requirements.txt
│
└── README.md
```

---

# Workflow

The project now contains two complementary workflows:

## 1️⃣ Random Forest Hyperparameter Tuning

```
Raw dataset
↓
Preprocess
↓
Split (random / scaffold)
↓
Featurize (fixed Morgan)
↓
Tune Random Forest
↓
Evaluate
↓
Plot RF tuning results
```

## 2️⃣ Morgan Fingerprint Tuning (NEW)

```
Split dataset
↓
Loop over (radius, n_bits)
↓
Generate Morgan fingerprints (on-the-fly)
↓
Train Random Forest (fixed parameters)
↓
Evaluate performance
↓
Analyze representation quality
↓
Plot tuning results
```

---

# Dataset Preprocessing

The preprocessing step performs:

- SMILES validation using RDKit
- removal of invalid structures
- canonicalization
- aggregation of repeated measurements

Final dataset format:

```
SMILES, pKa
```

---

# Molecular Representation

The model uses **Morgan fingerprints (ECFP-like)**.

## Key parameters:

- `radius` → size of local chemical environment
- `n_bits` → fingerprint dimensionality

## Interpretation

- radius = 0 → atom-level (no structure)
- radius = 1 → local functional groups
- radius ≥ 2 → larger structural context

---

# Machine Learning Model

Model: **Random Forest Regressor**

Fixed configuration (for Morgan tuning):

```
n_estimators = 500
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = 0.5
```

Why Random Forest:

- robust to sparse fingerprints
- handles nonlinear relationships
- interpretable

---

# Dataset Splitting Strategies

## Random Split

- easier task
- similar distribution

## Scaffold Split

- harder task
- tests generalization across chemical scaffolds
- more realistic

---

# Morgan Fingerprint Tuning (Core Contribution)

We systematically evaluated:

```
radius ∈ {0, 1, 2, 3}
n_bits ∈ {512, 1024, 2048, 4096}
```

## Key Findings

### 1️⃣ Radius does NOT monotonically improve performance

- radius = 1 performs best
- radius ≥ 2 introduces noise
- radius = 0 still provides useful baseline

### 2️⃣ pKa is a local property

Local chemical environments dominate pKa prediction.

### 3️⃣ Larger fingerprints help

- increasing `n_bits` improves performance
- diminishing returns beyond 2048–4096

### 4️⃣ Scaffold split reveals stronger trends

- performance degradation is more obvious
- confirms overfitting at larger radius

## Interpretation

```
pKa ≈ functional group + local substituent effects
```

---

# Running the Project

## 1️⃣ Preprocess dataset

```
python src/preprocess.py
```

## 2️⃣ Split dataset

```
python src/random_split.py
python src/scaffold_split.py
```

## 3️⃣ Train baseline model

```
python src/train.py
```

## 4️⃣ Tune Random Forest

```
python src/tune_rf.py
```

## 5️⃣ Plot RF tuning results

```
python src/plot_tuning_results.py
```

## 6️⃣ Tune Morgan fingerprint

```
python src/tune_morgan_rf.py --mode both
```

## 7️⃣ Plot Morgan tuning results

```
python src/plot_morgan_tuning_results.py
```

---

# Key Takeaways

1. Morgan fingerprints + Random Forest provide a strong baseline.
2. pKa prediction is primarily driven by local chemical environments.
3. radius = 1 is optimal for capturing functional groups and inductive effects.
4. Larger radius introduces noise rather than useful information.
5. Scaffold split is essential for realistic evaluation.

---

# Future Improvements

- graph neural networks
- quantum chemical descriptors
- pKa site prediction
- conformer-aware features
- hybrid models

---

# License

This project is intended for educational and research purposes.