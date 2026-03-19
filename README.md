# pKa Prediction using Morgan Fingerprints and Random Forest

This project implements a machine learning workflow for predicting molecular **pKa values** using **Morgan fingerprints** and a **Random Forest regressor**.  
In this project, the predicted pKa corresponds specifically to the **acid dissociation constant of the protonated form (AH)** of a molecule, i.e., the acidity of a species after it has accepted a proton.

The project focuses on two complementary and equally important aspects:

1. **Model-level optimization** (Random Forest hyperparameter tuning)  
2. **Representation-level optimization** (Morgan fingerprint parameter tuning)

Together, these two components allow us to systematically study how both model capacity and molecular representation affect pKa prediction performance.

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

## 1️⃣ Random Forest Hyperparameter Tuning (Core Component)

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

## 2️⃣ Morgan Fingerprint Tuning (Core Component)

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

---

## Definition of pKa in this Project

In this work, pKa is defined as:

\[
\mathrm{AH \rightleftharpoons A^- + H^+}
\]

where the molecule of interest is the **protonated species (AH)**.

This means:

- The model predicts how easily a **protonated molecule loses a proton**
- For basic molecules (e.g., amines), this corresponds to the acidity of their conjugate acids
- For acidic molecules, this aligns with the usual acid dissociation definition

---

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

# Results and Analysis

This section summarizes key findings from both **Random Forest hyperparameter tuning** and **Morgan fingerprint tuning**.

## 1️⃣ Random Forest Hyperparameter Tuning

We evaluated the effect of several key hyperparameters:

- `max_depth`
- `max_features`
- `min_samples_leaf`
- `min_samples_split`
- `n_estimators`

### Key Observations

#### (1) Model complexity vs generalization

- Increasing `max_depth` consistently improves performance  
- Best performance achieved at `max_depth = None`  
- Indicates that shallow trees underfit the data

#### (2) Feature subsampling is critical

- `max_features = 0.5` performs best  
- Too small (`0.3`) → underfitting  
- Too large (`log2`) → increased variance and worse MAE  

#### (3) Leaf size controls bias-variance tradeoff

- `min_samples_leaf = 1` gives best performance  
- Larger values increase bias and degrade accuracy  

#### (4) Split constraint has minor effect

- `min_samples_split` shows limited influence  
- Suggests dataset is not highly sensitive to this parameter  

#### (5) Number of trees saturates quickly

- Increasing `n_estimators` beyond ~500 provides minimal improvement  
- Confirms diminishing returns for ensemble size  

#### (6) Random vs Scaffold split

- Scaffold split consistently yields higher MAE  
- Demonstrates limited generalization across chemical scaffolds  

---

## 2️⃣ Morgan Fingerprint Tuning

We systematically evaluated:

```
radius ∈ {0, 1, 2, 3}
n_bits ∈ {512, 1024, 2048, 4096}
```

### Key Observations

#### (1) Locality dominates

- `radius = 1` consistently performs best  
- Confirms that pKa is primarily determined by **local chemical environments**

#### (2) Larger radius introduces noise

- `radius ≥ 2` leads to worse performance  
- Suggests global structure is less relevant than local functional context  

#### (3) Radius = 0 still works

- Even atom-level features provide useful signal  
- Indicates strong correlation between atom types and pKa  

#### (4) Increasing dimensionality helps

- Larger `n_bits` improves performance  
- Especially important for avoiding hash collisions  

#### (5) Scaffold split amplifies trends

- Performance degradation is more pronounced  
- Confirms overfitting when using overly large radius  

---

## 3️⃣ Combined Insight

The overall model behavior can be summarized as:

```
pKa prediction ≈ local chemical environment + nonlinear mapping (Random Forest)
```

- Morgan fingerprint defines **what information is available**
- Random Forest defines **how that information is used**

Optimal performance requires balancing both.

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