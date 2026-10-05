# Polynomial Regression Assignment

**Name:** Kulshresth  
**Roll Number:** IMT2024065

## Overview

This repository contains the complete implementation used for the polynomial regression assignment.

The program:

- searches polynomial degrees for both problems
- evaluates Ridge regression for every allowed degree
- evaluates Lasso and Elastic Net on the most promising degrees
- uses validation MSE for model and hyperparameter selection
- controls high-degree polynomial feature growth using random candidate selection followed by F-statistic screening
- standardizes selected polynomial features before regularized regression
- performs model search in parallel using CPU threads
- refits the selected model on the complete training data
- generates the required prediction CSV files for both problems

The implementation is CPU-only and does not require CUDA, PyTorch, or a GPU.

## Repository Structure

```text
ML-Assignment/
├── code.py
├── requirements.txt
├── README.md
├── IMT2024065_train_var1.csv       # input
├── IMT2024065_test_var1.csv        # input
├── IMT2024065_train_var2.csv       # input
├── IMT2024065_test_var2.csv        # input
├── IMT2024065_pred_var1.csv        # output
├── IMT2024065_pred_var2.csv        # output
├── Report.pdf
└── LICENSE
```

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Scikit-learn

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Input Files

The program expects the following four CSV files to be present in the same directory as `code.py`:

```text
IMT2024065_train_var1.csv
IMT2024065_test_var1.csv
IMT2024065_train_var2.csv
IMT2024065_test_var2.csv
```

The training files must contain a column named `y`, which is the regression target. All other training columns are treated as input features. The corresponding test files must contain the same feature columns.

## Running the Program

From the repository directory:

```bash
python code.py
```

The program first validates that all required CSV files are present. It then runs the complete model-selection pipeline for `var1` and `var2`.

## Output Files

Only the two required prediction files are generated:

```text
IMT2024065_pred_var1.csv
IMT2024065_pred_var2.csv
```

Each output file contains one column:

```text
y
```

with one prediction for each test row.

## Methodology

### 1. Train/Validation Split

The training data are split into:

- 80% training
- 20% validation

using:

```text
random_state = 2026
```

The validation set is used for selecting the polynomial degree, model type, and regularization hyperparameters. The test set is not used for model selection.

### 2. Polynomial Degree Search

The assignment allows the following degree ranges:

```text
var1: degrees 1 to 10
var2: degrees 1 to 20
```

Ridge regression is first evaluated for every degree. This provides a stable and relatively inexpensive first pass over the complete degree range.

The five degrees with the lowest Ridge validation MSE are then selected for the more expensive Lasso and Elastic Net searches.

The final winner is the configuration with the lowest validation MSE among all evaluated configurations.

### 3. Why Regularization Is Used

Polynomial expansion can produce many correlated features and can make an unregularized model unstable or prone to overfitting. Three regularized linear models are considered:

**Ridge**

Uses an L2 penalty. It shrinks coefficients toward zero and is useful when many correlated polynomial terms may contribute to the prediction.

**Lasso**

Uses an L1 penalty. It can force coefficients exactly to zero, which produces sparse models and can remove unnecessary polynomial terms.

**Elastic Net**

Combines L1 and L2 regularization. It provides sparsity while retaining some of Ridge's stability when polynomial features are correlated.

The validation set determines which of these behaviors works best for each problem.

### 4. Data-Driven Ridge Alpha Search

Ridge alpha values are not hard-coded.

The selected polynomial matrix is standardized temporarily, and the Gram matrix

```text
X^T X / n
```

is used to obtain a data-dependent scale through its largest eigenvalue.

A geometric grid of 12 positive Ridge alpha values is then constructed from:

- the largest eigenvalue
- the number of selected features
- the number of samples

A geometric grid is used because useful regularization strengths can span several orders of magnitude.

### 5. Data-Driven Lasso and Elastic Net Alpha Search

For Lasso and Elastic Net, the code first calculates the standardized-data quantity

```text
alpha_max = max(|X^T y|) / n
```

This provides a natural scale for the L1 penalty.

Twelve alpha values are generated geometrically from a sample-size-dependent lower fraction of `alpha_max` up to `alpha_max`.

This makes the search adapt to the actual dataset instead of relying on an arbitrary fixed list of alpha values.

### 6. Elastic Net L1 Ratio Search

The Elastic Net `l1_ratio` controls the balance between L1 and L2 regularization:

```text
l1_ratio = 1        -> Lasso-like
l1_ratio closer to 0 -> Ridge-like
```

Seven candidate ratios are generated from the number of selected features. Higher-dimensional polynomial spaces include more L2-heavy candidates because correlated polynomial terms can benefit from the stabilizing effect of L2 regularization.

### 7. Feature Scaling

After polynomial expansion and feature selection, the selected features are standardized using `StandardScaler`.

Scaling is important because polynomial terms can have very different magnitudes. For example, higher powers of an input can become much larger than the original feature.

Without scaling, regularization would not affect coefficients on comparable numerical scales, and the optimization of Lasso and Elastic Net could become poorly conditioned.

For validation:

```text
scaler.fit(...)      -> training split only
scaler.transform(...) -> validation split
```

For final inference:

```text
scaler.fit(...)      -> complete training set
scaler.transform(...) -> test set
```

This prevents validation/test information from being used during preprocessing.

### 8. Random Candidate Selection and F-Statistic Screening

Polynomial feature counts grow rapidly at higher degrees. To keep the optimization manageable, the program limits the working polynomial matrix to at most:

```text
800 features
```

When the full polynomial expansion is larger than this limit:

1. All terms up to degree 2 are preserved.
2. Up to 2500 higher-degree terms are randomly sampled.
3. The sampled terms are ranked using the univariate F-statistic from `f_regression`.
4. The highest-scoring terms are retained until the 800-feature limit is reached.

The random stage reduces the number of high-degree candidates that must be processed. The F-statistic stage then keeps candidates that show stronger individual relationships with the target.

This is more informative than pure random deletion while keeping memory usage and optimization cost under control.

A fixed, degree-dependent random seed makes the selection reproducible.

During validation, feature selection is performed only using the training split and the same selected columns are then applied to the validation matrix.

For the final model, the selection process is repeated using all available training data before transforming the test set.

### 9. Efficient Model Search

Model configurations are independent, so the program uses `ThreadPoolExecutor` to fit configurations concurrently.

The worker count is selected automatically from the CPU count and capped at 12.

BLAS-level threading is limited to one thread per worker so that multiple layers of parallelism do not oversubscribe the CPU.

This does not change the model-selection criterion; it only reduces the time required to evaluate the search space.

### 10. Convergence Settings

Lasso and Elastic Net are iterative optimization problems. Their iteration budgets are scaled according to the size of the current design matrix:

```text
max_iter = max(5000, factor * max(n_samples, n_features))
```

The factors are:

```text
20 during model search
80 during final fitting
```

The search uses:

```text
tol = 2e-4
```

while the final fit uses:

```text
tol = 1e-5
```

The looser search tolerance speeds up the large hyperparameter search, while the tighter final tolerance gives the selected model a more accurate final optimization.

## Model Selection Criterion

The final configuration is selected using validation Mean Squared Error:

```text
MSE = (1/n) * sum((y - y_pred)^2)
```

The configuration with the **lowest validation MSE** is chosen.

MSE is appropriate because this is a regression problem and directly measures squared prediction error. It also penalizes large errors more strongly.

Validation `R²` is also reported for interpretation, but it is not the selection criterion.

The test set is kept completely separate from these decisions and is used only for final inference.

## Final Results

The final run produced the following selected configurations:

| Problem | Degree | Model | Selected Features | Alpha | L1 Ratio | Validation MSE | Validation R² |
|---|---:|---|---:|---:|---:|---:|---:|
| var1 | 5 | Lasso | 461 | 0.0084739333 | -- | 0.2760779252 | 0.9729429883 |
| var2 | 10 | Ridge | 285 | 3.0173217001 | -- | 0.2510143744 | 0.9943283676 |

For `var1`, degree 5 achieved the best result after the Lasso/Elastic Net stage. The best Ridge result at degree 5 had validation MSE `0.454036`, while Lasso reduced this to `0.276078`.

For `var2`, degree 10 gave the best overall validation MSE. Although Elastic Net was also evaluated on the strongest Ridge candidate degrees, the global winner remained Ridge at degree 10 with validation MSE `0.251014`.

## Final Inference

After selecting the winning configuration for each problem, the program:

1. rebuilds the selected polynomial degree using all training rows
2. repeats the polynomial-term selection using the complete training set
3. standardizes the selected features using the complete training data
4. retrains the selected Ridge, Lasso, or Elastic Net model on all training targets
5. transforms the unseen test data
6. predicts `y` for every test row
7. writes the predictions to the required CSV file

This final prediction step is the inference stage.

## Notes

- The implementation does not use GPU-specific libraries.
- The program generates only the two required prediction files.
- Model-selection results are printed to the console during execution.
- The validation split uses a fixed random seed, making the model-selection process reproducible.
