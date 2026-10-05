# Polynomial regression assignment

The full configuration was run on both 1,000-row training datasets. These are
**mean outer-fold validation scores**, not scores on the hidden test targets:

| Problem | Final model | Degree | Alpha | L1 ratio | Outer MSE | Outer R2 |
|---|---|---:|---:|---:|---:|---:|
| var1 | Lasso | 5 | 0.01 | - | 0.351656 | 0.964071 |
| var2 | Elastic Net | 12 | 0.001 | 0.2 | 0.242820 | 0.995512 |

There were no convergence failures among the 9,720 inner candidate fits.
Both final CSVs contain 1,000 finite predictions. Saved-model reloads reproduced
the predictions, and source-data hashes were unchanged. The reproducibility
metadata and individual outer-fold results are in `results/summary.json`.

The selected alpha for var2 is at the lower edge of the searched sparse-model
grid. A wider grid could be a future experiment; this result does not establish
that 0.001 is the globally optimal regularization strength. The reported nested
scores evaluate the original fixed search described below.

## Method

Each problem is fitted independently. Problem 1 searches degrees **1 through 10**
with six inputs; problem 2 searches degrees **1 through 20** with three inputs.
Every expansion includes all monomials up to the selected **total degree**, including
interactions. The constant is excluded from the expansion because the regression
estimator fits a separate, unpenalized intercept.

The pipeline is:

1. Standardize the original inputs.
2. Generate polynomial features.
3. Standardize the resulting polynomial columns.
4. Fit Ridge, Lasso, or Elastic Net.

Both scalers are fitted only on the training rows of the current split. Validation
and test rows are transformed using these fitted scalers. The target is kept in
its original units. The supplied datasets are never modified.

### Nested cross-validation

- **Outer evaluation:** five shuffled folds, seed 43. Each outer validation fold
  stays outside all model selection and preprocessing fits.
- **Inner selection:** three shuffled folds, seed 42, using only the outer training
  rows. Search degree, model type, `alpha`, and Elastic Net's `l1_ratio` jointly.
  Select the lowest mean inner validation MSE. Each candidate must converge in
  every inner fold to be eligible. An exact MSE tie favors the smaller degree.
- **Outer refit:** refit the selected pipeline on the complete outer training set,
  evaluate the untouched outer fold, and save predictions and MSE/R2.
- **Final model:** repeat the inner search on all supplied training rows, then
  refit its winner on all rows. Outer results do not choose the final settings.

The standard deviation across outer folds describes variation, not a formal
confidence interval. Pooled out-of-fold R2 and average fold R2 are saved separately;
they need not agree. The final search score is a selection score, not an independent
performance estimate.

Ridge, Lasso, and Elastic Net scores are compared **separately**, never averaged
together. The regularization parameter is named `alpha` in scikit-learn; Ridge
and Lasso/Elastic Net use different objective normalizations, so equal numerical
alphas do not imply equal penalty strength.

The fixed initial grid is in `configs/full.json`. It is a finite search, not proof
of a globally optimal degree or penalty. Inspect `final_search.csv`, convergence
failures, and whether a winner sits at a grid boundary before drawing conclusions.
Changing the search after looking at outer scores makes those scores less independent;
record any such changes and use fresh evaluation data for a new unbiased estimate.

Random K-fold assumes rows are independent samples. If domain information reveals
time ordering, repeated groups, or a different sampling process, revise the splitter.

## Setup

Use Python 3.12 or newer. The tested local environment uses Python 3.14.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

With `uv`, the equivalent install is:

```powershell
uv venv --python 3.14 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt
```

## Run

First check the complete workflow using the deliberately small smoke configuration:

```powershell
.venv\Scripts\python train.py --config configs/smoke.json --output artifacts/smoke --jobs 2
```

Smoke predictions are for checking the code, **not the final assignment submission**.

Run the full nested experiment:

```powershell
.venv\Scripts\python train.py --config configs/full.json --output artifacts/full --jobs 2
```

Add `--problems 1` or `--problems 2` to run just one problem. Use `--rollno` and
`--data-dir` to change the student ID or data location.

The full search performs 3,240 inner model fits for problem 1 and 6,480 for problem 2,
including the final search, plus outer and final refits. High-degree correlated
features can make sparse regression slow. CPU execution is used; a GPU runtime
does not accelerate this implementation. `--jobs 2` runs two degrees concurrently,
with one numerical-library thread per worker to avoid CPU oversubscription.

Inner preprocessing is shared only within the same degree and fold. Sparse models
use warm starts along descending alpha paths on the same training rows. This
reduces work without leaking validation information. Nonconvergent candidates are
recorded and excluded; a nonconvergent selected refit stops the run.

An identical command resumes completed degree searches and outer evaluations.
Checkpoints are bound to hashes of both CSVs, the implementation, configuration,
and package versions. If any changes, use a new output directory. Incomplete degree
searches restart from their beginning. Never edit checkpoint files to force reuse.

## Outputs

```text
artifacts/full/
  predictions/
    BT2024151_pred_var1.csv
    BT2024151_pred_var2.csv
  var1/                         # var2 has the same structure
    manifest.json               # configuration, versions, source/data hashes
    nested_metrics.json         # independent outer evaluation scores
    outer_predictions.csv       # row index, fold, true y, held-out prediction
    final_selection.json        # final settings and number of nonzero coefficients
    final_search.csv            # every candidate's final selection CV score
    model.joblib                # fitted scalers, expansion, regressor, metadata
    search/                     # resumable inner searches and outer evaluations
```
If you want to regenerate predictions without training:

```powershell
.venv\Scripts\python predict.py --model artifacts/full/var1/model.joblib --test Traning-testing-data/BT2024151_test_var1.csv --output artifacts/full/predictions/BT2024151_pred_var1.csv
.venv\Scripts\python predict.py --model artifacts/full/var2/model.joblib --test Traning-testing-data/BT2024151_test_var2.csv --output artifacts/full/predictions/BT2024151_pred_var2.csv
```
