# Model Training

The `model` package contains several modeling directions, but the exact training recipe depends on the dataset representation you select.

## Suggested Workflow

1. Load data with a dataset manager.
2. Decide which feature space you want to train on.
3. Create train/valid/test indices with the splitter package.
4. Train a model from `pepbenchmark.model` or an external estimator.
5. Evaluate predictions with `pepbenchmark.evaluator`.

For repeat-run reporting, use `pepbenchmark.model.utils.load_results_from_dir()` and `summarize_results_grouped()`.
