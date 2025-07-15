# Subplan 2 – Fix existing failing tests

The current test file `src/tests/test_run.py` uses the argument `substring_pooling` which is not a valid choice for `--extract_embeddings`. It should be `substring_pooled`.

## Steps
1. Edit `src/tests/test_run.py` to replace `substring_pooling` with `substring_pooled`.
2. Run `pytest -q` to verify tests execute successfully.  This should run the small custom model and output results.
3. Commit the fixed test and update this subplan with completion notes.
