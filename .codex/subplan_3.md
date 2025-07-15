# Subplan 3 – Add new tests

Goals:
- Increase coverage by testing helper utilities and argument parsing.
- Keep tests lightweight so they run fast in CI.

## Proposed tests
1. **Argument parser**
   - Call `parse_arguments` with minimal args and check defaults.
   - Test `str2bool` and `str2ints` conversions.
2. **Utility functions**
   - `fasta_to_dict` should correctly parse example FASTA file.
   - `check_input_tokens` should raise `ValueError` when invalid tokens present.
3. **Dataset collation**
   - Create a small dummy tokenizer (from transformers) to test `HuggingFaceDataset.safe_collate` and `TokenBudgetBatchSampler` logic using simple sequences.

## Steps
1. Implement test functions in `src/tests/test_utils.py` covering the above functions.
2. Run `pytest` to ensure all tests pass including the pipeline test.
3. Commit the new tests.
