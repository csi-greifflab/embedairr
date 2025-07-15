# Subplan 1 – Familiarise with code base

Objective: understand the structure of the project and identify components needing tests and refactoring.

## Steps
1. Inspect project directories – done.
2. Review `src/pepe` package modules including embedders and utilities – done during initial exploration.
3. Note the following observations:
   - `utils.py` contains dataset classes, token validation functions, disk flushing IO worker classes, and other helpers. This file is over 900 lines and can be split into datasets and IO utilities.
   - `base_embedder.py` implements the core embedding pipeline including output management, disk preallocation, batching and extraction helpers. It is ~800 lines.
   - `parse_arguments.py` and `model_selecter.py` provide CLI argument parsing and model selection.
   - Current tests include `test_run.py` which runs the full embedding pipeline using a small custom model. The script uses an invalid argument (`substring_pooling`), causing test failure.
4. Identify potential new tests:
   - Argument parsing (ensure boolean/int conversions and defaults work).
   - `fasta_to_dict` and token checking utilities.
   - Basic dataset collation from `HuggingFaceDataset` or `ESMDataset` using dummy tokenizer/alphabet.
5. Future subplans will address fixes and refactoring based on these observations.
