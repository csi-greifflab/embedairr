# Plan for Pepe CLI Project Improvements

## Overview
We need to achieve three overall goals:
1. Gain a thorough understanding of how the package works.
2. Create additional tests for the package under `src/tests` and ensure all tests pass.
3. Refactor `base_embedder.py` and `utils.py` into smaller modules or move functions to existing modules.

To accomplish this we will break the work down into smaller sub‑problems and tackle them sequentially.  For each sub‑problem we will create a `subplan_<n>.md` with more detailed steps before making code changes.

## Sub‑problems

### 1. Familiarise with the code base
- Read through the package structure, main modules, and understand existing functionality.
- Document important observations in `subplan_1.md`.

### 2. Fix existing failing tests
- Investigate `src/tests/test_run.py` and fix incorrect argument (`substring_pooling` should be `substring_pooled`).
- Verify that tests run successfully after the fix.

### 3. Add new tests
- Identify key functions/classes lacking tests, especially in `utils.py`, `parse_arguments.py`, and embedding logic.
- Add unit tests for these components.  Aim for good coverage but keep individual tests lightweight so they run quickly.

### 4. Refactor `utils.py`
- Split dataset related classes into a new module, e.g. `datasets.py`.
- Move I/O worker classes into a new module, e.g. `io_utils.py`.
- Keep small helper functions in `utils.py` or move to more appropriate files.
- Update imports across the package and tests.

### 5. Refactor `base_embedder.py`
- Extract output preparation and extraction helper functions into a new module (e.g. `embedder_utils.py`).
- Keep the main `BaseEmbedder` class logic but trim down by importing helper functions.
- Ensure backward compatibility of public API.

### 6. Update documentation if needed
- If any public interfaces change, update README or docstrings.

### 7. Final verification
- Run `pytest` to confirm all tests pass after refactoring.
- Commit all changes.

Each sub‑problem will have its own subplan document describing concrete actions and tasks.
