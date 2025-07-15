# Subplan 5 – Refactor base_embedder.py

`base_embedder.py` still contains a large set of helper methods.  We will move extraction helper functions and tensor utilities into a new module `embedder_utils.py` while keeping the main `BaseEmbedder` class in place.

Functions to move:
- `_precision_to_dtype`
- `_prepare_tensor`, `_to_numpy`
- all `_extract_*` methods (`_extract_logits`, `_extract_mean_pooled`, etc.)
- `_mask_special_tokens`

`BaseEmbedder` will import these functions from the new module.  The public behaviour should remain unchanged.

## Steps
1. Create `src/pepe/embedder_utils.py` and move the helper functions there.
2. Update `BaseEmbedder` to import them and use as instance methods via composition (e.g. `from .embedder_utils import precision_to_dtype` etc.).
3. Run tests to ensure everything still works.
