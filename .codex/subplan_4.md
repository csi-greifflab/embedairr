# Subplan 4 – Refactor utils.py

`utils.py` currently holds a variety of unrelated classes and functions.  We will split it into two new modules while keeping backward compatibility:

1. `datasets.py` – contains `SequenceDictDataset`, `HuggingFaceDataset`, `ESMDataset`, and `TokenBudgetBatchSampler`.
2. `io_utils.py` – contains `IOFlushWorker`, `MultiIODispatcher`, `check_disk_free_space`, `flush_memmaps` and related helpers.

The remaining small helper functions (`fasta_to_dict`, `check_input_tokens`) will stay in `utils.py`.

## Steps
1. Create `src/pepe/datasets.py` and move dataset related classes there.
2. Create `src/pepe/io_utils.py` and move IO worker classes and helper functions there.
3. Update imports in other modules to reference the new files.
4. Keep re-export statements in `utils.py` so that old imports continue to work:
   ```python
   from .datasets import SequenceDictDataset, HuggingFaceDataset, ESMDataset, TokenBudgetBatchSampler
   from .io_utils import IOFlushWorker, MultiIODispatcher, flush_memmaps, check_disk_free_space
   ```
5. Run tests to ensure behaviour is unchanged.
