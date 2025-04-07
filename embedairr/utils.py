from typing import Sequence, Tuple
from torch.utils.data import Dataset
import re
import torch
import gc
import os
import psutil
import numpy as np
import tracemalloc


class TokenBudgetBatchSampler:
    def __init__(self, dataset, token_budget):
        self.dataset = dataset
        self.token_budget = token_budget

        # Assume all sequences have the same length (already padded)
        sample_seq_len = len(
            dataset[0][2]
        )  # dataset[idx] -> (label, seq_str, toks, mask)
        self.batch_size = token_budget // sample_seq_len

        self.batches = self._create_batches()

    def _create_batches(self):
        indices = list(range(len(self.dataset)))
        return [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class HuggingFaceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length, add_special_tokens=True):
        self.data = list(sequences.items())  # (label, seq)
        self.encoded_data = self.encode_sequences(
            tokenizer, max_length, add_special_tokens
        )  # (label, seq, toks, attention_mask)

    def encode_sequences(self, tokenizer, max_length, add_special_tokens):
        labels, strs = zip(*self.data)
        strs = self.gap_sequence(strs)
        max_token_length = max(len(seq.split(" ")) for seq in strs)

        if max_length == "max_length":
            max_length = max_token_length
            print(f"Setting max_length to {max_length}.")
        elif isinstance(max_length, int) and max_length < max_token_length:
            print(
                f"max_length {max_length} is less than the length of the longest sequence: {max_token_length}. Setting max_length to {max_token_length}."
            )
            max_length = max_token_length
        encoded = tokenizer(
            list(strs),
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
            max_length=max_length,
        )
        toks = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]
        return list(zip(labels, strs, list(toks), list(attention_masks)))

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def __len__(self):
        return len(self.encoded_data)

    def gap_sequence(self, sequences: Sequence[str]) -> Sequence[str]:
        """Space-separated tokenization for RoFormer input."""
        seqs = [" ".join(re.findall(r"\[.*?\]|.", sequence)) for sequence in sequences]
        return seqs

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks, _ in self.encoded_data)


class ESMDataset(Dataset):
    def __init__(
        self, sequences, alphabet, max_length, prepend_bos=True, append_eos=True
    ):
        self.data = list(sequences.items())  # (label, seq)
        self.encoded_data = self.encode_sequences(
            alphabet, max_length, prepend_bos=prepend_bos, append_eos=append_eos
        )  # (label, seq, toks)

    def encode_sequences(self, alphabet, max_length, prepend_bos=True, append_eos=True):
        labels, strs = zip(*self.data)
        encoded = [alphabet.encode(seq_str) for seq_str in strs]
        max_encoded_length = max(len(seq_encoded) for seq_encoded in encoded)
        if max_length == "max_length":
            max_length = max_encoded_length
        elif isinstance(max_length, int) and max_length < max_encoded_length:
            print(
                f"max_length {max_length} is less than the length of the longest sequence: {max_encoded_length}. Setting max_length to {max_encoded_length}."
            )
        tokens = torch.empty(
            (len(encoded), max_length + int(prepend_bos) + int(append_eos)),
            dtype=torch.int64,
        )
        tokens.fill_(alphabet.padding_idx)
        for i, seq_encoded in enumerate(encoded):
            if prepend_bos:
                tokens[i, 0] = alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(prepend_bos) : len(seq_encoded) + int(prepend_bos),
            ] = seq
            if append_eos:
                tokens[i, len(seq_encoded) + int(prepend_bos)] = alphabet.eos_idx
        return list(zip(labels, strs, list(tokens)))

    def __getitem__(self, idx):
        label, seq_str, toks = self.encoded_data[idx]
        return label, seq_str, toks, None

    def __len__(self):
        return len(self.encoded_data)

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks in self.encoded_data)

    def safe_collate(self, batch):
        labels, seqs, toks, _ = zip(*batch)
        return list(labels), list(seqs), torch.stack(toks), None


def check_input_tokens(valid_tokens, sequences, model_name):
    print("Checking input sequences for invalid tokens...")
    for i, (label, sequence) in enumerate(sequences.items()):
        if "esm" not in model_name:
            sequence = re.findall(r"\[.*?\]|.", sequence)
        else:
            sequence = re.findall(r"<.*?>|.", sequence)
        if not set(sequence).issubset(valid_tokens):
            raise ValueError(
                f"Invalid tokens in sequence {label}. Please check the alphabet used by the model."
            )
        print(f"Processed {i + 1} out of {len(sequences)} sequences", end="\r")

    print("\nNo invalid tokens in input sequences.")


def fasta_to_dict(fasta_path):
    """Convert FASTA file into a dictionary: {id: raw_sequence}."""
    seq_dict = {}
    sequence_id = None
    sequence_aa = []

    def flush():
        nonlocal sequence_id, sequence_aa
        if sequence_id:
            seq_dict[sequence_id] = "".join(sequence_aa)
        sequence_id, sequence_aa = None, []

    with open(fasta_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line.startswith(">"):
                flush()
                sequence_id = line[1:] or f"seqnum{line_idx:09d}"
            else:
                sequence_aa.append(line)
    flush()

    return seq_dict


def flush_memmaps(obj):
    """Recursively flush memory maps."""
    if hasattr(obj, "flush") and callable(obj.flush):
        obj.flush()
        gc.collect()
        print("Flushed output")
    elif isinstance(obj, dict):
        for value in obj.values():
            flush_memmaps(value)


import os
import gc
import sys
import psutil
import numpy as np
import tracemalloc

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import csv
import datetime


class MemoryProfiler:
    def __init__(
        self, memmap_paths=None, enable_tracemalloc=True, log_path="memory_log.csv"
    ):
        self.process = psutil.Process()
        self.memmap_paths = memmap_paths or []
        self.log_path = log_path

        if enable_tracemalloc:
            tracemalloc.start()

        self._init_log_file()

    def _init_log_file(self):
        # Only write header if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "tag",
                        "rss_gb",
                        "numpy_mb",
                        "torch_mb",
                        "memmap_mb",
                        "tracemalloc_top1",
                        "tracemalloc_top2",
                        "tracemalloc_top3",
                    ]
                )

    def log(self, tag=""):
        rss_gb = self.process.memory_info().rss / 1e9

        # NumPy buffers
        numpy_bytes = sum(
            arr.nbytes for arr in gc.get_objects() if isinstance(arr, np.ndarray)
        )

        # PyTorch tensor memory (CPU only)
        torch_bytes = 0
        if TORCH_AVAILABLE:
            torch_bytes = sum(
                t.element_size() * t.nelement()
                for t in gc.get_objects()
                if isinstance(t, torch.Tensor)
            )

        # Memmap file sizes
        memmap_bytes = sum(
            os.path.getsize(path) for path in self.memmap_paths if os.path.exists(path)
        )

        # Tracemalloc top 3
        top_stats = []
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:3]

        top_lines = [str(stat) for stat in top_stats]
        top_lines += [""] * (3 - len(top_lines))  # pad to always 3

        # Write to CSV
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.datetime.now().isoformat(),
                    tag,
                    f"{rss_gb:.2f}",
                    f"{numpy_bytes / 1e6:.2f}",
                    f"{torch_bytes / 1e6:.2f}",
                    f"{memmap_bytes / 1e6:.2f}",
                    *top_lines,
                ]
            )
