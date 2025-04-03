from typing import Sequence, Tuple
from torch.utils.data import Dataset
import re
import torch
import gc
import os
import psutil
import numpy as np
import tracemalloc


class ESMBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, truncation_seq_length: int = None):
        self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        if self.truncation_seq_length:
            seq_encoded_list = [
                seq_str[: self.truncation_seq_length] for seq_str in seq_encoded_list
            ]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = (
                    self.alphabet.eos_idx
                )

        return labels, strs, tokens, None


class HuggingFaceBatchConverter:
    def __init__(self, tokenizer, max_length=512, add_special_tokens=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        labels, strs = zip(*raw_batch)

        tokens = self.tokenizer(
            list(strs),
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_length,
        )

        return list(labels), list(strs), tokens["input_ids"], tokens["attention_mask"]


class HuggingFaceBatchSampler:
    def __init__(
        self,
        tokenizer,
        sequences,
        token_budget: int,
        extra_toks_per_seq: int = 0,
        add_special_tokens=True,
    ):
        """
        sequences: dict of {label: sequence_string}
        token_budget: max total tokens per batch
        extra_toks_per_seq: e.g., 2 for [CLS] and [SEP]/[EOS]
        """
        self.tokenizer = tokenizer
        self.sequences = list(sequences.items())
        self.token_budget = token_budget
        self.extra_toks_per_seq = extra_toks_per_seq
        self.add_special_tokens = add_special_tokens

        # Compute token lengths for each sequence
        self.token_lengths = self._compute_token_lengths()
        self.batches = self._create_batches()

    def _compute_token_lengths(self):
        _, seqs = zip(*self.sequences)
        lengths = self.tokenizer(
            list(seqs),
            add_special_tokens=self.add_special_tokens,
            return_length=True,
            truncation=False,
            padding=False,
        )["length"]
        return [l + self.extra_toks_per_seq for l in lengths]

    def _create_batches(self):
        sorted_indices = sorted(
            range(len(self.sequences)), key=lambda i: self.token_lengths[i]
        )
        batches = []
        batch = []
        total_tokens = 0

        for idx in sorted_indices:
            seq_len = self.token_lengths[idx]
            if total_tokens + seq_len > self.token_budget:
                if batch:
                    batches.append(batch)
                batch = [idx]
                total_tokens = seq_len
            else:
                batch.append(idx)
                total_tokens += seq_len

        if batch:
            batches.append(batch)

        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class HuggingFaceDataset(Dataset):
    def __init__(self, sequences):
        self.data = list(sequences.items())  # (label, seq)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def check_input_tokens(valid_tokens, sequences, gaps=False):
    print("Checking input sequences for invalid tokens...")
    for i, (label, sequence) in enumerate(sequences.items()):
        if gaps:
            sequence = sequence.split()
        else:
            sequence = re.findall(r"<.*?>|.", sequence)
        if not set(sequence).issubset(valid_tokens):
            # find invalid tokens
            invalid_tokens = set(sequence) - valid_tokens
            raise ValueError(
                f"Invalid tokens in sequence {label}. Please check the alphabet used by the model."
            )
        print(f"Processed {i + 1} out of {len(sequences)} sequences", end="\r")

    print("\nNo invalid tokens in input sequences.")


def fasta_to_dict(fasta_path, max_length, gaps=False, padding=False):
    """Convert FASTA file into a dictionary."""
    print("Reading FASTA file...")

    seq_dict = dict()
    sequence_id = None
    sequence_aa = []

    def _flush_current_seq():
        nonlocal sequence_id, sequence_aa
        if sequence_id is None:
            return
        seq = "".join(sequence_aa)
        if gaps:
            seq_dict[sequence_id] = " ".join(
                re.findall(r"\[.*?\]|.", seq)
            )  # split sequences by space except for special tokens in brackets
        else:
            seq_dict[sequence_id] = seq
        sequence_id = None
        sequence_aa = []

    with open(fasta_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if line.startswith(">"):
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    sequence_id = line
                else:
                    sequence_id = f"seqnum{line_idx:09d}"  # if no id, use line number
            else:
                sequence_aa.append(line)

    _flush_current_seq()

    if gaps:  # if gaps, split sequences by space
        longest_sequence = max(len(seq.split()) for seq in seq_dict.values())
    else:  # if no gaps, split sequences by amino acids and special tokens
        longest_sequence = max(
            len(re.findall(r"<.*?>|.", seq)) for seq in seq_dict.values()
        )

    if max_length == "max_length":
        max_length = longest_sequence
    elif max_length < longest_sequence:
        # raise warning
        print(
            f"Warning: Longest sequence with length {longest_sequence} is longer than the specified max_length {max_length} for padding"
        )
        max_length = longest_sequence
    if not padding:
        return seq_dict, None, max_length
    else:
        padded_seq_dict = dict()
        for label, sequence in seq_dict.items():
            if len(sequence) < max_length:
                padded_seq_dict[label] = sequence + "<pad>" * (
                    max_length - len(re.findall(r"<.*?>|.", sequence))
                )
            else:
                padded_seq_dict[label] = sequence
        return seq_dict, padded_seq_dict, max_length


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
