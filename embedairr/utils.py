from typing import Sequence, Tuple
from torch.utils.data import Dataset
import re
import torch
import gc
import os
import psutil
import numpy as np
import tracemalloc
from transformers import RoFormerTokenizer


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


class SequenceDictDataset(Dataset):
    def __init__(self, sequences, cdr3_dict, context):
        self.data = list(sequences.items())  # (label, seq)
        if cdr3_dict:
            self.cdr3_dict = cdr3_dict
            self.context = context
            self.filtered_cdr3_data = self.filter_cdr3()
        else:
            self.cdr3_dict = None
            self.context = None
            self.filtered_cdr3_data = None
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def filter_cdr3(self):
        """Filter the cdr3_dict to only include sequences that are in the dataset."""
        labels, _ = zip(*self.data)
        filtered_cdr3_dict = {
            label: self.cdr3_dict[label] for label in labels if label in self.cdr3_dict
        }
        assert len(filtered_cdr3_dict) == len(
            self.data
        ), "Not all sequences have matching CDR3 sequences."
        return filtered_cdr3_dict.items()

    def get_subsequence_masks(self):
        """Get the subsequence masks for the CDR3 sequences."""
        # Get the full sequences and CDR3 sequences
        if torch.cuda.is_available():
            full_sequence_tokens = [entry[2].cuda() for entry in self.encoded_data]
            cdr3_sequences_tokens = [
                entry[2].cuda() for entry in self.encoded_cdr3_data
            ]
        else:
            full_sequence_tokens = [entry[2] for entry in self.encoded_data]
            cdr3_sequences_tokens = [entry[2] for entry in self.encoded_cdr3_data]

        # Create masks for each sequence
        masks = [
            self.find_subsequence(full_seq, cdr3_seq, self.pad_token_id).cpu()
            for full_seq, cdr3_seq in zip(full_sequence_tokens, cdr3_sequences_tokens)
        ]

        return list(masks)

    def find_subsequence(self, full_tensor, subtensor, pad_token_id=0):
        subsequence_mask = torch.zeros_like(full_tensor)

        # Remove padding from b
        trimmed_subtensor = subtensor[subtensor != pad_token_id]
        trimmed_subtensor_length = trimmed_subtensor.size(0)
        if trimmed_subtensor_length == 0:
            return subsequence_mask

        full_tensor_length = full_tensor.size(0)

        for start in range(full_tensor_length):
            match_positions = []
            full_tensor_index = start
            subtensor_index = 0

            while (
                full_tensor_index < full_tensor_length
                and subtensor_index < trimmed_subtensor_length
            ):
                if full_tensor[full_tensor_index] == pad_token_id:
                    full_tensor_index += 1
                    continue
                if full_tensor[full_tensor_index] == trimmed_subtensor[subtensor_index]:
                    match_positions.append(full_tensor_index)
                    full_tensor_index += 1
                    subtensor_index += 1
                else:
                    break

            if subtensor_index == trimmed_subtensor_length:
                subsequence_mask[match_positions] = 1

        return subsequence_mask


class HuggingFaceDataset(SequenceDictDataset):
    def __init__(
        self,
        sequences,
        cdr3_dict,
        context,
        tokenizer,
        max_length,
        add_special_tokens=True,
    ):
        super().__init__(sequences, cdr3_dict, context)
        self.encoded_data = self.encode_sequences(
            self.data, tokenizer, max_length, add_special_tokens
        )  # (label, seq, toks, attention_mask)
        self.pad_token_id = tokenizer.pad_token_type_id
        if self.cdr3_dict:
            print("Tokenizing CDR3 sequences...")
            self.encoded_cdr3_data = self.encode_sequences(
                self.filtered_cdr3_data,
                tokenizer,
                "max_length",
                add_special_tokens=False,
            )  # (label, seq, toks, attention_mask)
            self.cdr3_masks = self.get_subsequence_masks()

    def encode_sequences(self, data, tokenizer, max_length, add_special_tokens):
        labels, strs = zip(*data)
        if isinstance(tokenizer, RoFormerTokenizer):
            # RoFormerTokenizer requires space-separated tokenization
            # for the input sequences
            print("Using RoFormerTokenizer, applying gap_sequence.")
            strs = self.gap_sequence(strs)
            max_token_length = max(len(seq.split(" ")) for seq in strs)
        else:
            # For other tokenizers, use the default tokenization
            max_token_length = max(len(seq) for seq in strs)

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

    def gap_sequence(self, sequences: Sequence[str]) -> Sequence[str]:
        """Space-separated tokenization for RoFormer input."""
        seqs = [" ".join(re.findall(r"\[.*?\]|.", sequence)) for sequence in sequences]
        return seqs

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks, _ in self.encoded_data)

    def __getitem__(self, idx):
        labels, seqs, toks, attention_masks = self.encoded_data[idx]
        if self.cdr3_dict:
            cdr3_masks = self.cdr3_masks[idx]
            return labels, seqs, toks, attention_masks, cdr3_masks
        else:
            return labels, seqs, toks, attention_masks, None

    def safe_collate(self, batch):
        if self.cdr3_dict:
            labels, seqs, toks, attention_matrices, cdr3_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attention_matrices),
                torch.stack(cdr3_masks),
            )
        else:
            labels, seqs, toks, attention_matrices, _ = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attention_matrices),
                None,
            )

    def __len__(self):
        return len(self.data)


class ESMDataset(SequenceDictDataset):
    def __init__(
        self,
        sequences,
        cdr3_dict,
        context,
        alphabet,
        max_length,
        prepend_bos=True,
        append_eos=True,
    ):
        super().__init__(sequences, cdr3_dict, context)
        self.encoded_data = self.encode_sequences(
            self.data,
            alphabet,
            max_length,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )  # (label, seq, toks)
        self.pad_token_id = alphabet.padding_idx
        if self.cdr3_dict:
            print("Tokenizing CDR3 sequences...")
            self.encoded_cdr3_data = self.encode_sequences(
                self.filtered_cdr3_data,
                alphabet,
                "max_length",
                prepend_bos=False,
                append_eos=False,
            )  # (label, seq, toks)
            print("Matching CDR3 sequences to full sequences...")
            self.cdr3_masks = self.get_subsequence_masks()

    def encode_sequences(
        self, data, alphabet, max_length, prepend_bos=True, append_eos=True
    ):
        labels, strs = zip(*data)
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

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks in self.encoded_data)

    def safe_collate(self, batch):
        labels, seqs, toks, _, cdr3_masks = zip(*batch)
        return (
            list(labels),
            list(seqs),
            torch.stack(toks),
            None,
            torch.stack(cdr3_masks),
        )

    def __getitem__(self, idx):
        labels, seqs, toks = self.encoded_data[idx]
        if self.cdr3_dict:
            cdr3_masks = self.cdr3_masks[idx]
            return labels, seqs, toks, None, cdr3_masks
        else:
            return labels, seqs, toks, None, None


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
        # print(f"Processed {i + 1} out of {len(sequences)} sequences", end="\r")

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
