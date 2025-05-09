from typing import Sequence
from torch.utils.data import Dataset
import re
import torch
import gc
from transformers import RoFormerTokenizer
import threading, queue
from alive_progress import alive_bar


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
            self.filtered_cdr3_data = self._filter_cdr3()
        else:
            self.cdr3_dict = None
            self.context = None
            self.filtered_cdr3_data = None
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _filter_cdr3(self):
        """Filter the cdr3_dict to only include sequences that are in the dataset."""
        labels, _ = zip(*self.data)
        filtered_cdr3_dict = {
            label: self.cdr3_dict[label] for label in labels if label in self.cdr3_dict
        }
        assert len(filtered_cdr3_dict) == len(
            self.data
        ), "Not all sequences have matching CDR3 sequences."
        return filtered_cdr3_dict.items()

    def _get_subsequence_masks(self):
        """Get the subsequence masks for the CDR3 sequences."""
        # Get the full sequences and CDR3 sequences
        full_sequence_tokens = [entry[2] for entry in self.encoded_data]
        cdr3_sequences_tokens = [entry[2] for entry in self.encoded_cdr3_data]

        # Create masks for each sequence
        masks = [
            self._find_subsequence(full_seq, cdr3_seq, self.pad_token_id)
            for full_seq, cdr3_seq in zip(full_sequence_tokens, cdr3_sequences_tokens)
        ]

        return list(masks)

    def _find_subsequence(self, full_tensor, subtensor, pad_token_id=0):
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
        self.encoded_data = self._encode_sequences(
            self.data, tokenizer, max_length, add_special_tokens
        )  # (label, seq, toks, attention_mask)
        self.pad_token_id = tokenizer.pad_token_type_id
        if self.cdr3_dict:
            print("Tokenizing CDR3 sequences...")
            self.encoded_cdr3_data = self._encode_sequences(
                self.filtered_cdr3_data,
                tokenizer,
                "max_length",
                add_special_tokens=False,
            )  # (label, seq, toks, attention_mask)
            self.cdr3_masks = self._get_subsequence_masks()

    def _encode_sequences(self, data, tokenizer, max_length, add_special_tokens):
        labels, strs = zip(*data)
        if isinstance(tokenizer, RoFormerTokenizer):
            # RoFormerTokenizer requires space-separated tokenization
            # for the input sequences
            print("Using RoFormerTokenizer, applying gap_sequence.")
            strs = self._gap_sequence(strs)
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
        loop_input_ids = []
        loop_attention_mask = []
        with alive_bar(len(strs), title="Tokenizing sequences...") as bar:
            for s in strs:
                out = tokenizer(
                    s,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                    return_tensors="pt",
                )
                loop_input_ids.append(out.input_ids)
                loop_attention_mask.append(out.attention_mask)
                bar()

        toks = torch.cat(loop_input_ids, dim=0)
        attention_masks = torch.cat(loop_attention_mask, dim=0)
        return list(zip(labels, strs, list(toks), list(attention_masks)))

    def _gap_sequence(self, sequences: Sequence[str]) -> Sequence[str]:
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
        self.encoded_data = self._encode_sequences(
            self.data,
            alphabet,
            max_length,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )  # (label, seq, toks)
        self.pad_token_id = alphabet.padding_idx
        if self.cdr3_dict:
            print("Tokenizing CDR3 sequences...")
            self.encoded_cdr3_data = self._encode_sequences(
                self.filtered_cdr3_data,
                alphabet,
                "max_length",
                prepend_bos=False,
                append_eos=False,
            )  # (label, seq, toks)
            print("Matching CDR3 sequences to full sequences...")
            self.cdr3_masks = self._get_subsequence_masks()

    def _encode_sequences(
        self, data, alphabet, max_length, prepend_bos=True, append_eos=True
    ):
        labels, strs = zip(*data)
        encoded = []
        with alive_bar(len(strs), title="Tokenizing sequences...") as bar:
            for s in strs:
                seq_encoded = alphabet.encode(s)
                encoded.append(seq_encoded)
                bar()

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
        if self.cdr3_dict:
            labels, seqs, toks, _, cdr3_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                None,
                torch.stack(cdr3_masks),
            )
        else:
            labels, seqs, toks, _, _ = zip(*batch)
            return list(labels), list(seqs), torch.stack(toks), None, None

    def __getitem__(self, idx):
        labels, seqs, toks = self.encoded_data[idx]
        if self.cdr3_dict:
            cdr3_masks = self.cdr3_masks[idx]
            return labels, seqs, toks, None, cdr3_masks
        else:
            return labels, seqs, toks, None, None


def check_input_tokens(valid_tokens, sequences, model_name):
    with alive_bar(
        len(sequences), title="Checking input sequences for invalid tokens..."
    ) as bar:
        for label, sequence in sequences.items():
            if "esm" not in model_name:
                sequence = re.findall(r"\[.*?\]|.", sequence)
            else:
                sequence = re.findall(r"<.*?>|.", sequence)
            if not set(sequence).issubset(valid_tokens):
                raise ValueError(
                    f"Invalid tokens found in sequence {label}: {set(sequence) - set(valid_tokens)}"
                )
            bar()
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


class IOFlushWorker(threading.Thread):
    def __init__(self, memmap_registry, flush_bytes_limit=64 * 1024 * 1024):
        super().__init__()
        self.memmap_registry = memmap_registry
        self.flush_limit = flush_bytes_limit
        self.write_q = queue.Queue(maxsize=128)
        self.buffers = {}  # key → {'active': [], 'flushing': [], 'bytes': int}
        self.total_buffered = 0
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.flush_executor = threading.Thread(target=self._background_flusher)
        self.flush_signal = threading.Event()

    def run(self):
        self.flush_executor.start()
        while True:
            item = self.write_q.get()
            if item is None:
                break
            key, offset, array = item
            arr_bytes = array.nbytes

            with self.lock:
                buf = self.buffers.setdefault(
                    key, {"active": [], "flushing": [], "bytes": 0}
                )
                buf["active"].append((offset, array))
                buf["bytes"] += arr_bytes
                self.total_buffered += arr_bytes

                if self.total_buffered >= self.flush_limit:
                    self._swap_buffers()
                    self.flush_signal.set()

        self._swap_buffers()
        self.flush_signal.set()
        self.shutdown_flag.set()
        self.flush_executor.join()

    def _swap_buffers(self):
        for buf in self.buffers.values():
            buf["active"], buf["flushing"], buf["bytes"] = [], buf["active"], 0

    def _background_flusher(self):
        while not self.shutdown_flag.is_set() or any(
            buf["flushing"] for buf in self.buffers.values()
        ):
            self.flush_signal.wait(timeout=1)
            self.flush_signal.clear()

            with self.lock:
                for key, buf in self.buffers.items():
                    if buf["flushing"]:
                        self._flush_key(key, buf)

    def _flush_key(self, key, buf):
        mmap_handle = self.memmap_registry[key]
        for offset, arr in buf["flushing"]:
            mmap_handle[offset : offset + arr.shape[0]] = arr
        mmap_handle.flush()
        self.total_buffered -= buf["bytes"]
        buf["bytes"] = 0
        buf["flushing"].clear()

    def enqueue(self, output_type, layer, head, offset, array):
        key = (output_type, layer, head)
        self.write_q.put((key, offset, array))

    def stop(self):
        self.write_q.put(None)
        self.join()


class MultiIODispatcher:
    def __init__(
        self, memmap_registry, num_workers=4, flush_bytes_limit=64 * 1024 * 1024
    ):
        self.num_workers = num_workers
        self.workers = []

        # Distribute files across workers by hashing the key
        sharded_registries = [{} for _ in range(num_workers)]
        for key, mmap in memmap_registry.items():
            shard_id = hash(key) % num_workers
            sharded_registries[shard_id][key] = mmap

        for i in range(num_workers):
            worker = IOFlushWorker(
                memmap_registry=sharded_registries[i],
                flush_bytes_limit=flush_bytes_limit,
            )
            worker.start()
            self.workers.append(worker)

    def enqueue(self, output_type, layer, head, offset, array):
        key = (output_type, layer, head)
        shard_id = hash(key) % self.num_workers
        self.workers[shard_id].enqueue(output_type, layer, head, offset, array)

    def stop(self):
        for worker in self.workers:
            worker.stop()
