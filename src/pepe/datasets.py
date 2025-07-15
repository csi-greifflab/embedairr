import logging
import re
from typing import Sequence

import torch
from torch.utils.data import Dataset
from transformers import RoFormerTokenizer
from alive_progress import alive_bar

logger = logging.getLogger("src.datasets")


class TokenBudgetBatchSampler:
    def __init__(self, dataset, token_budget):
        self.dataset = dataset
        self.token_budget = token_budget
        sample_seq_len = len(dataset[0][2])
        self.batch_size = token_budget // sample_seq_len
        self.batches = self._create_batches()

    def _create_batches(self):
        indices = list(range(len(self.dataset)))
        return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class SequenceDictDataset(Dataset):
    def __init__(self, sequences, substring_dict, context):
        self.data = list(sequences.items())
        if substring_dict:
            self.substring_dict = substring_dict
            self.context = context
            self.filtered_substring_data = self._filter_substrings()
        else:
            self.substring_dict = None
            self.context = None
            self.filtered_substring_data = None

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _filter_substrings(self):
        labels, _ = zip(*self.data)
        filtered_substring_dict = {label: self.substring_dict[label] for label in labels if label in self.substring_dict}
        assert len(filtered_substring_dict) == len(self.data), "Not all sequences have matching substrings."
        return filtered_substring_dict.items()

    def _get_substring_masks(self):
        full_sequence_tokens = [entry[2] for entry in self.encoded_data]
        substring_tokens = [entry[2] for entry in self.encoded_substring_data]
        masks = [self._find_subsequence(full_seq, substring, self.pad_token_id) for full_seq, substring in zip(full_sequence_tokens, substring_tokens)]
        return list(masks)

    def _find_subsequence(self, full_tensor, subtensor, pad_token_id=0):
        subsequence_mask = torch.zeros_like(full_tensor)
        trimmed_subtensor = subtensor[subtensor != pad_token_id]
        trimmed_subtensor_length = trimmed_subtensor.size(0)
        if trimmed_subtensor_length == 0:
            return subsequence_mask

        full_tensor_length = full_tensor.size(0)
        for start in range(full_tensor_length):
            match_positions = []
            full_tensor_index = start
            subtensor_index = 0
            while full_tensor_index < full_tensor_length and subtensor_index < trimmed_subtensor_length:
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
    def __init__(self, sequences, substring_dict, context, tokenizer, max_length, add_special_tokens=True):
        super().__init__(sequences, substring_dict, context)
        self.encoded_data = self._encode_sequences(self.data, tokenizer, max_length, add_special_tokens)
        self.pad_token_id = tokenizer.pad_token_type_id
        if self.substring_dict:
            logger.info("Tokenizing substrings...")
            self.encoded_substring_data = self._encode_sequences(self.filtered_substring_data, tokenizer, "max_length", add_special_tokens=False)
            self.substring_masks = self._get_substring_masks()

    def _encode_sequences(self, data, tokenizer, max_length, add_special_tokens):
        labels, strs = zip(*data)
        if isinstance(tokenizer, RoFormerTokenizer):
            logger.info("Using RoFormerTokenizer, applying gap_sequence.")
            strs = self._gap_sequence(strs)
            max_token_length = max(len(seq.split(" ")) for seq in strs)
        else:
            max_token_length = max(len(seq) for seq in strs)
        if max_length == "max_length":
            max_length = max_token_length
            logger.info(f"Setting max_length to {max_length}.")
        elif isinstance(max_length, int) and max_length < max_token_length:
            logger.warning(
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
        return [" ".join(re.findall(r"\[.*?\]|.", sequence)) for sequence in sequences]

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks, _ in self.encoded_data)

    def __getitem__(self, idx):
        labels, seqs, toks, attention_masks = self.encoded_data[idx]
        if self.substring_dict:
            substring_masks = self.substring_masks[idx]
            return labels, seqs, toks, attention_masks, substring_masks
        else:
            return labels, seqs, toks, attention_masks, None

    def safe_collate(self, batch):
        if self.substring_dict:
            labels, seqs, toks, attention_matrices, substring_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                torch.stack(attention_matrices),
                torch.stack(substring_masks),
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
    def __init__(self, sequences, substring_dict, context, alphabet, max_length, prepend_bos=True, append_eos=True):
        super().__init__(sequences, substring_dict, context)
        self.encoded_data = self._encode_sequences(
            self.data,
            alphabet,
            max_length,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )
        self.pad_token_id = alphabet.padding_idx
        if self.substring_dict:
            logger.info("Tokenizing substrings...")
            self.encoded_substring_data = self._encode_sequences(
                self.filtered_substring_data,
                alphabet,
                "max_length",
                prepend_bos=False,
                append_eos=False,
            )
            logger.info("Matching substrings to full sequences...")
            self.substring_masks = self._get_substring_masks()

    def _encode_sequences(self, data, alphabet, max_length, prepend_bos=True, append_eos=True):
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
            logger.warning(
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
            tokens[i, int(prepend_bos) : len(seq_encoded) + int(prepend_bos)] = seq
            if append_eos:
                tokens[i, len(seq_encoded) + int(prepend_bos)] = alphabet.eos_idx
        return list(zip(labels, strs, list(tokens)))

    def get_max_encoded_length(self):
        return max(len(toks) for _, _, toks in self.encoded_data)

    def safe_collate(self, batch):
        if self.substring_dict:
            labels, seqs, toks, _, substring_masks = zip(*batch)
            return (
                list(labels),
                list(seqs),
                torch.stack(toks),
                None,
                torch.stack(substring_masks),
            )
        else:
            labels, seqs, toks, _, _ = zip(*batch)
            return list(labels), list(seqs), torch.stack(toks), None, None

    def __getitem__(self, idx):
        labels, seqs, toks = self.encoded_data[idx]
        if self.substring_dict:
            substring_masks = self.substring_masks[idx]
            return labels, seqs, toks, None, substring_masks
        else:
            return labels, seqs, toks, None, None
