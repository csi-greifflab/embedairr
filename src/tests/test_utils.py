import os
import sys
import torch
import pytest

from pepe.parse_arguments import parse_arguments, str2bool, str2ints
from pepe import utils


def test_str2bool_and_str2ints():
    assert str2bool("True") is True
    assert str2bool("false") is False
    assert str2ints("1 2 3") == [1, 2, 3]
    assert str2ints("all") is None
    assert str2ints("last") == [-1]


def test_parse_arguments_defaults(tmp_path):
    out_dir = tmp_path / "out"
    sys.argv = [
        "pepe",
        "--model_name",
        "examples/custom_model/example_protein_model",
        "--fasta_path",
        "src/tests/test_files/test.fasta",
        "--output_path",
        str(out_dir),
    ]
    args = parse_arguments()
    assert args.extract_embeddings == ["mean_pooled"]
    assert args.device == "cuda" or args.device == "cpu"
    assert args.experiment_name is None


def test_fasta_to_dict():
    seqs = utils.fasta_to_dict("src/tests/test_files/test.fasta")
    assert len(seqs) == 10
    assert "tz_heavy_77" in seqs


def test_check_input_tokens_invalid():
    seqs = {"bad": "ACDX"}
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    with pytest.raises(ValueError):
        utils.check_input_tokens(valid, seqs, "model")


class DummyTokenizer:
    pad_token_type_id = 0
    all_special_ids = [0]

    def __init__(self):
        self.vocab = {ch: i + 1 for i, ch in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(self, s, truncation=True, padding="max_length", max_length=10, add_special_tokens=True, return_tensors="pt"):
        ids = [self.vocab.get(ch, 1) for ch in s]
        ids = ids[:max_length] + [0] * (max_length - len(ids))
        mask = [1 if id != 0 else 0 for id in ids]
        return type("O", (), {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])})

    def get_vocab(self):
        return self.vocab


def test_huggingface_dataset_collate():
    tokenizer = DummyTokenizer()
    sequences = {"a": "ACD", "b": "EFG"}
    dataset = utils.HuggingFaceDataset(sequences, None, 0, tokenizer, 5)
    sampler = utils.TokenBudgetBatchSampler(dataset, token_budget=10)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.safe_collate)

    batch = next(iter(loader))
    labels, seqs, toks, masks, substr = batch
    assert labels == ["a", "b"]
    assert toks.shape == (2, 5)
    assert masks.shape == (2, 5)
    assert substr is None
