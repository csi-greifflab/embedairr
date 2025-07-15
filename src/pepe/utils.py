import logging

from .datasets import (
    SequenceDictDataset,
    HuggingFaceDataset,
    ESMDataset,
    TokenBudgetBatchSampler,
)
from .io_utils import (
    IOFlushWorker,
    MultiIODispatcher,
    flush_memmaps,
    check_disk_free_space,
)

logger = logging.getLogger("src.utils")


def check_input_tokens(valid_tokens, sequences, model_name):
    from alive_progress import alive_bar
    import re

    with alive_bar(len(sequences), title="Checking input sequences for invalid tokens...") as bar:
        for label, sequence in sequences.items():
            if "esm" not in model_name:
                sequence = re.findall(r"\[.*?\]|.", sequence)
                if "antiberta" in model_name:
                    assert len(sequence) <= 256, (
                        "Antiberta2 does not support sequences longer than 256 tokens. "
                        f"Found {len(sequence)} tokens in sequence {label}."
                    )
            else:
                sequence = re.findall(r"<.*?>|.", sequence)
            if not set(sequence).issubset(valid_tokens):
                raise ValueError(f"Invalid tokens found in sequence {label}: {set(sequence) - set(valid_tokens)}")
            bar()
    logger.info("No invalid tokens in input sequences.")


def fasta_to_dict(fasta_path):
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
