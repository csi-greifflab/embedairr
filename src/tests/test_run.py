import os
import sys

from pepe.__main__ import parse_arguments
from pepe.model_selecter import select_model


def test_run_pipeline():
    """Run the embedding pipeline on the example data."""

    sys.argv = [
        "pepe",
        "--experiment_name",
        "test",
        "--model_name",
        "examples/custom_model/example_protein_model",
        "--fasta_path",
        "src/tests/test_files/test.fasta",
        "--output_path",
        "src/tests/test_files/test_output",
        "--substring_path",
        "src/tests/test_files/test_substring.csv",
        "--extract_embeddings",
        "mean_pooled",
        "per_token",
        "substring_pooled",
        "attention_head",
        "--device",
        "cpu",
    ]

    args = parse_arguments()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    embedder_cls = select_model(args.model_name)
    embedder = embedder_cls(args)
    embedder.run()
