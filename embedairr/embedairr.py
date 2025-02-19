import argparse
import sys
import os

# Add the parent directory to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from embedairr.model_selecter import select_model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    parser = argparse.ArgumentParser(description="Input path")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name. Example: esm2_t33_650M_UR50D",
    )
    parser.add_argument(
        "--fasta_path", type=str, required=True, help="Fasta path + filename.fa"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory for output files \n Will generate a subdirectory for outputs of each output_type.\n Will output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
    )
    parser.add_argument(
        "--cdr3_path",
        default=None,
        type=str,
        help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.",
    )
    parser.add_argument(
        "--context",
        default=0,
        type=int,
        help="Number of amino acids to include before and after CDR3 sequence",
    )
    parser.add_argument(
        "--layers",
        type=str,
        nargs="?",
        default=["-1"],  # TODO: add option to return all layers
        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.",
    )
    parser.add_argument(
        "--extract_embeddings",
        choices=["pooled", "unpooled", "false"],
        default=["pooled"],
        nargs="+",
        help="Set the embedding return types. Choose one or more from: 'True', 'False', 'unpooled'. Default is 'True'.",
    )
    parser.add_argument(
        "--extract_cdr3_embeddings",
        choices=["pooled", "unpooled", "false"],
        default=["pooled"],
        nargs="+",
        help="Set the CDR3 embedding return types. Choose one or more from: 'True', 'False', 'unpooled'. Requires --cdr3_path to be set. Default is 'True'.",
    )
    parser.add_argument(
        "--extract_attention_matrices",
        choices=["false", "all_heads", "average_layer", "average_all"],
        default=["false"],
        nargs="+",
        help="Set the attention matrix return types. Choose one or more from: 'False', 'all_heads', 'average_layer', 'average_all'. Default is 'False'.",
    )
    parser.add_argument(
        "--extract_cdr3_attention_matrices",
        choices=["false", "all_heads", "average_layer", "average_all"],
        default=["false"],
        nargs="+",
        help="Set the CDR3 attention matrix return types. Choose one or more from: 'False', 'all_heads', 'average_layer', 'average_all'. Requires --cdr3_path to be set. Default is 'False'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for loading sequences. Default is 1024.",
    )
    parser.add_argument(
        "--discard_padding",
        action="store_true",
        help="Discard padding tokens from unpooled embeddings output. Default is False.",
    )
    parser.add_argument(
        "--max_length",
        default=140,
        help="Length to which sequences will be padded. Default is 200.",
    )

    # TODO add experiment name
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse and store arguments

    args = parse_arguments()

    # Check if output directory exists and creates it if it's missing
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    embedder = select_model(args.model_name)

    embedder = embedder(args)

    print("Embedder initialized")

    embedder.run()

    print("All outputs saved.")

# Clear gpu memory
import torch

torch.cuda.empty_cache()
