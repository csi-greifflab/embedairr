import argparse
import sys
import os

# Add the parent directory to the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from embedairr.model_selecter import select_model


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
    # TODO add argument for batch_size
    # TODO add experiment name
    args = parser.parse_args()
    output_types = get_output_types(args)
    return args, output_types


# When changes made here, also update base_embedder.py BaseEmbedder.extract_batch() method.
def get_output_types(args):
    output_types = []

    # Process embeddings options
    if "pooled" in args.extract_embeddings:
        output_types.append("embeddings")
    if "unpooled" in args.extract_embeddings:
        output_types.append("embeddings_unpooled")

    # Process cdr3 embeddings options
    if args.cdr3_path:
        if "pooled" in args.extract_cdr3_embeddings:
            output_types.append("cdr3_extracted")
        if "unpooled" in args.extract_cdr3_embeddings:
            output_types.append("cdr3_extracted_unpooled")

    # Process attention matrices options
    if "average_all" in args.extract_attention_matrices:
        output_types.append("attention_matrices_average_all")
    if "average_layer" in args.extract_attention_matrices:
        output_types.append("attention_matrices_average_layer")
    if "all_heads" in args.extract_attention_matrices:
        output_types.append("attention_matrices_all_heads")

    # Process cdr3 attention matrices options
    if args.cdr3_path:
        if "average_all" in args.extract_cdr3_attention_matrices:
            output_types.append("cdr3_attention_matrices_average_all")
        if "average_layer" in args.extract_cdr3_attention_matrices:
            output_types.append("cdr3_attention_matrices_average_layer")
        if "all_heads" in args.extract_cdr3_attention_matrices:
            output_types.append("cdr3_attention_matrices_all_heads")

    return output_types


if __name__ == "__main__":
    # Parse and store arguments
    args, output_types = parse_arguments()
    model_name = args.model_name
    fasta_path = args.fasta_path
    output_path = args.output_path
    cdr3_path = args.cdr3_path
    context = args.context
    layers = list(map(int, args.layers.strip().split()))

    print("Arguments parsed successfully")
    # Print summary of arguments
    print(f"FASTA file: {fasta_path}")
    print(f"Output file: {output_path}")
    print(f"CDR3 file: {cdr3_path}")
    print(f"Context: {context}")
    print(f"Layers: {layers}")
    print(f"Output types: {output_types}\n")

    # Check if output directory exists and creates it if it's missing
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    embedder = select_model(model_name)

    embedder = embedder(
        fasta_path,
        model_name,
        output_path,
        cdr3_path,
        context,
        layers,
        output_types,
    )

    print("Embedder initialized")

    embedder.run()

    print("All outputs saved.")

# sys.argv = [
#   "embedairr.py",
#   "--model_name",
#    "ab2",
#    "--fasta_path",
#    "/doctorai/userdata/airr_atlas/data/sequences/test_500.fa",
#    "--output_path",
#    "data/embeddings/test/",
#    "--layers",
#    "-1",
#    "--extract_embeddings",
#    "pooled",
#    "--extract_attention_matrices",
#    "average_all",
# ]
