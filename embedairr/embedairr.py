import argparse
import os
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
        help="Directory for output files \n Will generate a subdirectory for outputs of each output_type.\nWill output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
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
        type=return_embeddings_type,
        default=True,
        nargs="?",
        help="Set the embedding return type: None, True, False, or 'unpooled'.",
    )
    parser.add_argument(
        "--extract_cdr3_embeddings",
        type=return_embeddings_type,
        nargs="?",
        default=True,
        help="Whether to pool the CDR3 extracted embeddings or not. Default is True.",
    )
    parser.add_argument(
        "--extract_attention_matrices",
        type=return_attention_matrix_type,
        nargs="?",
        default="pooled",
        help="Whether to pool the CDR3 extracted embeddings or not. Default is True.",
    )
    # TODO add argument return_attention_matrix
    # TODO add argument for batch_size
    # TODO add experiment name
    output_types = get_output_types(parser.parse_args())
    return parser.parse_args(), output_types


def return_embeddings_type(value):
    if value.lower() in ("true", "false", "unpooled", ""):
        if (
            value.lower()
            == "true" | value.lower()
            == "t" | value.lower()
            == "yes" | value.lower()
            == "y" | value.lower()
            == "1"
        ):
            return True
        elif (
            value.lower()
            == "false" | value.lower()
            == "f" | value.lower()
            == "no" | value.lower()
            == "n" | value.lower()
            == "0"
        ):
            return False
        elif value.lower() == "unpooled":
            return "unpooled"
        else:
            return True  # Represents empty (default) case
    raise argparse.ArgumentTypeError(
        "Value must be empty, 'true', 'false', or 'unpooled'."
    )


def return_attention_matrix_type(value):
    if value.lower() in ("true", "false", "pooled", "layer_pooled", "unpooled", ""):
        if (
            value.lower()
            == "true" | value.lower()
            == "t" | value.lower()
            == "yes" | value.lower()
            == "y" | value.lower()
            == "1"
        ):
            return "pooled"
        elif (
            value.lower()
            == "false" | value.lower()
            == "f" | value.lower()
            == "no" | value.lower()
            == "n" | value.lower()
            == "0"
        ):
            return False
        elif value.lower() == "pooled":
            return "pooled"
        elif value.lower() == "layer_pooled":
            return "layer_pooled"
        elif value.lower() == "unpooled":
            return "unpooled"
        elif value.lower() == "":
            return "pooled"
        else:
            return False  # Represents empty (default) case
    raise argparse.ArgumentTypeError(
        "Value must be empty, 'true', 'false', 'pooled' or 'layer_pooled'."
    )


# When changes made here, also update base_embedder.py BaseEmbedder.extract_batch() method.
def get_output_types(args):
    output_types_dict = {
        "embeddings": args.extract_embeddings is True and not bool(args.cdr3_path),
        "embeddings_unpooled": args.extract_embeddings is "unpooled",
        "cdr3_extracted": args.extract_cdr3_embeddings is True and bool(args.cdr3_path),
        "cdr3_extracted_unpooled": args.extract_cdr3_embeddings is "unpooled",
        "attention_matrices_average_all": args.extract_attention_matrices is "pooled"
        and not bool(args.cdr3_path),
        "attention_matrices_average_layer": args.extract_attention_matrices
        is "layer_pooled"
        and not bool(args.cdr3_path),
        "attention_matrices_all_heads": args.extract_attention_matrices is "unpooled"
        and not bool(args.cdr3_path),
        "cdr3_attention_matrices_average_all": args.extract_attention_matrices
        is "pooled"
        and bool(args.cdr3_path),
        "cdr3_attention_matrices_average_layer": args.extract_attention_matrices
        is "layer_pooled"
        and bool(args.cdr3_path),
        "cdr3_attention_matrices_all_heads": args.extract_attention_matrices
        is "unpooled"
        and bool(args.cdr3_path),
    }
    # Filter out the output types that are not enabled
    output_types = [key for key, condition in output_types_dict.items() if condition]
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
