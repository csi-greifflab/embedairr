import argparse
import os
from embedairr.model_selecter import select_model


# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    PARSER = argparse.ArgumentParser(description="Input path")
    PARSER.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name. Example: esm2_t33_650M_UR50D",
    )
    PARSER.add_argument(
        "--fasta_path", type=str, required=True, help="Fasta path + filename.fa"
    )
    PARSER.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory for output files \n Will generate a subdirectory for outputs of each output_type.\nWill output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
    )
    PARSER.add_argument(
        "--cdr3_path",
        default=None,
        type=str,
        help="Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.",
    )
    PARSER.add_argument(
        "--context",
        default=0,
        type=int,
        help="Number of amino acids to include before and after CDR3 sequence",
    )
    PARSER.add_argument(
        "--layers",
        type=str,
        nargs="?",
        default=["-1"],  # TODO: add option to return all layers
        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.",
    )
    PARSER.add_argument(
        "--pooling",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to pool the embeddings or not. Default is True.",
    )
    PARSER.add_argument(
        "--return_embeddings",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to pool the embeddings or not. Default is True.",
    )
    PARSER.add_argument(
        "--pool_cdr3",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Whether to pool the CDR3 extracted embeddings or not. Default is True.",
    )
    # TODO add argument return_attention_matrix
    # TODO add argument for batch_size
    # TODO add experiment name
    output_types = get_output_types(PARSER.parse_args())
    return PARSER.parse_args(), output_types


def get_output_types(args):
    output_types_dict = {
        "embeddings": args.return_embeddings is True,
        "embeddings_unpooled": args.pooling is False,
        "cdr3_extracted": args.cdr3_path
        and not args.context
        and args.pool_cdr3 is True,
        "cdr3_extracted_unpooled": args.cdr3_path
        and not args.context
        and args.pool_cdr3 is False,
        "cdr3_context_extracted": args.cdr3_path
        and args.context
        and args.pool_cdr3 is True,
        "cdr3_context_extracted_unpooled": args.cdr3_path
        and args.context
        and args.pool_cdr3 is False,
        # TODO: add options for attention matrix and all layers
    }

    # Filter out the output types that are not enabled
    return [key for key, condition in output_types_dict.items() if condition]


if __name__ == "__main__":
    import sys

    print(sys.argv)
    # Parse and store arguments
    args, output_types = parse_arguments()
    model_name = args.model_name
    fasta_path = args.fasta_path
    output_path = args.output_path
    cdr3_path = args.cdr3_path
    context = args.context
    layers = list(map(int, args.layers.strip().split()))
    pooling = bool(args.pooling)
    cdr3_pooling = bool(args.pool_cdr3)

    print("Arguments parsed successfully")
    # Print summary of arguments
    print(f"FASTA file: {fasta_path}")
    print(f"Output file: {output_path}")
    print(f"CDR3 file: {cdr3_path}")
    print(f"Context: {context}")
    print(f"Layers: {layers}")
    print(f"Pooling: {pooling}\n")
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
        pooling,
        output_types,
    )

    print("Embedder initialized")

    embedder.run()

    print("All outputs saved.")
