import argparse
from embedairr.model_selecter import select_model

supported_models = [
    "esm1_t34_670M_UR50S",
    "esm1_t34_670M_UR50D",
    "esm1_t34_670M_UR100",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
    "esm1b_t33_650M_UR50S",
    #'esm_msa1_t12_100M_UR50S',
    #'esm_msa1b_t12_100M_UR50S',
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
    #'esm_if1_gvp4_t16_142M_UR50',
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
    "Rostlab/prot_t5_xl_half_uniref50-enc",
    "Rostlab/ProstT5",
    "alchemab/antiberta2-cssp",
    "alchemab/antiberta2",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Parsing command-line arguments for input and output file paths
def parse_arguments():
    """Parse command-line arguments for input and output file paths."""
    parser = argparse.ArgumentParser(description="Input path")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment. Will be used to name the output files. If not provided, the output files will be named after the input file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=supported_models,
        help="Model name. Example: esm2_t33_650M_UR50D",
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        required=True,
        help="Path to the input FASTA file. Required. If no experiment name is provided, the output files will be named after the input file.",
    )
    parser.add_argument(
        "--output_path",
        type=str.lower,
        required=True,
        help="Directory for output files \n Will generate a subdirectory for outputs of each output_type.\n Will output multiple files if multiple layers are specified with '--layers'. Output file is a single tensor or a list of tensors when --pooling is False.",
    )
    parser.add_argument(
        "--cdr3_path",
        default=None,
        type=str.lower,
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
        default="-1",  # TODO: add option to return all layers
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
        choices=["pooled", "false"],
        default=["false"],
        nargs="+",
        help="Set the CDR3 embedding return types. Choose one or more from: 'pooled', 'false'. Requires --cdr3_path to be set. Default is 'false'.",
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
        choices=["false", "average_layer", "average_all"],
        default=["false"],
        nargs="+",
        help="Set the CDR3 attention matrix return types. Choose one or more from: 'False', 'average_layer', 'average_all'. Requires --cdr3_path to be set. Default is 'False'.",
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
        help="Length to which sequences will be padded. Default is 140.",
    )
    parser.add_argument(
        "--batch_writing",
        type=str2bool,
        choices=[True, False],
        default=True,
        help="Preallocate output files and write embeddings to disk in batches. Default is True.",
    )
    parser.add_argument(
        "--disable_special_tokens",
        type=str2bool,
        choices=[True, False],
        default=False,
        help="Disable special tokens in the model. Default is False.",
    )

    # TODO add experiment name
    args = parser.parse_args()
    return args


def main():
    # Parse and store arguments

    args = parse_arguments()
    if args.batch_writing and args.extract_cdr3_attention_matrices != "false":
        raise ValueError(
            "Batch writing is not supported with CDR3 attention matrices. Set '--batch_writing False' to disable."
        )

    embedder = select_model(args.model_name)

    embedder = embedder(args)

    print("Embedder initialized")

    embedder.run()

    print("All outputs saved.")


if __name__ == "__main__":
    main()
