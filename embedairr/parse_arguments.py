import argparse
from embedairr.model_selecter import supported_models


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2ints(value):
    """Convert a string to a list of integers."""
    if isinstance(value, str):
        if value.lower() == "all":
            return None
        elif value.lower() == "last":
            return [-1]
        else:
            try:
                return [int(x) for x in value.split(" ")]
            except ValueError:
                raise argparse.ArgumentTypeError(
                    "Invalid input. Expected integer(s) or spaced list of integers or 'all' or 'last'."
                )
    elif isinstance(value, int):
        return [value]


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
        type=str2ints,
        nargs="*",
        default=[[-1]],  # TODO: add option to return all layers
        help="Representation layers to extract from the model. Default is the last layer. Example: argument '--layers -1 6' will output the last layer and the sixth layer.",
    )
    parser.add_argument(
        "--extract_logits",
        type=str2bool,
        choices=[True, False],
        default=False,
        help="Set to True to extract logits. Default is False.",
    )
    parser.add_argument(
        "--extract_embeddings",
        choices=["pooled", "unpooled", "false"],
        default=["pooled"],
        nargs="+",
        help="Set the embedding return types. Choose one or more from: 'pooled', 'unpooled', 'false'. Default is 'pooled'.",
    )
    parser.add_argument(
        "--extract_cdr3_embeddings",
        choices=["pooled", "unpooled", "false"],
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
        help="Number of tokens (not sequences!) per batch. Default is 1024.",
    )
    parser.add_argument(
        "--discard_padding",
        action="store_true",
        help="Discard padding tokens from unpooled embeddings output. Default is False.",
    )
    parser.add_argument(
        "--max_length",
        default="max_length",
        help="Length to which sequences will be padded. Default is longest sequence.",
    )
    parser.add_argument(
        "--batch_writing",
        type=str2bool,
        choices=[True, False],
        default=True,
        help="Preallocate output files and write embeddings to disk in batches. Default is True.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for asynchronous data writing. Only relevant when --batch_writing is enabled. Default is 4.",
    )
    parser.add_argument(
        "--disable_special_tokens",
        type=str2bool,
        choices=[True, False],
        default=False,
        help="Disable special tokens in the model. Default is False.",
    )
    parser.add_argument(
        "--ram_limit",
        type=int,
        default=32,
        help="RAM limit in GB for memory usage to buffer outputs for disk writing. Program will pause output computation and flush outputs to disk until under limit. Default is 32.",
    )
    parser.add_argument(
        "--log_memory",
        action="store_true",
        help="Log memory usage to file. Default is False.",
    )
    parser.add_argument(
        "--flatten",
        type=str2bool,
        choices=[True, False],
        default=True,
        help="Flatten the output tensors. Default is False.",
    )
    parser.add_argument(
        "--flush_batches_after",
        type=int,
        default=512,
        help="Size (in MB) of outputs to accumulate in RAM per worker (--num_workers) before flushing to disk. Default is 512.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["float16", "16", "half", "float32", "32", "full"],
        help="Precision of the output data. Inference during embedding is not affected. Default is 'float32'.",
    )

    # TODO add experiment name
    args = parser.parse_args()
    return args
