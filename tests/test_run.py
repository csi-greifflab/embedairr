import sys
import os
from embedairr.__main__ import parse_arguments
from embedairr.model_selecter import select_model


# os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = [
    "embedairr.py",
    "--experiment_name",
    "test",
    "--model_name",
    "esm2_t33_650M_UR50D",
    "--fasta_path",
    "tests/test_files/test.fasta",
    "--output_path",
    "tests/test_files/test_output",
    "--cdr3_path",
    "tests/test_files/test_cdr3.csv",
    "--extract_embeddings",
    "pooled",
    "unpooled",
    "--extract_cdr3_embeddings",
    "pooled",
    "--extract_attention_matrices",
    "all_heads",
    "--batch_writing",
    "false",
]

args = parse_arguments()

# Check if output directory exists and creates it if it's missing
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

embedder = select_model(args.model_name)

embedder = embedder(args)
print("Embedder initialized")

embedder.run()
print("All outputs saved.")
