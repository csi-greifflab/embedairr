import argparse
import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#max split XLA 512
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
# Add the parent directory to the PYTHONPATH
current_dir = os.getcwd()
parent_dir = os.path.dirname("/doctorai/niccoloc/embedairr/embedairr/esm2_embedder.py")
sys.path.insert(0, parent_dir)
from embedairr.model_selecter import select_model
from embedairr.model_selecter import select_model
#import embedairr.model_selector


# import torch

# # Step 1: Create a list of four random tensors, each of shape [231, 231]
# tensor_list = [torch.rand(231, 231) for _ in range(4)]

# # Step 2: Stack the list of tensors along a new dimension
# stacked_tensor = torch.stack(tensor_list, dim=0)

# # Step 3: Print the shape to verify
# print("Shape of each original tensor:", tensor_list[0].shape)  # Should be torch.Size([231, 231])
# print("Shape of stacked tensor:", stacked_tensor.shape)         # Should be torch.Size([4




print("PYTHONPATH:", sys.path)


sys.argv = [
    "embedairr.py",  # Script name
    "--model_name", "ab2",
    "--fasta_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_5k_antiberta2.fa",
    # "--fasta_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_antiberta2.fa",
    "--output_path", "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/",
    "--cdr3_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_cdr3.csv",
    "--extract_cdr3_attention_matrices", "average_layer",
    "--extract_cdr3_embeddings", "unpooled",
    "--layers", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ",
    # "--extract_embeddings", "unpooled",
    # "--extract_attention_matrices", "average_layer"
]


sys.argv = [
    "embedairr.py",  # Script name
    "--model_name", "esm2",
    "--fasta_path","/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_esm2.fa",
    # "--fasta_path","/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_5k_esm2.fa",
    "--output_path", "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/",
    "--layers", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33",
    # "--extract_embeddings", "unpooled",
    "--extract_attention_matrices", "average_layer"
]




 
sys.argv = [
    "embedairr.py",  # Script name
    "--model_name", "esm2",
    # "--fasta_path","/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_esm2.fa",
    "--fasta_path","/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_heavy_chain_100k.fa",
    "--output_path", "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/",
    "--cdr3_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_cdr3.csv",
    "--extract_cdr3_attention_matrices", "average_layer",
    "--extract_cdr3_embeddings", "unpooled",
    "--layers", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33",
    "--extract_embeddings", "false",
    # "--extract_attention_matrices", "average_layer"
]
    



# Parsing command-line arguments for input and output file paths
def parse_arguments():
    # """Parse command-line arguments for input and output file paths."""
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
    # arguemnt for batch_size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for loading sequences. Default is 1024.",
    )
    # TODO add experiment name
    args = parser.parse_args()
    return args

# gino=1
# import torch
# x1 = torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/ab2/attention_matrices_average_layers/tz_paired_chain_5k_antiberta2_ab2_attention_matrices_average_layers_layer_1.pt")
# x1 = torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/esm2/attention_matrices_average_layers/tz_paired_chain_5k_esm2_esm2_attention_matrices_average_layers_layer_1.pt")
# x1 = torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/esm2/attention_matrices_average_layers/tz_paired_chain_5k_esm2_esm2_attention_matrices_average_layers_layer_1.pt")


# x1 = torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/esm2/embeddings_unpooled/tz_paired_chain_5k_esm2_esm2_embeddings_unpooled_layer_1.pt")
# x1 = torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/ab2/embeddings_unpooled/tz_paired_chain_5k_antiberta2_ab2_embeddings_unpooled_layer_1.pt")
# x1[0].shape

# torch.stack(x1[0:3],dim=0).shape



# torch.flatten(x1, start_dim =1).shape
# x2 = x1.view(x1.size(0), -1).shape

# x2 = x1.reshape(x1.size(0), -1)

# torch.save(x2.to(torch.float16), "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/esm2/attention_matrices_average_layers/tz_paired_chain_5k_esm2_esm2_attention_matrices_average_layers_layer_1_FLAT.pt")

# Check the data type (precision) of the tensor

# x3= torch.load  ("/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/esm2/attention_matrices_average_layers/tz_paired_chain_5k_esm2_esm2_attention_matrices_average_layers_layer_1_FLAT.pt")
# torch.save(x2, "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis3/ab2/attention_matrices_average_layers/tz_paired_chain_5k_antiberta2_ab2_attention_matrices_average_layers_layer_1_STACK.pt")


# if __name__ == "__main__":
    # Parse and store arguments
#re import select_model





args = parse_arguments()

# Check if output directory exists and creates it if it's missing
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

embedder = select_model(args.model_name)

embedder = embedder(args)

print("Embedder initialized")

embedder.run()

print("All outputs saved.")

if embedder.flatten and "unpooled" in embedder.output_types[1]:
  print('ciao')



sys.argv = [
    "embedairr.py",  # Script name
    "--model_name", "ab2",
    # "--fasta_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_5k_antiberta2.fa",
    # "--fasta_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_paired_chain_100k_antiberta2.fa",
    "--fasta_path","/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_heavy_chain_100k.fa",
    "--output_path", "/doctorai/userdata/airr_atlas/data/embeddings/levels_analysis2/",
    "--cdr3_path", "/doctorai/userdata/airr_atlas/data/sequences/bcr/trastuzumab/tz_cdr3.csv",
    "--extract_cdr3_attention_matrices", "average_layer",
    "--extract_cdr3_embeddings", "unpooled",
    "--layers", "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ",
    # "--extract_embeddings", "unpooled",
    # "--extract_attention_matrices", "average_layer"
]
    

del embedder
args = parse_arguments()

# Check if output directory exists and creates it if it's missing
if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)

embedder = select_model(args.model_name)

embedder = embedder(args)

print("Embedder initialized")

embedder.run()

print("All outputs saved.")



# import torch

# torch.stack(embedder.cdr3_context_extracted_unpooled[1][0:4],dim=0).flatten(start_dim=1)

