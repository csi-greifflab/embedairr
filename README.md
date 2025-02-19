# EmbedAIRR

EmbedAIRR is a tool for extracting embeddings and attention matrices from protein sequences using pre-trained models. This tool supports various configurations for extracting embeddings and attention matrices, including options for handling CDR3 sequences. Currently implemented models are ESM2 and AntiBERTa2-CSSP.

## Usage

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the embedding script:
    ```sh
    python embed_airr.py --input <input-file> --output <output-file>
    ```

## Arguments
- --model_name (str, required): Model name. Example: esm2_t33_650M_UR50D.
- --fasta_path (str, required): Path to the FASTA file.
- --output_path (str, required): Directory for output files. Will generate a subdirectory for outputs of each output type.
- --cdr3_path (str, optional): Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.
- --context (int, optional): Number of amino acids to include before and after the CDR3 sequence. Default is 0.
- --layers (str, optional): Representation layers to extract from the model. Default is the last layer. Example: --layers -1 6.
- --extract_embeddings (str, optional): Set the embedding return types. Choose one or more from: pooled, unpooled, false. Default is pooled.
- --extract_cdr3_embeddings (str, optional): Set the CDR3 embedding return types. Choose one or more from: pooled, unpooled, false. Requires --cdr3_path to be set. Default is pooled.
- --extract_attention_matrices (str, optional): Set the attention matrix return types. Choose one or more from: false, all_heads, average_layer, average_all. Default is false.
- --extract_cdr3_attention_matrices (str, optional): Set the CDR3 attention matrix return types. Choose one or more from: false, all_heads, average_layer, average_all. Requires --cdr3_path to be set. Default is false.
- --batch_size (int, optional): Batch size for loading sequences. Default is 1024.
- --discard_padding (bool, optional): Discard padding tokens from unpooled embeddings output. Default is False.
- --max_length (int, optional): Length to which sequences will be padded. Default is 140.
