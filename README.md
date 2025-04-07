# EmbedAIRR

EmbedAIRR is a tool for extracting embeddings and attention matrices from protein sequences using pre-trained models. This tool supports various configurations for extracting embeddings and attention matrices, including options for handling CDR3 sequences. Currently implemented models are ESM2 from the 2023 paper ["Evolutionary-scale prediction of atomic-level protein structure with a language model"](https://science.org/doi/10.1126/science.ade2574) and AntiBERTa2-CSSP from the 2023 pre-print ["Enhancing Antibody Language Models with Structural Information"](https://www.mlsb.io/papers_2023/Enhancing_Antibody_Language_Models_with_Structural_Information.pdf).

## Usage

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

2. Install the required dependencies:
    Rust tools are required to build some dependencies:
    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
    
    ```sh
    cd embedairr
    pip install --upgrade pip
    pip install .
    ```

3. Run the embedding script:
    ```sh
    embedairr --experiment_name <optional_string> --fasta_path <file_path> --output_path <directory> --model_name <model_name> --<other_optional_arguments>
    ```

## List of supported models:
- ESM-family models
    - ESM1:
        - esm1_t34_670M_UR50S
        - esm1_t34_670M_UR50D
        - esm1_t34_670M_UR100
        - esm1_t12_85M_UR50S
        - esm1_t6_43M_UR50S
        - esm1b_t33_650M_UR50S
        - esm1v_t33_650M_UR90S_1
        - esm1v_t33_650M_UR90S_2
        - esm1v_t33_650M_UR90S_3
        - esm1v_t33_650M_UR90S_4
        - esm1v_t33_650M_UR90S_5
    - ESM2:
        - esm2_t6_8M_UR50D
        - esm2_t12_35M_UR50D
        - esm2_t30_150M_UR50D
        - esm2_t33_650M_UR50D
        - esm2_t36_3B_UR50D
        - esm2_t48_15B_UR50D
- Huggingface Transformer models
    - T5 transformer models (tested)
        - Rostlab/prot_t5_xl_half_uniref50-enc
        - Rostlab/ProstT5
    - RoFormer models (tested)
        - AntiBERTa2-CSSP
        - AntiBERTa2


## Arguments
- --experiment_name (str, optional): Prefix for names of output files. If not provided, name of input file will be used for prefix.
- --model_name (str, required): Name of model or link to model. Choose from 'List of supported models'. Example: esm2_t33_650M_UR50D.
- --fasta_path (str, required): "Path to the input FASTA file. If no experiment name is provided, the output files will be named after the input file.",
- --output_path (str, required): Directory for output files. Will generate a subdirectory for outputs of each output type.
- --cdr3_path (str, optional): Path to the CDR3 CSV file. Only required when calculating CDR3 sequence embeddings.
- --context (int, optional): Number of amino acids to include before and after the CDR3 sequence. Default is 0.
- --layers (str, optional): Representation layers to extract from the model. Default is the last layer. Example: --layers -1 6.
- --extract_logits (str, optional): If true, logits from selected layers will be exported to file. Default is false.
- --extract_embeddings (str, optional): Set the embedding return types. Choose one or more from: pooled, unpooled, false. Default is pooled.
- --extract_cdr3_embeddings (str, optional): Set the CDR3 embedding return types. Choose one or more from: pooled, unpooled, false. Requires --cdr3_path to be set. Default is pooled.
- --extract_attention_matrices (str, optional): Set the attention matrix return types. Choose one or more from: false, all_heads, average_layer, average_all. Default is false.
- --extract_cdr3_attention_matrices (str, optional): Set the CDR3 attention matrix return types. Choose one or more from: false, all_heads, average_layer, average_all. Requires --cdr3_path to be set. Default is false.
- --batch_size (int, optional): Batch size for loading sequences. Default is 1024.
- --discard_padding (bool, optional): Discard padding tokens from unpooled embeddings output. Default is False.
- --max_length (int, optional): Length to which sequences will be padded. Default is length of longest sequence in input file. If shorter than longest sequence, will forcefully default to length of longest sequence.
- --batch_writing (str, optional): When True, embedair preallocates the required disk space and writes each batch of outputs to .npy files. When False, all outputs are stored in RAM and written to disk at once after computation has finished and stored as .pt files. Default is True.
- --disable_special_tokens (str, optional): When True, embedair disables pre- and appending BOS/CLS and EOS/SEP tokens before embedding. Default is false.