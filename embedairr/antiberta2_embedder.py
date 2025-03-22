import torch
from transformers import RoFormerTokenizer, RoFormerModel
import os
from embedairr.huggingface_embedder import HuggingfaceEmbedder

# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class Antiberta2Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences_gapped, self.sequences = self.fasta_to_dict(
            args.fasta_path, gaps=True
        )
        self.num_sequences = len(self.sequences_gapped)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        self.check_input_tokens(self.valid_tokens, self.sequences_gapped, gaps=True)
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data(self.sequences_gapped)
        self.sequences = {
            sequence_id: sequence_aa.replace(" ", "")
            for sequence_id, sequence_aa in self.sequences_gapped.items()
        }
        self.set_output_objects()

    def initialize_model(self, model_link="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device type: {device}")
        tokenizer = RoFormerTokenizer.from_pretrained(model_link)
        model = RoFormerModel.from_pretrained(model_link).to(device)
        model.eval()
        # disable eos and bos
        model.config
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size
