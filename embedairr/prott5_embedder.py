import torch
from transformers import T5EncoderModel, T5Tokenizer
from embedairr.huggingface_embedder import HuggingfaceEmbedder


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences, _ = self.fasta_to_dict(args.fasta_path, gaps=False)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        self.check_input_tokens(self.valid_tokens, self.sequences, gaps=False)
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data(self.sequences)
        self.set_output_objects()

    def get_valid_tokens(self):
        valid_tokens = set(
            k[1:] if k.startswith("▁") else k
            for k in set(self.tokenizer.get_vocab().keys())
        )
        return valid_tokens

    def initialize_model(self, model_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        """Initialize the model, tokenizer, and device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device type: {device}")
        tokenizer = T5Tokenizer.from_pretrained(model_link)
        model = T5EncoderModel.from_pretrained(model_link).to(device)
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size
