import os
import torch
import embedairr.utils
from embedairr.embedders.base_embedder import BaseEmbedder
from transformers import T5EncoderModel, T5Tokenizer
from transformers import RoFormerTokenizer, RoFormerModel

# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        if self.return_logits:
            print("Warning: Logits are not supported for this model. Setting to False.")
            self.return_logits = False
            self.output_types.remove("logits")

    def load_layers(self, layers):
        """Check if the specified representation layers are valid."""
        if not layers:
            layers = list(range(1, self.model.config.num_hidden_layers + 1))
            return layers
        assert all(
            -(self.model.config.num_hidden_layers + 1)
            <= i
            <= self.model.config.num_hidden_layers
            for i in layers
        )
        layers = [
            (i + self.model.config.num_hidden_layers + 1)
            % (self.model.config.num_hidden_layers + 1)
            for i in layers
        ]
        return layers

    def load_data(self, sequences, cdr3_dict):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        print("Tokenizing sequences...")
        dataset = embedairr.utils.HuggingFaceDataset(
            sequences,
            cdr3_dict,
            self.context,
            self.tokenizer,
            self.max_length,
            add_special_tokens=not self.disable_special_tokens,
        )
        print("Batching sequences...")
        batch_sampler = embedairr.utils.TokenBudgetBatchSampler(
            dataset=dataset, token_budget=self.batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=dataset.safe_collate
        )
        max_length = dataset.get_max_encoded_length()
        print("Finished tokenizing and batching sequences")

        return data_loader, max_length

    def compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits=False,
    ):
        outputs = model(
            input_ids=toks,
            attention_mask=attention_mask,
            output_hidden_states=return_embeddings,
            output_attentions=return_contacts,
        )
        if return_contacts:
            attention_matrices = torch.stack(outputs.attentions).to(
                dtype=torch.float16
            )  # stack attention matrices across layers
            torch.cuda.empty_cache()
        else:
            attention_matrices = None
        if return_embeddings:
            representations = {
                layer: outputs.hidden_states[layer].to(
                    device="cpu", dtype=torch.float16
                )
                for layer in self.layers
            }
            torch.cuda.empty_cache()
        else:
            representations = None
        logits = None  # Model doesn't return logits
        return logits, representations, attention_matrices


class Antiberta2Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = embedairr.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        embedairr.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader, self.max_length = self.load_data(
            self.sequences, self.cdr3_dict
        )
        self.set_output_objects()

    def initialize_model(self, model_link="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")
        tokenizer = RoFormerTokenizer.from_pretrained(model_link, use_fast=True)
        model = RoFormerModel.from_pretrained(model_link).to(device)
        model.eval()
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = self.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        embedairr.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader, self.max_length = self.load_data(
            self.sequences, self.cdr3_dict
        )
        self.set_output_objects()

    def get_valid_tokens(self):
        valid_tokens = set(
            k[1:] if k.startswith("▁") else k
            for k in set(self.tokenizer.get_vocab().keys())
        )
        return valid_tokens

    def initialize_model(self, model_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        """Initialize the model, tokenizer, and device."""

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")
        tokenizer = T5Tokenizer.from_pretrained(model_link, use_fast=True)
        model = T5EncoderModel.from_pretrained(model_link).to(device)
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size
