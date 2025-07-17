import logging
import os
import torch
import pepe.utils
from pepe.embedders.base_embedder import BaseEmbedder


# Lazy imports to avoid loading heavy dependencies at import time
def _import_transformers():
    """Lazy import of transformers components to avoid loading issues."""
    try:
        from transformers import T5EncoderModel, T5Tokenizer
        from transformers import RoFormerTokenizer, RoFormerModel
        from transformers.models.roformer.modeling_roformer import (
            RoFormerSinusoidalPositionalEmbedding,
        )
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        return (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoConfig,
        )
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
        raise ImportError(
            "Failed to import transformers. Please ensure transformers is installed: pip install transformers"
        ) from e


# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logger = logging.getLogger("src.embedders.huggingface_embedder")


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        if self.return_logits:
            logger.warning(
                "Warning: Logits are not supported for this model. Setting to False."
            )
            self.return_logits = False
            self.output_types.remove("logits")

    def _load_layers(self, layers):
        """Check if the specified representation layers are valid."""
        if not layers:
            layers = list(range(1, self.model.config.num_hidden_layers + 1))  # type: ignore
            return layers
        assert all(
            -(self.model.config.num_hidden_layers + 1)  # type: ignore
            <= i
            <= self.model.config.num_hidden_layers  # type: ignore
            for i in layers
        )
        layers = [
            (i + self.model.config.num_hidden_layers + 1)  # type: ignore
            % (self.model.config.num_hidden_layers + 1)  # type: ignore
            for i in layers
        ]
        return layers

    def _load_data(self, sequences, substring_dict):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        dataset = pepe.utils.HuggingFaceDataset(
            sequences,
            substring_dict,
            self.context,
            self.tokenizer,  # type: ignore
            self.max_length,
            add_special_tokens=not self.disable_special_tokens,
        )
        logger.info("Batching sequences...")
        batch_sampler = pepe.utils.TokenBudgetBatchSampler(
            dataset=dataset, token_budget=self.batch_size
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=dataset.safe_collate
        )
        max_length = dataset.get_max_encoded_length()
        logger.info("Finished tokenizing and batching sequences")

        return data_loader, max_length

    def _compute_outputs(
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
            attention_matrices = (
                torch.stack(outputs.attentions)  # type: ignore
                .to(self._precision_to_dtype(self.precision, "torch"))  # type: ignore
                .cpu()
            )  # stack attention matrices across layers
            torch.cuda.empty_cache()
        else:
            attention_matrices = None
        if return_embeddings:
            representations = {
                layer: outputs.hidden_states[layer]
                .to(
                    self._precision_to_dtype(self.precision, "torch"),
                )
                .cpu()
                for layer in self.layers  # type: ignore
            }
            torch.cuda.empty_cache()
        else:
            representations = None
        logits = None  # Model doesn't return logits
        return logits, representations, attention_matrices


class Antiberta2Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        pepe.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_length = self._load_data(
            self.sequences, self.substring_dict
        )
        self._set_output_objects()
        assert self.max_length <= 256, "AntiBERTa2 only supports max_length <= 256"

    def _initialize_model(self, model_link="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        if torch.cuda.is_available() and self.device == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
        ) = _import_transformers()

        tokenizer = RoFormerTokenizer.from_pretrained(model_link, use_fast=True)
        model = RoFormerModel.from_pretrained(model_link).to(device)  # type: ignore
        model.eval()
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        pepe.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_length = self._load_data(
            self.sequences, self.substring_dict
        )
        self._set_output_objects()

    def get_valid_tokens(self):
        valid_tokens = set(
            k[1:] if k.startswith("▁") else k
            for k in set(self.tokenizer.get_vocab().keys())
        )
        return valid_tokens

    def _initialize_model(self, model_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        """Initialize the model, tokenizer, and device."""

        if torch.cuda.is_available() and self.device == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
        ) = _import_transformers()

        tokenizer = T5Tokenizer.from_pretrained(model_link, use_fast=True)
        model = T5EncoderModel.from_pretrained(model_link).to(device)  # type: ignore
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class GenericHuggingFaceEmbedder(HuggingfaceEmbedder):
    """Generic HuggingFace embedder that can handle any model using AutoModel and AutoTokenizer."""
    
    def __init__(self, args):
        super().__init__(args)
        self.sequences = pepe.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self._initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        pepe.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self._load_layers(self.layers)
        self.data_loader, self.max_length = self._load_data(
            self.sequences, self.substring_dict
        )
        self._set_output_objects()

    def get_valid_tokens(self):
        """Get valid tokens from the tokenizer vocabulary."""
        vocab = self.tokenizer.get_vocab()
        valid_tokens = set()
        
        for token in vocab.keys():
            # Handle different tokenizer formats
            if token.startswith("▁"):  # SentencePiece tokenizer
                valid_tokens.add(token[1:])
            elif token.startswith("##"):  # BERT-style subword tokenizer
                valid_tokens.add(token[2:])
            else:
                valid_tokens.add(token)
        
        return valid_tokens

    def _initialize_model(self, model_link):
        """Initialize the model, tokenizer, and device using AutoModel and AutoTokenizer."""
        
        if torch.cuda.is_available() and self.device == "cuda":
            device = torch.device("cuda")
            logger.info("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")

        # Lazy import transformers components
        (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoConfig,
        ) = _import_transformers()

        try:
            # Load config first to get model information
            config = AutoConfig.from_pretrained(model_link)
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_link, use_fast=True)
            model = AutoModel.from_pretrained(model_link).to(device)
            model.eval()
            
            # Extract model configuration
            num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_heads', 12))
            num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', 12))
            embedding_size = getattr(config, 'hidden_size', 768)
            
            logger.info(f"Loaded model {model_link} with {num_layers} layers, {num_heads} heads, and {embedding_size} embedding size")
            
            return model, tokenizer, num_heads, num_layers, embedding_size
            
        except Exception as e:
            logger.error(f"Failed to load model {model_link}: {e}")
            raise ValueError(f"Could not load model {model_link}. Error: {e}")

    def _compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits=False,
    ):
        """Compute model outputs with generic handling for different model types."""
        try:
            outputs = model(
                input_ids=toks,
                attention_mask=attention_mask,
                output_hidden_states=return_embeddings,
                output_attentions=return_contacts,
            )
        except Exception as e:
            logger.warning(f"Failed to get outputs with attention/hidden states: {e}")
            # Fallback to basic forward pass
            outputs = model(input_ids=toks, attention_mask=attention_mask)
        
        attention_matrices = None
        representations = None
        
        if return_contacts and hasattr(outputs, 'attentions') and outputs.attentions:
            try:
                attention_matrices = (
                    torch.stack(outputs.attentions)
                    .to(self._precision_to_dtype(self.precision, "torch"))
                    .cpu()
                )
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to extract attention matrices: {e}")
        
        if return_embeddings and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            try:
                representations = {
                    layer: outputs.hidden_states[layer]
                    .to(self._precision_to_dtype(self.precision, "torch"))
                    .cpu()
                    for layer in self.layers
                }
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to extract hidden states: {e}")
        
        logits = None  # Generic embedder doesn't return logits
        return logits, representations, attention_matrices
