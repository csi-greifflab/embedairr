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
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        from transformers.models.roformer.modeling_roformer import (
            RoFormerSinusoidalPositionalEmbedding,
        )

        return (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
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
            layers = list(range(1, self.num_layers + 1))  # type: ignore
            return layers
        assert all(
            -(self.num_layers + 1)  # type: ignore
            <= i
            <= self.num_layers  # type: ignore
            for i in layers
        )
        layers = [
            (i + self.num_layers + 1)  # type: ignore
            % (self.num_layers + 1)  # type: ignore
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
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
        ) = _import_transformers()

        # Try without trust_remote_code first, then with it if needed
        try:
            tokenizer = RoFormerTokenizer.from_pretrained(model_link, use_fast=True)
            model = RoFormerModel.from_pretrained(model_link).to(device)  # type: ignore
        except (OSError, ValueError) as e:
            # If the model requires custom code, try with trust_remote_code=True
            if "trust_remote_code" in str(e).lower() or "custom code" in str(e).lower():
                tokenizer = RoFormerTokenizer.from_pretrained(model_link, use_fast=True, trust_remote_code=True)
                model = RoFormerModel.from_pretrained(model_link, trust_remote_code=True).to(device)  # type: ignore
            else:
                raise
        model.eval()
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = self.fasta_to_dict(args.fasta_path)  # type: ignore
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
            AutoModel,
            AutoTokenizer,
            AutoModelForCausalLM,
        ) = _import_transformers()

        # Try without trust_remote_code first, then with it if needed
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_link, use_fast=True)
            model = T5EncoderModel.from_pretrained(model_link).to(device)  # type: ignore
        except (OSError, ValueError) as e:
            # If the model requires custom code, try with trust_remote_code=True
            if "trust_remote_code" in str(e).lower() or "custom code" in str(e).lower():
                tokenizer = T5Tokenizer.from_pretrained(model_link, use_fast=True, trust_remote_code=True)
                model = T5EncoderModel.from_pretrained(model_link, trust_remote_code=True).to(device)  # type: ignore
            else:
                raise
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class GenericHuggingFaceEmbedder(HuggingfaceEmbedder):
    """Generic HuggingFace embedder that can handle models with unknown architectures using AutoModel and AutoTokenizer."""
    
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
        self.valid_tokens = self._get_valid_tokens()
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

    def _get_valid_tokens(self):
        """Get valid tokens from the tokenizer."""
        if hasattr(self.tokenizer, 'get_vocab'):
            vocab = self.tokenizer.get_vocab()
            # Handle different tokenizer types
            if hasattr(self.tokenizer, 'decoder') and self.tokenizer.decoder:
                # T5-style tokenizer
                valid_tokens = set(
                    k[1:] if k.startswith("▁") else k for k in vocab.keys()
                )
            else:
                # Standard tokenizer
                valid_tokens = set(vocab.keys())
            return valid_tokens
        return set()

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
            AutoModelForCausalLM,
        ) = _import_transformers()

        # For models that commonly require custom code, use trust_remote_code=True immediately
        requires_custom_code = any(pattern in model_link.lower() for pattern in [
            'progen', 'protgpt', 'esm-fold', 'esmfold', 'alphafold'
        ])
        
        if requires_custom_code:
            tokenizer = AutoTokenizer.from_pretrained(model_link, use_fast=True, trust_remote_code=True)
            try:
                model = AutoModel.from_pretrained(model_link, trust_remote_code=True).to(device)
            except ValueError as model_error:
                # If AutoModel fails, try AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_link, trust_remote_code=True).to(device)
        else:
            # Try without trust_remote_code first, then with it if needed
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_link, use_fast=True)
                try:
                    model = AutoModel.from_pretrained(model_link).to(device)
                except ValueError as model_error:
                    # If AutoModel fails, try AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(model_link).to(device)
            except (OSError, ValueError) as e:
                # If the model requires custom code, try with trust_remote_code=True
                if "trust_remote_code" in str(e).lower() or "custom code" in str(e).lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_link, use_fast=True, trust_remote_code=True)
                    try:
                        model = AutoModel.from_pretrained(model_link, trust_remote_code=True).to(device)
                    except ValueError as model_error:
                        # If AutoModel fails, try AutoModelForCausalLM
                        model = AutoModelForCausalLM.from_pretrained(model_link, trust_remote_code=True).to(device)
                else:
                    raise

        # Handle tokenizer padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                # Add a special padding token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))

        model.eval()
        
        # Get model configuration
        config = model.config
        num_heads = getattr(config, 'num_attention_heads', getattr(config, 'num_heads', getattr(config, 'n_head', 12)))
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'num_layers', getattr(config, 'n_layer', 12)))
        embedding_size = getattr(config, 'hidden_size', getattr(config, 'd_model', getattr(config, 'embed_dim', 768)))
        
        logger.info(f"Loaded generic HuggingFace model: {model_link}")
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Number of heads: {num_heads}, layers: {num_layers}, embedding size: {embedding_size}")
        
        return model, tokenizer, num_heads, num_layers, embedding_size
