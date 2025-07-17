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
        from transformers import AutoModelForCausalLM

        return (
            T5EncoderModel,
            T5Tokenizer,
            RoFormerTokenizer,
            RoFormerModel,
            RoFormerSinusoidalPositionalEmbedding,
            AutoModel,
            AutoTokenizer,
            AutoConfig,
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


class HuggingFaceEmbedder(BaseEmbedder):
    """Generic HuggingFace embedder that can handle arbitrary models with fallback logic."""
    
    def __init__(self, model_name, **kwargs):
        # Initialize with model name instead of args for simpler interface
        self.model_name = model_name
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = kwargs.get('precision', 'float32')
        self.batch_size = kwargs.get('batch_size', 1024)
        self.max_length = kwargs.get('max_length', 512)
        self.layers = kwargs.get('layers', None)
        self.disable_special_tokens = kwargs.get('disable_special_tokens', False)
        
        # Initialize model and tokenizer
        self.model, self.tokenizer, self.config = self._initialize_model(model_name)
        self.num_heads = getattr(self.config, 'num_attention_heads', getattr(self.config, 'num_heads', 12))
        self.num_layers = getattr(self.config, 'num_hidden_layers', getattr(self.config, 'num_layers', 12))
        self.embedding_size = getattr(self.config, 'hidden_size', 768)
        
        # Set up layers
        self.layers = self._load_layers(self.layers)
        
        # Set up special tokens
        if hasattr(self.tokenizer, 'all_special_ids'):
            self.special_tokens = torch.tensor(
                self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
            )
        else:
            self.special_tokens = torch.tensor([], device=self.device, dtype=torch.int8)
    
    def _initialize_model(self, model_name):
        """Initialize the model and tokenizer with fallback logic."""
        try:
            # Import transformers components
            (
                T5EncoderModel,
                T5Tokenizer,
                RoFormerTokenizer,
                RoFormerModel,
                RoFormerSinusoidalPositionalEmbedding,
                AutoModel,
                AutoTokenizer,
                AutoConfig,
                AutoModelForCausalLM,
            ) = _import_transformers()
            
            # Set up device
            if torch.cuda.is_available() and self.device == "cuda":
                device = torch.device("cuda")
                logger.info("Using GPU")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU")
            
            # Try to load with trust_remote_code=True for models that require it
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                # Try different model classes based on the model type
                model_type = getattr(config, 'model_type', None)
                if model_type == 'progen' or 'progen' in model_name.lower():
                    # For ProGen models, use AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
                else:
                    # For other models, try AutoModel first
                    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
                
                logger.info(f"Successfully loaded model {model_name} with trust_remote_code=True")
            except Exception as e:
                logger.warning(f"Failed to load with trust_remote_code=True: {e}")
                # Fallback to loading without trust_remote_code
                try:
                    config = AutoConfig.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).to(device)
                    logger.info(f"Successfully loaded model {model_name} without trust_remote_code")
                except Exception as e2:
                    logger.error(f"Failed to load model {model_name}: {e2}")
                    raise e2
            
            model.eval()
            
            # Set padding token if not present
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    # Add a padding token if none exists
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {e}")
            raise e
    
    def _load_layers(self, layers):
        """Check if the specified representation layers are valid."""
        if not layers:
            layers = list(range(1, self.num_layers + 1))
            return layers
        assert all(
            -(self.num_layers + 1) <= i <= self.num_layers
            for i in layers
        )
        layers = [
            (i + self.num_layers + 1) % (self.num_layers + 1)
            for i in layers
        ]
        return layers
    
    def get_embeddings(self, sequences):
        """Get embeddings for a list of sequences."""
        if not isinstance(sequences, list):
            sequences = [sequences]
        
        embeddings = []
        
        for sequence in sequences:
            # Tokenize the sequence
            inputs = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=not self.disable_special_tokens
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get embeddings from the last layer by default
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use the last layer's hidden states
                    last_hidden_states = outputs.hidden_states[-1]
                    # Mean pooling over sequence length
                    attention_mask = inputs['attention_mask']
                    embeddings_seq = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                elif hasattr(outputs, 'last_hidden_state'):
                    # Fallback to last_hidden_state
                    attention_mask = inputs['attention_mask']
                    embeddings_seq = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    # Final fallback - try to get any tensor output
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings_seq = outputs.pooler_output
                    else:
                        # For causal LM models, we might need to access the hidden states differently
                        # Let's try to get the first available tensor
                        for attr_name in ['hidden_states', 'last_hidden_state']:
                            if hasattr(outputs, attr_name):
                                hidden_states = getattr(outputs, attr_name)
                                if hidden_states is not None:
                                    if isinstance(hidden_states, tuple):
                                        hidden_states = hidden_states[-1]  # Get last layer
                                    attention_mask = inputs['attention_mask']
                                    embeddings_seq = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
                                    break
                        else:
                            raise ValueError(f"Could not extract embeddings from model output: {type(outputs)}")
                
                embeddings.append(embeddings_seq.cpu().numpy())
        
        return torch.tensor(embeddings).squeeze()
    
    def _load_data(self, sequences, substring_dict):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        dataset = pepe.utils.HuggingFaceDataset(
            sequences,
            substring_dict,
            self.context,
            self.tokenizer,
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
                torch.stack(outputs.attentions)
                .to(self._precision_to_dtype(self.precision, "torch"))
                .cpu()
            )
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
                for layer in self.layers
            }
            torch.cuda.empty_cache()
        else:
            representations = None
        logits = None  # Model doesn't return logits by default
        return logits, representations, attention_matrices


class HuggingfaceEmbedder(BaseEmbedder):
    """Legacy class for backward compatibility"""
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
        ) = _import_transformers()

        tokenizer = T5Tokenizer.from_pretrained(model_link, use_fast=True)
        model = T5EncoderModel.from_pretrained(model_link).to(device)  # type: ignore
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size
