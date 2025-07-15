"""
DataProcessor class for handling data loading and preprocessing in PEPE embedders.

This module provides a centralized interface for managing data loading, sequence
preprocessing, tokenization, and batch generation for embedding extraction.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import torch
import pepe.utils

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading and preprocessing for embedding extraction.
    
    This class provides a unified interface for FASTA loading, sequence validation,
    tokenization, and batch generation across different embedder implementations.
    """
    
    def __init__(
        self,
        fasta_path: str,
        batch_size: int,
        max_length: int,
        substring_dict: Optional[Dict[str, str]] = None,
        context: int = 0,
        device: torch.device = torch.device("cpu"),
        disable_special_tokens: bool = False,
    ):
        """
        Initialize DataProcessor with configuration parameters.
        
        Args:
            fasta_path: Path to FASTA file containing sequences
            batch_size: Batch size for data loading
            max_length: Maximum sequence length
            substring_dict: Optional dictionary mapping sequence IDs to substrings
            context: Context length for substring extraction
            device: Device to use for tensor operations
            disable_special_tokens: Whether to disable special tokens
        """
        self.fasta_path = fasta_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.substring_dict = substring_dict
        self.context = context
        self.device = device
        self.disable_special_tokens = disable_special_tokens
        
        # Initialize data attributes
        self.sequences: Optional[Dict[str, str]] = None
        self.num_sequences: int = 0
        self.data_loader: Optional[torch.utils.data.DataLoader] = None
        self.actual_max_length: int = max_length
        
        # Load sequences from FASTA file
        self._load_sequences()
        
        logger.info(f"DataProcessor initialized with {self.num_sequences} sequences")
    
    def _load_sequences(self) -> None:
        """Load sequences from FASTA file."""
        logger.info(f"Loading sequences from {self.fasta_path}")
        try:
            self.sequences = pepe.utils.fasta_to_dict(self.fasta_path)
            self.num_sequences = len(self.sequences)
            logger.info(f"Loaded {self.num_sequences} sequences")
        except Exception as e:
            logger.error(f"Error loading sequences from {self.fasta_path}: {e}")
            raise
    
    def validate_sequences(self, valid_tokens: set, model_name: str) -> None:
        """
        Validate that sequences contain only valid tokens.
        
        Args:
            valid_tokens: Set of valid tokens for the model
            model_name: Name of the model (for error reporting)
        """
        if self.sequences is None:
            raise ValueError("Sequences not loaded. Call _load_sequences() first.")
        
        logger.info("Validating sequence tokens...")
        try:
            pepe.utils.check_input_tokens(valid_tokens, self.sequences, model_name)
            logger.info("All sequences contain valid tokens")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise
    
    def create_dataset(
        self,
        dataset_class: type,
        tokenizer_or_alphabet: Any,
        **kwargs
    ) -> Any:
        """
        Create dataset using the specified dataset class.
        
        Args:
            dataset_class: Dataset class to instantiate
            tokenizer_or_alphabet: Tokenizer or alphabet for sequence encoding
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            Dataset instance
        """
        if self.sequences is None:
            raise ValueError("Sequences not loaded. Call _load_sequences() first.")
        
        logger.info("Creating dataset...")
        
        # Common dataset parameters
        dataset_kwargs = {
            "sequences": self.sequences,
            "substring_dict": self.substring_dict,
            "context": self.context,
            "max_length": self.max_length,
            **kwargs
        }
        
        # Add tokenizer/alphabet with appropriate parameter name
        if hasattr(dataset_class, "__init__"):
            import inspect
            sig = inspect.signature(dataset_class.__init__)
            if "tokenizer" in sig.parameters:
                dataset_kwargs["tokenizer"] = tokenizer_or_alphabet
            elif "alphabet" in sig.parameters:
                dataset_kwargs["alphabet"] = tokenizer_or_alphabet
            else:
                # Try to determine from class name or other heuristics
                class_name = dataset_class.__name__.lower()
                if "esm" in class_name:
                    dataset_kwargs["alphabet"] = tokenizer_or_alphabet
                else:
                    dataset_kwargs["tokenizer"] = tokenizer_or_alphabet
        
        # Special handling for special tokens
        if not self.disable_special_tokens:
            if "add_special_tokens" in inspect.signature(dataset_class.__init__).parameters:
                dataset_kwargs["add_special_tokens"] = True
            elif "prepend_bos" in inspect.signature(dataset_class.__init__).parameters:
                dataset_kwargs["prepend_bos"] = True
                dataset_kwargs["append_eos"] = True
        
        try:
            dataset = dataset_class(**dataset_kwargs)
            logger.info("Dataset created successfully")
            return dataset
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise
    
    def create_data_loader(
        self,
        dataset: Any,
        use_token_budget: bool = True,
        collate_fn: Optional[callable] = None,
    ) -> Tuple[torch.utils.data.DataLoader, int]:
        """
        Create data loader with appropriate batching strategy.
        
        Args:
            dataset: Dataset to create loader for
            use_token_budget: Whether to use token budget batching
            collate_fn: Optional custom collate function
            
        Returns:
            Tuple of (data_loader, actual_max_length)
        """
        logger.info("Creating data loader...")
        
        try:
            if use_token_budget:
                # Use token budget batching for efficient memory usage
                logger.info("Using token budget batching...")
                batch_sampler = pepe.utils.TokenBudgetBatchSampler(
                    dataset=dataset, 
                    token_budget=self.batch_size
                )
                
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn or dataset.safe_collate
                )
            else:
                # Use regular batching
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn
                )
            
            # Get actual max length from dataset
            actual_max_length = self.max_length
            if hasattr(dataset, "get_max_encoded_length"):
                actual_max_length = dataset.get_max_encoded_length()
            
            self.data_loader = data_loader
            self.actual_max_length = actual_max_length
            
            logger.info(f"Data loader created with max length: {actual_max_length}")
            return data_loader, actual_max_length
            
        except Exception as e:
            logger.error(f"Error creating data loader: {e}")
            raise
    
    def get_special_tokens(
        self,
        tokenizer_or_alphabet: Any,
        embedder_type: str = "auto"
    ) -> torch.Tensor:
        """
        Extract special tokens from tokenizer or alphabet.
        
        Args:
            tokenizer_or_alphabet: Tokenizer or alphabet object
            embedder_type: Type of embedder ("esm", "huggingface", "custom", "auto")
            
        Returns:
            Tensor of special token IDs
        """
        logger.info("Extracting special tokens...")
        
        try:
            # Auto-detect embedder type if not specified
            if embedder_type == "auto":
                if hasattr(tokenizer_or_alphabet, "all_special_tokens"):
                    embedder_type = "esm"
                elif hasattr(tokenizer_or_alphabet, "all_special_ids"):
                    embedder_type = "huggingface"
                else:
                    embedder_type = "custom"
            
            # Extract special tokens based on embedder type
            if embedder_type == "esm":
                special_tokens = tokenizer_or_alphabet.all_special_tokens
                special_token_ids = torch.tensor(
                    [tokenizer_or_alphabet.tok_to_idx[tok] for tok in special_tokens],
                    device=self.device,
                    dtype=torch.int8,
                )
            elif embedder_type == "huggingface":
                special_token_ids = torch.tensor(
                    tokenizer_or_alphabet.all_special_ids,
                    device=self.device,
                    dtype=torch.int8,
                )
            elif embedder_type == "custom":
                if hasattr(tokenizer_or_alphabet, "all_special_ids"):
                    special_token_ids = torch.tensor(
                        tokenizer_or_alphabet.all_special_ids,
                        device=self.device,
                        dtype=torch.int8,
                    )
                else:
                    # Default special tokens (pad, cls, sep, unk)
                    special_token_ids = torch.tensor(
                        [0, 1, 2, 3], 
                        device=self.device, 
                        dtype=torch.int8
                    )
            else:
                raise ValueError(f"Unsupported embedder type: {embedder_type}")
            
            logger.info(f"Extracted {len(special_token_ids)} special tokens")
            return special_token_ids
            
        except Exception as e:
            logger.error(f"Error extracting special tokens: {e}")
            raise
    
    def get_valid_tokens(
        self,
        tokenizer_or_alphabet: Any,
        embedder_type: str = "auto"
    ) -> set:
        """
        Get valid tokens from tokenizer or alphabet.
        
        Args:
            tokenizer_or_alphabet: Tokenizer or alphabet object
            embedder_type: Type of embedder ("esm", "huggingface", "custom", "auto")
            
        Returns:
            Set of valid tokens
        """
        logger.info("Extracting valid tokens...")
        
        try:
            # Auto-detect embedder type if not specified
            if embedder_type == "auto":
                if hasattr(tokenizer_or_alphabet, "all_toks"):
                    embedder_type = "esm"
                elif hasattr(tokenizer_or_alphabet, "get_vocab"):
                    embedder_type = "huggingface"
                else:
                    embedder_type = "custom"
            
            # Extract valid tokens based on embedder type
            if embedder_type == "esm":
                valid_tokens = set(tokenizer_or_alphabet.all_toks)
            elif embedder_type == "huggingface":
                valid_tokens = set(tokenizer_or_alphabet.get_vocab().keys())
            elif embedder_type == "custom":
                if hasattr(tokenizer_or_alphabet, "get_vocab"):
                    valid_tokens = set(tokenizer_or_alphabet.get_vocab().keys())
                elif hasattr(tokenizer_or_alphabet, "all_toks"):
                    valid_tokens = set(tokenizer_or_alphabet.all_toks)
                else:
                    # Default amino acid tokens
                    valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
            else:
                raise ValueError(f"Unsupported embedder type: {embedder_type}")
            
            logger.info(f"Extracted {len(valid_tokens)} valid tokens")
            return valid_tokens
            
        except Exception as e:
            logger.error(f"Error extracting valid tokens: {e}")
            raise
    
    def get_substring_positions(
        self,
        label: str,
        special_tokens: bool = True,
        context: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Get the start and end positions of a substring in the full sequence.
        
        Args:
            label: Sequence label
            special_tokens: Whether special tokens are present
            context: Context length (uses self.context if None)
            
        Returns:
            Tuple of (start_position, end_position)
        """
        if self.sequences is None:
            raise ValueError("Sequences not loaded")
        
        if self.substring_dict is None:
            raise ValueError("Substring dictionary not provided")
        
        if context is None:
            context = self.context
        
        try:
            full_sequence = self.sequences[label]
            substring = self.substring_dict[label]
            
            # Remove gaps from substring
            substring = substring.replace("-", "")
            
            # Find substring position in sequence
            start = max(full_sequence.find(substring) - context, 0)
            if special_tokens:
                start += 1  # Account for special tokens
            
            end = min(start + len(substring) + context, len(full_sequence))
            if special_tokens:
                end += 1  # Account for special tokens
            
            return start, end
            
        except KeyError:
            raise KeyError(f"No matching substring found for {label}")
    
    def get_sequence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded sequences.
        
        Returns:
            Dictionary containing sequence statistics
        """
        if self.sequences is None:
            return {"num_sequences": 0, "sequences_loaded": False}
        
        sequence_lengths = [len(seq) for seq in self.sequences.values()]
        
        stats = {
            "num_sequences": self.num_sequences,
            "sequences_loaded": True,
            "min_length": min(sequence_lengths) if sequence_lengths else 0,
            "max_length": max(sequence_lengths) if sequence_lengths else 0,
            "avg_length": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
            "total_residues": sum(sequence_lengths),
            "has_substrings": self.substring_dict is not None,
            "num_substrings": len(self.substring_dict) if self.substring_dict else 0,
            "batch_size": self.batch_size,
            "max_length_config": self.max_length,
            "actual_max_length": self.actual_max_length,
        }
        
        return stats
    
    def log_statistics(self) -> None:
        """Log sequence statistics for debugging."""
        stats = self.get_sequence_statistics()
        logger.info("DataProcessor statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
    
    def __repr__(self) -> str:
        """String representation of DataProcessor."""
        return f"DataProcessor(num_sequences={self.num_sequences}, batch_size={self.batch_size}, max_length={self.max_length})"


class DataProcessorFactory:
    """
    Factory class for creating DataProcessor instances with embedder-specific configurations.
    """
    
    @staticmethod
    def create_esm_processor(
        fasta_path: str,
        batch_size: int,
        max_length: int,
        alphabet: Any,
        substring_dict: Optional[Dict[str, str]] = None,
        context: int = 0,
        device: torch.device = torch.device("cpu"),
        disable_special_tokens: bool = False,
        prepend_bos: bool = True,
        append_eos: bool = True,
    ) -> Tuple[torch.utils.data.DataLoader, int, torch.Tensor, set]:
        """
        Create ESM-specific data processing pipeline.
        
        Returns:
            Tuple of (data_loader, max_length, special_tokens, valid_tokens)
        """
        processor = DataProcessor(
            fasta_path=fasta_path,
            batch_size=batch_size,
            max_length=max_length,
            substring_dict=substring_dict,
            context=context,
            device=device,
            disable_special_tokens=disable_special_tokens,
        )
        
        # Get valid tokens and validate sequences
        valid_tokens = processor.get_valid_tokens(alphabet, "esm")
        processor.validate_sequences(valid_tokens, "ESM")
        
        # Create ESM dataset
        dataset = processor.create_dataset(
            dataset_class=pepe.utils.ESMDataset,
            tokenizer_or_alphabet=alphabet,
            prepend_bos=prepend_bos,
            append_eos=append_eos,
        )
        
        # Create data loader
        data_loader, actual_max_length = processor.create_data_loader(dataset)
        
        # Get special tokens
        special_tokens = processor.get_special_tokens(alphabet, "esm")
        
        return data_loader, actual_max_length, special_tokens, valid_tokens
    
    @staticmethod
    def create_huggingface_processor(
        fasta_path: str,
        batch_size: int,
        max_length: int,
        tokenizer: Any,
        substring_dict: Optional[Dict[str, str]] = None,
        context: int = 0,
        device: torch.device = torch.device("cpu"),
        disable_special_tokens: bool = False,
    ) -> Tuple[torch.utils.data.DataLoader, int, torch.Tensor, set]:
        """
        Create HuggingFace-specific data processing pipeline.
        
        Returns:
            Tuple of (data_loader, max_length, special_tokens, valid_tokens)
        """
        processor = DataProcessor(
            fasta_path=fasta_path,
            batch_size=batch_size,
            max_length=max_length,
            substring_dict=substring_dict,
            context=context,
            device=device,
            disable_special_tokens=disable_special_tokens,
        )
        
        # Get valid tokens and validate sequences
        valid_tokens = processor.get_valid_tokens(tokenizer, "huggingface")
        processor.validate_sequences(valid_tokens, "HuggingFace")
        
        # Create HuggingFace dataset
        dataset = processor.create_dataset(
            dataset_class=pepe.utils.HuggingFaceDataset,
            tokenizer_or_alphabet=tokenizer,
            add_special_tokens=not disable_special_tokens,
        )
        
        # Create data loader
        data_loader, actual_max_length = processor.create_data_loader(dataset)
        
        # Get special tokens
        special_tokens = processor.get_special_tokens(tokenizer, "huggingface")
        
        return data_loader, actual_max_length, special_tokens, valid_tokens
    
    @staticmethod
    def create_custom_processor(
        fasta_path: str,
        batch_size: int,
        max_length: int,
        tokenizer: Any,
        dataset_class: type,
        substring_dict: Optional[Dict[str, str]] = None,
        context: int = 0,
        device: torch.device = torch.device("cpu"),
        disable_special_tokens: bool = False,
        **dataset_kwargs
    ) -> Tuple[torch.utils.data.DataLoader, int, torch.Tensor, set]:
        """
        Create custom data processing pipeline.
        
        Returns:
            Tuple of (data_loader, max_length, special_tokens, valid_tokens)
        """
        processor = DataProcessor(
            fasta_path=fasta_path,
            batch_size=batch_size,
            max_length=max_length,
            substring_dict=substring_dict,
            context=context,
            device=device,
            disable_special_tokens=disable_special_tokens,
        )
        
        # Get valid tokens and validate sequences
        valid_tokens = processor.get_valid_tokens(tokenizer, "custom")
        processor.validate_sequences(valid_tokens, "Custom")
        
        # Create custom dataset
        dataset = processor.create_dataset(
            dataset_class=dataset_class,
            tokenizer_or_alphabet=tokenizer,
            **dataset_kwargs
        )
        
        # Create data loader
        data_loader, actual_max_length = processor.create_data_loader(dataset)
        
        # Get special tokens
        special_tokens = processor.get_special_tokens(tokenizer, "custom")
        
        return data_loader, actual_max_length, special_tokens, valid_tokens