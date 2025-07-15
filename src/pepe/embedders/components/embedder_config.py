"""
EmbedderConfig class for managing embedder configuration and validation.

This module provides a centralized interface for managing all configuration
parameters, validation logic, and computed configuration values for embedders.
"""

import os
import csv
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class EmbedderConfig:
    """
    Manages all configuration parameters for embedding extraction.
    
    This class encapsulates configuration validation, parameter processing,
    and computed configuration values to provide a clean interface for embedders.
    """
    
    def __init__(self, args):
        """
        Initialize EmbedderConfig with argument parsing and validation.
        
        Args:
            args: Argument namespace containing all configuration parameters
        """
        # Process basic configuration parameters
        self.fasta_path = args.fasta_path
        self.model_link = args.model_name
        self.disable_special_tokens = args.disable_special_tokens
        self.substring_path = args.substring_path
        self.context = args.context
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.discard_padding = args.discard_padding
        self.flatten = args.flatten
        self.streaming_output = args.streaming_output
        self.precision = args.precision
        
        # Process model name and paths
        self.model_name = self._process_model_name(self.model_link)
        self.output_path = os.path.join(args.output_path, self.model_name)
        self.output_prefix = self._process_output_prefix(args.experiment_name, self.fasta_path)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        # Process layers configuration
        self.layers = self._process_layers(args.layers)
        
        # Process device configuration
        self.device = self._process_device(args.device)
        
        # Process output types and compute derived configuration
        self.output_types = self._get_output_types(args)
        self.return_embeddings, self.return_contacts, self.return_logits = self._compute_return_flags()
        
        # Process streaming configuration
        self.num_workers = args.num_workers if self.streaming_output else 1
        self.max_in_flight = self.num_workers * 2
        self.flush_batches_after = args.flush_batches_after * 1024**2  # in bytes
        
        # Set up checkpoint directory
        self.checkpoint_dir = self.output_path
        
        # Load substrings if specified
        self.substring_dict = self._load_substrings(self.substring_path)
        
        # Validate configuration
        self._validate_configuration()
    
    def _process_model_name(self, model_link: str) -> str:
        """
        Process model link to extract model name.
        
        Args:
            model_link: Model link or path
            
        Returns:
            Processed model name
        """
        if (
            model_link.endswith(".pt")
            or model_link.endswith(".pth")
            or model_link.startswith("custom:")
            or (
                os.path.exists(model_link)
                and (os.path.isfile(model_link) or os.path.isdir(model_link))
            )
        ):
            return Path(model_link).stem
        else:
            return re.sub(r"^.*?/", "", model_link)
    
    def _process_output_prefix(self, experiment_name: Optional[str], fasta_path: str) -> str:
        """
        Process output prefix from experiment name or FASTA path.
        
        Args:
            experiment_name: Optional experiment name
            fasta_path: Path to FASTA file
            
        Returns:
            Processed output prefix
        """
        if not experiment_name:
            return os.path.splitext(os.path.basename(fasta_path))[0]
        else:
            return experiment_name
    
    def _process_layers(self, layers: List[List[int]]) -> Optional[List[int]]:
        """
        Process layers configuration.
        
        Args:
            layers: List of layer specifications
            
        Returns:
            Processed layers list or None
        """
        if layers == [None]:
            return None
        return [j for i in layers for j in i]
    
    def _process_device(self, device: str) -> torch.device:
        """
        Process device configuration.
        
        Args:
            device: Device specification string
            
        Returns:
            Configured torch device
        """
        if torch.cuda.is_available() and device == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _get_output_types(self, args) -> List[str]:
        """
        Get output types from extraction embeddings configuration.
        
        Args:
            args: Argument namespace
            
        Returns:
            List of output types
        """
        output_types = []
        
        options_mapping = {
            "per_token": "per_token",
            "mean_pooled": "mean_pooled",
            "substring_pooled": "substring_pooled",
            "attention_head": "attention_head",
            "attention_layer": "attention_layer",
            "attention_model": "attention_model",
            "logits": "logits",
        }
        
        for option in args.extract_embeddings:
            if option in options_mapping:
                output_type = options_mapping[option]
                if output_type not in output_types:
                    output_types.append(output_type)
        
        return output_types
    
    def _compute_return_flags(self) -> Tuple[bool, bool, bool]:
        """
        Compute return flags based on output types.
        
        Returns:
            Tuple of (return_embeddings, return_contacts, return_logits)
        """
        return_embeddings = False
        return_contacts = False
        return_logits = False
        
        for output_type in self.output_types:
            if "pooled" in output_type or "per_token" in output_type:
                return_embeddings = True
            if "attention" in output_type:
                return_contacts = True
            if output_type == "logits":
                return_logits = True
        
        return return_embeddings, return_contacts, return_logits
    
    def _load_substrings(self, substring_path: Optional[str]) -> Optional[Dict[str, str]]:
        """
        Load substrings from file if specified.
        
        Args:
            substring_path: Path to substring file
            
        Returns:
            Dictionary mapping sequence labels to substrings, or None
        """
        if not substring_path:
            return None
            
        try:
            with open(substring_path) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                substring_dict = {rows[0]: rows[1] for rows in reader}
            logger.info(f"Loaded {len(substring_dict)} substrings from {substring_path}")
            return substring_dict
        except Exception as e:
            logger.error(f"Error loading substrings from {substring_path}: {e}")
            raise
    
    def _validate_configuration(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        
        # Validate max length
        if self.max_length <= 0:
            raise ValueError(f"Max length must be positive, got {self.max_length}")
        
        # Validate precision
        valid_precisions = ["float16", "16", "half", "float32", "32", "full"]
        if self.precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.precision}. Must be one of {valid_precisions}")
        
        # Validate output types
        if not self.output_types:
            raise ValueError("At least one output type must be specified")
        
        # Validate streaming configuration
        if self.streaming_output and self.num_workers <= 0:
            raise ValueError("Number of workers must be positive when streaming is enabled")
        
        # Validate FASTA path
        if not os.path.exists(self.fasta_path):
            raise ValueError(f"FASTA file not found: {self.fasta_path}")
        
        # Validate substring path if specified
        if self.substring_path and not os.path.exists(self.substring_path):
            raise ValueError(f"Substring file not found: {self.substring_path}")
    
    def precision_to_dtype(self, framework: str = "numpy") -> Union[torch.dtype, np.dtype]:
        """
        Convert precision string to appropriate dtype.
        
        Args:
            framework: Framework to get dtype for ('torch' or 'numpy')
            
        Returns:
            Appropriate dtype for the framework
            
        Raises:
            ValueError: If precision or framework is invalid
        """
        half_precision = ["float16", "16", "half"]
        full_precision = ["float32", "32", "full"]
        
        if self.precision in half_precision:
            if framework == "torch":
                return torch.float16
            elif framework == "numpy":
                return np.float16
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        elif self.precision in full_precision:
            if framework == "torch":
                return torch.float32
            elif framework == "numpy":
                return np.float32
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        else:
            raise ValueError(
                f"Unsupported precision: {self.precision}. "
                f"Supported values are {half_precision} or {full_precision}."
            )
    
    def get_output_directory(self, output_type: str) -> str:
        """
        Get output directory for a specific output type.
        
        Args:
            output_type: Type of output
            
        Returns:
            Output directory path
        """
        return os.path.join(self.output_path, output_type)
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information for logging/debugging.
        
        Returns:
            Dictionary containing device information
        """
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    
    def get_streaming_info(self) -> Dict[str, Any]:
        """
        Get streaming configuration information.
        
        Returns:
            Dictionary containing streaming configuration
        """
        return {
            "streaming_output": self.streaming_output,
            "num_workers": self.num_workers,
            "max_in_flight": self.max_in_flight,
            "flush_batches_after": self.flush_batches_after,
        }
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all configuration parameters.
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            "model_name": self.model_name,
            "fasta_path": self.fasta_path,
            "output_path": self.output_path,
            "output_prefix": self.output_prefix,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "layers": self.layers,
            "output_types": self.output_types,
            "precision": self.precision,
            "device": str(self.device),
            "streaming_output": self.streaming_output,
            "num_workers": self.num_workers,
            "return_embeddings": self.return_embeddings,
            "return_contacts": self.return_contacts,
            "return_logits": self.return_logits,
            "flatten": self.flatten,
            "discard_padding": self.discard_padding,
            "disable_special_tokens": self.disable_special_tokens,
            "has_substrings": self.substring_dict is not None,
            "num_substrings": len(self.substring_dict) if self.substring_dict else 0,
        }
    
    def log_configuration(self) -> None:
        """Log configuration summary for debugging."""
        config_summary = self.get_configuration_summary()
        logger.info("EmbedderConfig initialized with:")
        for key, value in config_summary.items():
            logger.info(f"  {key}: {value}")
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"EmbedderConfig(model_name='{self.model_name}', output_types={self.output_types}, device='{self.device}')"