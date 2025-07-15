"""
StreamingIO class for handling I/O operations in PEPE embedders.

This module provides a centralized interface for managing streaming I/O operations,
including file path generation, directory creation, disk space allocation,
and checkpoint management.
"""

import os
import csv
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from pepe.utils import MultiIODispatcher, check_disk_free_space

logger = logging.getLogger(__name__)


class StreamingIO:
    """
    Handles all I/O operations for embedding extraction and streaming.
    
    This class encapsulates file management, streaming I/O coordination,
    and checkpoint operations to provide a clean interface for data persistence.
    """
    
    def __init__(
        self,
        output_path: str,
        output_prefix: str,
        model_name: str,
        output_types: List[str],
        precision: str,
        streaming_output: bool = False,
        num_workers: int = 1,
        flush_batches_after: int = 64 * 1024 * 1024,  # 64MB default
        layers: Optional[List[int]] = None,
        num_heads: Optional[int] = None,
        flatten: bool = False,
        fasta_path: Optional[str] = None,
    ):
        """
        Initialize StreamingIO with configuration parameters.
        
        Args:
            output_path: Base output directory path
            output_prefix: Prefix for output files
            model_name: Name of the model being used
            output_types: List of output types to generate
            precision: Precision for data storage (e.g., 'float16', 'float32')
            streaming_output: Whether to use streaming output
            num_workers: Number of workers for streaming
            flush_batches_after: Flush threshold in bytes
            layers: List of layer indices (if applicable)
            num_heads: Number of attention heads (if applicable)
            flatten: Whether to flatten output tensors
            fasta_path: Path to FASTA file (for sequence index export)
        """
        self.output_path = output_path
        self.output_prefix = output_prefix
        self.model_name = model_name
        self.output_types = output_types
        self.precision = precision
        self.streaming_output = streaming_output
        self.num_workers = num_workers if streaming_output else 1
        self.flush_batches_after = flush_batches_after
        self.layers = layers or []
        self.num_heads = num_heads
        self.flatten = flatten
        self.fasta_path = fasta_path
        
        # Set up checkpoint directory
        self.checkpoint_dir = self.output_path
        
        # Initialize I/O dispatcher (will be set up when needed)
        self.io_dispatcher: Optional[MultiIODispatcher] = None
        self.memmap_registry: Dict[Tuple[str, Optional[int], Optional[int]], np.memmap] = {}
        
    def _precision_to_dtype(self, framework: str = "numpy") -> np.dtype:
        """Convert precision string to numpy dtype."""
        half_precision = ["float16", "16", "half"]
        full_precision = ["float32", "32", "full"]
        
        if self.precision in half_precision:
            return np.float16
        elif self.precision in full_precision:
            return np.float32
        else:
            raise ValueError(
                f"Unsupported precision: {self.precision}. "
                f"Supported values are {half_precision} or {full_precision}."
            )
    
    def make_output_filepath(
        self, 
        output_type: str, 
        output_dir: str, 
        layer: Optional[int] = None, 
        head: Optional[int] = None
    ) -> str:
        """
        Generate output file path for a given output type.
        
        Args:
            output_type: Type of output (e.g., 'mean_pooled', 'per_token')
            output_dir: Directory for the output
            layer: Layer index (if applicable)
            head: Head index (if applicable)
            
        Returns:
            Full file path for the output
        """
        base = f"{self.output_prefix}_{self.model_name}_{output_type}"
        if layer is not None:
            base += f"_layer_{layer}"
        if head is not None:
            base += f"_head_{head + 1}"
        return os.path.join(output_dir, base + ".npy")
    
    def create_output_dirs(self) -> None:
        """Create output directories for all output types."""
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)
                logger.debug(f"Created output directory: {output_type_path}")
    
    def preallocate_disk_space(
        self, 
        output_data_registry: Dict[str, Dict[str, Any]], 
        num_sequences: int,
        max_length: int,
        embedding_size: int
    ) -> Dict[Tuple[str, Optional[int], Optional[int]], np.memmap]:
        """
        Preallocate disk space for streaming output using memory mapping.
        
        Args:
            output_data_registry: Registry of output data configurations
            num_sequences: Total number of sequences
            max_length: Maximum sequence length
            embedding_size: Size of embedding vectors
            
        Returns:
            Dictionary mapping output keys to memory-mapped arrays
        """
        memmap_registry = {}
        total_bytes = 0
        np_dtype = self._precision_to_dtype()
        
        for output_type in self.output_types:
            output_config = output_data_registry[output_type]
            output_data = output_config["output_data"]
            output_dir = output_config["output_dir"]
            
            # Calculate shape based on output type
            if output_type == "logits":
                shape = (num_sequences, max_length)
            elif output_type == "mean_pooled":
                shape = (num_sequences, embedding_size)
            elif output_type == "per_token":
                if self.flatten:
                    shape = (num_sequences, max_length * embedding_size)
                else:
                    shape = (num_sequences, max_length, embedding_size)
            elif output_type == "substring_pooled":
                shape = (num_sequences, embedding_size)
            elif output_type in ["attention_head", "attention_layer", "attention_model"]:
                if self.flatten:
                    shape = (num_sequences, max_length**2)
                else:
                    shape = (num_sequences, max_length, max_length)
            else:
                # Default shape
                shape = (num_sequences, embedding_size)
            
            bytes_per_array = np.dtype(np_dtype).itemsize * np.prod(shape)
            
            if isinstance(output_data, dict):
                for layer in self.layers:
                    if isinstance(output_data[layer], dict):  # e.g., attention_head
                        for head in range(self.num_heads or 1):
                            file_path = self.make_output_filepath(
                                output_type, output_dir, layer, head
                            )
                            mode = "r+" if os.path.exists(file_path) else "w+"
                            mmap_array = open_memmap(
                                file_path, mode=mode, dtype=np_dtype, shape=shape
                            )
                            output_data[layer][head] = mmap_array
                            memmap_registry[(output_type, layer, head)] = mmap_array
                            total_bytes += bytes_per_array
                    else:
                        file_path = self.make_output_filepath(
                            output_type, output_dir, layer
                        )
                        mode = "r+" if os.path.exists(file_path) else "w+"
                        mmap_array = open_memmap(
                            file_path, mode=mode, dtype=np_dtype, shape=shape
                        )
                        output_data[layer] = mmap_array
                        memmap_registry[(output_type, layer, None)] = mmap_array
                        total_bytes += bytes_per_array
            else:
                file_path = self.make_output_filepath(output_type, output_dir)
                mode = "r+" if os.path.exists(file_path) else "w+"
                mmap_array = open_memmap(
                    file_path, mode=mode, dtype=np_dtype, shape=shape
                )
                # Update the output_data reference
                output_config["output_data"] = mmap_array
                memmap_registry[(output_type, None, None)] = mmap_array
                total_bytes += bytes_per_array
        
        logger.info(f"Preparing to write {total_bytes / 1024**3:.2f} GB to disk.")
        check_disk_free_space(self.output_path, total_bytes)
        
        self.memmap_registry = memmap_registry
        return memmap_registry
    
    def initialize_streaming_io(self) -> None:
        """Initialize the streaming I/O dispatcher."""
        if self.streaming_output and self.memmap_registry:
            self.io_dispatcher = MultiIODispatcher(
                self.memmap_registry,
                num_workers=self.num_workers,
                flush_bytes_limit=self.flush_batches_after,
                heavy_output_type="per_token",
                checkpoint_dir=self.checkpoint_dir,
            )
            
            # Check for resume info
            resume_info = self.io_dispatcher.get_resume_info()
            if resume_info:
                logger.info(f"Resuming from checkpoint: {resume_info}")
    
    def get_queue_fullness(self) -> float:
        """Get the current queue fullness of the I/O dispatcher."""
        if self.io_dispatcher:
            return self.io_dispatcher.queue_fullness()
        return 0.0
    
    def stop_streaming(self) -> None:
        """Stop the streaming I/O dispatcher."""
        if self.io_dispatcher:
            self.io_dispatcher.stop()
            self.io_dispatcher = None
    
    def export_to_disk(
        self, 
        output_data_registry: Dict[str, Dict[str, Any]], 
        prepare_tensor_fn: callable
    ) -> None:
        """
        Export all output data to disk.
        
        Args:
            output_data_registry: Registry of output data configurations
            prepare_tensor_fn: Function to prepare tensors for export
        """
        for output_type in self.output_types:
            logger.info(f"Saving {output_type} representations...")
            
            output_config = output_data_registry[output_type]
            output_data = output_config["output_data"]
            output_dir = output_config["output_dir"]
            
            if isinstance(output_data, dict):
                for layer in self.layers:
                    if isinstance(output_data[layer], dict):  # e.g., attention_head
                        for head in range(self.num_heads or 1):
                            tensor = prepare_tensor_fn(
                                output_data[layer][head], self.flatten
                            )
                            file_path = self.make_output_filepath(
                                output_type, output_dir, layer, head
                            )
                            np.save(file_path, tensor)
                            logger.info(
                                f"Saved {output_type} layer {layer} head {head + 1} to {file_path}"
                            )
                    else:
                        # Handle layer-based outputs
                        flatten = self.flatten and output_type == "per_token"
                        tensor = prepare_tensor_fn(output_data[layer], flatten)
                        file_path = self.make_output_filepath(
                            output_type, output_dir, layer
                        )
                        np.save(file_path, tensor)
                        logger.info(f"Saved {output_type} layer {layer} to {file_path}")
            else:
                # Handle model-level outputs
                tensor = prepare_tensor_fn(output_data, self.flatten)
                file_path = self.make_output_filepath(output_type, output_dir)
                np.save(file_path, tensor)
                logger.info(f"Saved {output_type} to {file_path}")
    
    def export_sequence_indices(self, sequence_labels: List[str]) -> None:
        """
        Export sequence indices to a CSV file.
        
        Args:
            sequence_labels: List of sequence labels/identifiers
        """
        if not self.fasta_path:
            logger.warning("No FASTA path provided, skipping sequence index export")
            return
            
        input_file_name = os.path.basename(self.fasta_path)
        output_file_name = os.path.splitext(input_file_name)[0] + "_idx.csv"
        output_file_idx = os.path.join(self.output_path, output_file_name)
        
        with open(output_file_idx, "w") as f:
            f.write("index,sequence_id\n")
            for i, label in enumerate(sequence_labels):
                f.write(f"{i},{label}\n")
        
        logger.info(f"Saved sequence indices to {output_file_idx}")
    
    def cleanup_checkpoint(self) -> None:
        """Clean up checkpoint files after successful completion."""
        checkpoint_file = os.path.join(self.checkpoint_dir, "global_checkpoint.json")
        if os.path.exists(checkpoint_file):
            try:
                os.remove(checkpoint_file)
                logger.info(f"Cleaned up checkpoint file: {checkpoint_file}")
            except Exception as e:
                logger.error(
                    f"Warning: Could not remove checkpoint file {checkpoint_file}: {e}"
                )
        else:
            logger.debug("No checkpoint file found to clean up.")
    
    def get_resume_info(self) -> Optional[Dict[str, Any]]:
        """Get resume information from the I/O dispatcher."""
        if self.io_dispatcher:
            return self.io_dispatcher.get_resume_info()
        return None 