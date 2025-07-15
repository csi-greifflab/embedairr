"""
OutputManager class for handling output operations in PEPE embedders.

This module contains the OutputManager class that handles all output-related
operations extracted from the BaseEmbedder class.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import open_memmap

from pepe.utils import check_disk_free_space

logger = logging.getLogger(__name__)


class OutputManager:
    """
    Manages output operations for embedders including directory creation,
    file path generation, disk space allocation, and data export.
    """

    def __init__(
        self,
        output_path: str,
        model_name: str,
        output_prefix: str,
        output_types: List[str],
        precision: str,
        streaming_output: bool = False,
        flatten: bool = False,
    ):
        """
        Initialize the OutputManager.

        Args:
            output_path: Base output directory path
            model_name: Name of the model being used
            output_prefix: Prefix for output files
            output_types: List of output types to generate
            precision: Precision for output data (float16, float32, etc.)
            streaming_output: Whether to use streaming output mode
            flatten: Whether to flatten multi-dimensional outputs
        """
        self.output_path = output_path
        self.model_name = model_name
        self.output_prefix = output_prefix
        self.output_types = output_types
        self.precision = precision
        self.streaming_output = streaming_output
        self.flatten = flatten
        self.memmap_registry: Dict[Tuple[str, Optional[int], Optional[int]], Any] = {}

    def _precision_to_dtype(self, precision: str, framework: str) -> Union[torch.dtype, np.dtype]:
        """
        Convert precision string to appropriate dtype.

        Args:
            precision: Precision string (float16, float32, etc.)
            framework: Framework type (torch or numpy)

        Returns:
            Appropriate dtype for the framework

        Raises:
            ValueError: If precision is not supported
        """
        half_precision = ["float16", "16", "half"]
        full_precision = ["float32", "32", "full"]
        
        if precision in half_precision:
            if framework == "torch":
                return torch.float16
            elif framework == "numpy":
                return np.float16
        elif precision in full_precision:
            if framework == "torch":
                return torch.float32
            elif framework == "numpy":
                return np.float32
        else:
            raise ValueError(
                f"Unsupported precision: {precision}. "
                f"Supported values are {half_precision} or {full_precision}."
            )

    def create_output_directories(self) -> None:
        """Create output directories for each output type."""
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)
                logger.info(f"Created output directory: {output_type_path}")

    def make_output_filepath(
        self, 
        output_type: str, 
        output_dir: str, 
        layer: Optional[int] = None, 
        head: Optional[int] = None
    ) -> str:
        """
        Generate output file path for given parameters.

        Args:
            output_type: Type of output (embeddings, attention, etc.)
            output_dir: Output directory path
            layer: Layer number (optional)
            head: Head number (optional)

        Returns:
            Complete file path for the output
        """
        base = f"{self.output_prefix}_{self.model_name}_{output_type}"
        if layer is not None:
            base += f"_layer_{layer}"
        if head is not None:
            base += f"_head_{head + 1}"
        return os.path.join(output_dir, base + ".npy")

    def preallocate_disk_space(self, shapes_info: Dict[str, Any]) -> Dict[Tuple[str, Optional[int], Optional[int]], Any]:
        """
        Preallocate disk space for memory-mapped files.

        Args:
            shapes_info: Dictionary containing shape information for each output type

        Returns:
            Memory map registry dictionary

        Raises:
            SystemExit: If insufficient disk space is available
        """
        memmap_registry = {}
        total_bytes = 0
        np_dtype = self._precision_to_dtype(self.precision, "numpy")

        for output_type in self.output_types:
            if output_type not in shapes_info:
                logger.warning(f"No shape information available for output type: {output_type}")
                continue

            shape_info = shapes_info[output_type]
            shape = shape_info["shape"]
            output_dir = os.path.join(self.output_path, output_type)
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            bytes_per_array = np.dtype(np_dtype).itemsize * np.prod(shape)

            # Handle different output data structures
            if "layers" in shape_info:
                layers = shape_info["layers"]
                if "heads" in shape_info:
                    # Handle attention head outputs
                    heads = shape_info["heads"]
                    for layer in layers:
                        for head in range(heads):
                            file_path = self.make_output_filepath(output_type, output_dir, layer, head)
                            mode = "r+" if os.path.exists(file_path) else "w+"
                            memmap_array = open_memmap(file_path, mode=mode, dtype=np_dtype, shape=shape)
                            memmap_registry[(output_type, layer, head)] = memmap_array
                            total_bytes += bytes_per_array
                else:
                    # Handle layer-specific outputs
                    for layer in layers:
                        file_path = self.make_output_filepath(output_type, output_dir, layer)
                        mode = "r+" if os.path.exists(file_path) else "w+"
                        memmap_array = open_memmap(file_path, mode=mode, dtype=np_dtype, shape=shape)
                        memmap_registry[(output_type, layer, None)] = memmap_array
                        total_bytes += bytes_per_array
            else:
                # Handle model-level outputs
                file_path = self.make_output_filepath(output_type, output_dir)
                mode = "r+" if os.path.exists(file_path) else "w+"
                memmap_array = open_memmap(file_path, mode=mode, dtype=np_dtype, shape=shape)
                memmap_registry[(output_type, None, None)] = memmap_array
                total_bytes += bytes_per_array

        logger.info(f"Preparing to write {total_bytes / 1024**3:.2f} GB to disk.")
        check_disk_free_space(self.output_path, total_bytes)
        self.memmap_registry = memmap_registry
        return memmap_registry

    def export_data(
        self, 
        output_type: str, 
        layer: Optional[int], 
        head: Optional[int], 
        data: np.ndarray
    ) -> None:
        """
        Export data to disk using memory-mapped files.

        Args:
            output_type: Type of output data
            layer: Layer number (optional)
            head: Head number (optional)
            data: Data array to export
        """
        key = (output_type, layer, head)
        if key in self.memmap_registry:
            memmap_array = self.memmap_registry[key]
            # Write data to memory-mapped file
            memmap_array[:] = data
            memmap_array.flush()
        else:
            logger.warning(f"No memory-mapped file found for {key}")

    def export_sequence_indices(self, sequence_labels: List[str], fasta_path: str) -> None:
        """
        Export sequence indices to CSV file.

        Args:
            sequence_labels: List of sequence labels
            fasta_path: Path to the original FASTA file
        """
        input_file_name = os.path.basename(fasta_path)
        output_file_name = os.path.splitext(input_file_name)[0] + "_idx.csv"
        output_file_idx = os.path.join(self.output_path, output_file_name)
        
        with open(output_file_idx, "w") as f:
            f.write("index,sequence_id\n")
            for i, label in enumerate(sequence_labels):
                f.write(f"{i},{label}\n")
        
        logger.info(f"Saved sequence indices to {output_file_idx}")

    def get_output_types(self, extract_embeddings: List[str]) -> List[str]:
        """
        Get normalized output types list.

        Args:
            extract_embeddings: List of embedding types to extract

        Returns:
            Normalized list of output types
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

        for option in extract_embeddings:
            if option in options_mapping:
                output_type = options_mapping[option]
                if output_type not in output_types:
                    output_types.append(output_type)

        return output_types

    def _prepare_tensor(self, data_list: List[torch.Tensor], flatten: bool) -> np.ndarray:
        """
        Prepare tensor data for export.

        Args:
            data_list: List of tensor data
            flatten: Whether to flatten the tensors

        Returns:
            Prepared numpy array
        """
        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()

    def export_to_disk(
        self, 
        output_data: Dict[str, Any], 
        layers: List[int], 
        num_heads: Optional[int] = None
    ) -> None:
        """
        Export all output data to disk in batch mode.

        Args:
            output_data: Dictionary containing output data for each type
            layers: List of layer numbers
            num_heads: Number of attention heads (optional)
        """
        for output_type in self.output_types:
            if output_type not in output_data:
                logger.warning(f"No data available for output type: {output_type}")
                continue

            logger.info(f"Saving {output_type} representations...")
            
            type_data = output_data[output_type]["output_data"]
            output_dir = os.path.join(self.output_path, output_type)
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if isinstance(type_data, dict):
                for layer in layers:
                    if layer not in type_data:
                        continue
                        
                    if isinstance(type_data[layer], dict):  # attention_head case
                        if num_heads is None:
                            logger.error(f"num_heads required for {output_type} but not provided")
                            continue
                            
                        for head in range(num_heads):
                            if head not in type_data[layer]:
                                continue
                                
                            tensor = self._prepare_tensor(type_data[layer][head], self.flatten)
                            file_path = self.make_output_filepath(output_type, output_dir, layer, head)
                            np.save(file_path, tensor)
                            logger.info(f"Saved {output_type} layer {layer} head {head + 1} to {file_path}")
                    else:
                        # Handle layer-based outputs
                        flatten_flag = self.flatten and output_type == "per_token"
                        tensor = self._prepare_tensor(type_data[layer], flatten_flag)
                        file_path = self.make_output_filepath(output_type, output_dir, layer)
                        np.save(file_path, tensor)
                        logger.info(f"Saved {output_type} layer {layer} to {file_path}")
            else:
                # Handle model-level outputs
                tensor = self._prepare_tensor(type_data, self.flatten)
                file_path = self.make_output_filepath(output_type, output_dir)
                np.save(file_path, tensor)
                logger.info(f"Saved {output_type} to {file_path}")

    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        # Close memory-mapped files
        for memmap_array in self.memmap_registry.values():
            if hasattr(memmap_array, 'close'):
                memmap_array.close()
        
        self.memmap_registry.clear()
        logger.info("OutputManager cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup() 