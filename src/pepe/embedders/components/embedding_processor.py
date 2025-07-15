import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Handles embedding extraction and processing from model outputs.
    
    This class encapsulates all the logic for extracting different types of embeddings
    and attention matrices from model outputs, including logits, mean pooled embeddings,
    per-token embeddings, substring pooled embeddings, and attention matrices.
    """
    
    def __init__(
        self,
        layers: List[int],
        precision: str,
        device: torch.device,
        flatten: bool = False,
        discard_padding: bool = False,
        streaming_output: bool = False,
        io_dispatcher: Optional[Any] = None,
        special_tokens: Optional[torch.Tensor] = None,
        num_heads: Optional[int] = None,
    ):
        """
        Initialize the EmbeddingProcessor.
        
        Args:
            layers: List of layer indices to extract embeddings from
            precision: Precision for tensor operations ('float16', 'float32', etc.)
            device: Device to perform computations on
            flatten: Whether to flatten output tensors
            discard_padding: Whether to discard padding tokens
            streaming_output: Whether to use streaming output mode
            io_dispatcher: I/O dispatcher for streaming output
            special_tokens: Tensor of special token IDs
            num_heads: Number of attention heads (for attention extraction)
        """
        self.layers = layers
        self.precision = precision
        self.device = device
        self.flatten = flatten
        self.discard_padding = discard_padding
        self.streaming_output = streaming_output
        self.io_dispatcher = io_dispatcher
        self.special_tokens = special_tokens
        self.num_heads = num_heads
        
        # Initialize storage for non-streaming mode
        self.logits_data = {layer: [] for layer in layers}
        self.mean_pooled_data = {layer: [] for layer in layers}
        self.per_token_data = {layer: [] for layer in layers}
        self.substring_pooled_data = {layer: [] for layer in layers}
        self.attention_head_data = {layer: {head: [] for head in range(num_heads)} for layer in layers} if num_heads else {}
        self.attention_layer_data = {layer: [] for layer in layers}
        self.attention_model_data = []
    
    def _precision_to_dtype(self, precision: str, framework: str):
        """Convert precision string to appropriate dtype."""
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
        
        raise ValueError(
            f"Unsupported precision: {precision}. Supported values are {half_precision} or {full_precision}."
        )
    
    def mask_special_tokens(self, input_tensor: torch.Tensor, special_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create a boolean mask for special tokens in the input tensor.
        
        Args:
            input_tensor: Input tensor to mask
            special_tokens: Optional tensor of special token IDs
            
        Returns:
            Boolean mask tensor (True for non-special tokens)
        """
        if special_tokens is not None:
            # Create a boolean mask: True where the value is not in special_tokens
            mask = ~torch.isin(input_tensor, special_tokens)
        else:
            # Create a boolean mask: True where the value is not 0, 1, or 2
            mask = (input_tensor != 0) & (input_tensor != 1) & (input_tensor != 2)
        
        return mask
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        return tensor.detach().cpu().contiguous().numpy()
    
    def prepare_tensor(self, data_list: List[torch.Tensor], flatten: bool) -> np.ndarray:
        """Prepare tensor for saving."""
        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()
    
    def extract_logits(self, logits: Dict[int, torch.Tensor], offset: int) -> None:
        """
        Extract logits from model outputs.
        
        Args:
            logits: Dictionary mapping layer indices to logit tensors
            offset: Batch offset for streaming output
        """
        for layer in self.layers:
            tensor = logits[layer - 1]
            
            if self.streaming_output and self.io_dispatcher:
                self.io_dispatcher.enqueue(
                    output_type="logits",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self.to_numpy(tensor),
                )
            else:
                self.logits_data[layer].append(tensor)
    
    def extract_mean_pooled(
        self,
        representations: Dict[int, torch.Tensor],
        batch_labels: List[str],
        pooling_mask: torch.Tensor,
        offset: int,
    ) -> None:
        """
        Extract mean pooled embeddings.
        
        Args:
            representations: Dictionary mapping layer indices to representation tensors
            batch_labels: List of sequence labels in the batch
            pooling_mask: Boolean mask for pooling (True for tokens to include)
            offset: Batch offset for streaming output
        """
        for layer in self.layers:
            tensor = torch.stack([
                (pooling_mask[i].unsqueeze(-1) * representations[layer][i]).sum(0)
                / pooling_mask[i].unsqueeze(-1).sum(0)
                for i in range(len(batch_labels))
            ])
            
            if self.streaming_output and self.io_dispatcher:
                self.io_dispatcher.enqueue(
                    output_type="mean_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self.to_numpy(tensor),
                )
            else:
                self.mean_pooled_data[layer].append(tensor)
    
    def extract_per_token(
        self,
        representations: Dict[int, torch.Tensor],
        batch_labels: List[str],
        offset: int,
    ) -> None:
        """
        Extract per-token embeddings.
        
        Args:
            representations: Dictionary mapping layer indices to representation tensors
            batch_labels: List of sequence labels in the batch
            offset: Batch offset for streaming output
        """
        if not self.discard_padding:
            for layer in self.layers:
                tensor = torch.stack([
                    representations[layer][i] for i in range(len(batch_labels))
                ])
                
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                
                if self.streaming_output and self.io_dispatcher:
                    self.io_dispatcher.enqueue(
                        output_type="per_token",
                        layer=layer,
                        head=None,
                        offset=offset,
                        array=np.ascontiguousarray(tensor.cpu().numpy()),
                    )
                else:
                    self.per_token_data[layer].append(tensor)
        else:
            # TODO: Implement padding removal
            logger.warning("Padding removal feature not implemented yet")
            for layer in self.layers:
                if self.flatten:
                    self.per_token_data[layer].extend([
                        representations[layer][i].flatten(start_dim=1)
                        for i in range(len(batch_labels))
                    ])
                else:
                    self.per_token_data[layer].extend([
                        representations[layer][i] for i in range(len(batch_labels))
                    ])
    
    def extract_substring_pooled(
        self,
        representations: Dict[int, torch.Tensor],
        substring_mask: torch.Tensor,
        offset: int,
    ) -> None:
        """
        Extract substring pooled embeddings.
        
        Args:
            representations: Dictionary mapping layer indices to representation tensors
            substring_mask: Boolean mask for substring pooling
            offset: Batch offset for streaming output
        """
        for layer in self.layers:
            tensor = torch.stack([
                (mask.unsqueeze(-1) * representations[layer][i]).sum(0)
                / mask.unsqueeze(-1).sum(0)
                for i, mask in enumerate(substring_mask)
            ])
            
            if self.streaming_output and self.io_dispatcher:
                self.io_dispatcher.enqueue(
                    output_type="substring_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self.to_numpy(tensor),
                )
            else:
                self.substring_pooled_data[layer].append(tensor)
    
    def extract_attention_head(
        self,
        attention_matrices: torch.Tensor,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        """
        Extract attention head matrices.
        
        Args:
            attention_matrices: Attention matrices tensor
            batch_labels: List of sequence labels in the batch
            offset: Batch offset for streaming output
        """
        if not self.num_heads:
            logger.warning("num_heads not set, cannot extract attention heads")
            return
        
        for layer in self.layers:
            for head in range(self.num_heads):
                tensor = torch.stack([
                    attention_matrices[layer - 1, i, head]
                    for i in range(len(batch_labels))
                ])
                
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                
                if self.streaming_output and self.io_dispatcher:
                    self.io_dispatcher.enqueue(
                        output_type="attention_matrices_all_heads",
                        layer=layer,
                        head=head,
                        offset=offset,
                        array=np.ascontiguousarray(tensor.cpu().numpy()),
                    )
                else:
                    self.attention_head_data[layer][head].append(tensor)
    
    def extract_attention_layer(
        self,
        attention_matrices: torch.Tensor,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        """
        Extract attention layer matrices (averaged over heads).
        
        Args:
            attention_matrices: Attention matrices tensor
            batch_labels: List of sequence labels in the batch
            offset: Batch offset for streaming output
        """
        for layer in self.layers:
            tensor = torch.stack([
                attention_matrices[layer - 1, i].mean(0)
                for i in range(len(batch_labels))
            ])
            
            if self.flatten:
                tensor = tensor.flatten(start_dim=1)
            
            if self.streaming_output and self.io_dispatcher:
                self.io_dispatcher.enqueue(
                    output_type="attention_matrices_average_layers",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self.to_numpy(tensor),
                )
            else:
                self.attention_layer_data[layer].append(tensor)
    
    def extract_attention_model(
        self,
        attention_matrices: torch.Tensor,
        batch_labels: List[str],
        offset: int,
    ) -> None:
        """
        Extract attention model matrices (averaged over layers and heads).
        
        Args:
            attention_matrices: Attention matrices tensor
            batch_labels: List of sequence labels in the batch
            offset: Batch offset for streaming output
        """
        tensor = torch.stack([
            attention_matrices[:, i].mean(dim=(0, 1))
            for i in range(len(batch_labels))
        ])
        
        if self.flatten:
            tensor = tensor.flatten(start_dim=1)
        
        if self.streaming_output and self.io_dispatcher:
            self.io_dispatcher.enqueue(
                output_type="attention_matrices_average_all",
                layer=None,
                head=None,
                offset=offset,
                array=np.ascontiguousarray(tensor.cpu().numpy()),
            )
        else:
            self.attention_model_data.append(tensor)
    
    def get_extraction_method(self, output_type: str):
        """
        Get the extraction method for a given output type.
        
        Args:
            output_type: Type of output to extract
            
        Returns:
            Extraction method function
        """
        method_mapping = {
            "logits": self.extract_logits,
            "mean_pooled": self.extract_mean_pooled,
            "per_token": self.extract_per_token,
            "substring_pooled": self.extract_substring_pooled,
            "attention_head": self.extract_attention_head,
            "attention_layer": self.extract_attention_layer,
            "attention_model": self.extract_attention_model,
        }
        
        return method_mapping.get(output_type)
    
    def get_output_data(self, output_type: str):
        """
        Get the stored output data for a given output type.
        
        Args:
            output_type: Type of output data to retrieve
            
        Returns:
            Dictionary or list of stored output data
        """
        data_mapping = {
            "logits": self.logits_data,
            "mean_pooled": self.mean_pooled_data,
            "per_token": self.per_token_data,
            "substring_pooled": self.substring_pooled_data,
            "attention_head": self.attention_head_data,
            "attention_layer": self.attention_layer_data,
            "attention_model": self.attention_model_data,
        }
        
        return data_mapping.get(output_type, {})
    
    def clear_data(self) -> None:
        """Clear all stored output data."""
        for layer in self.layers:
            self.logits_data[layer].clear()
            self.mean_pooled_data[layer].clear()
            self.per_token_data[layer].clear()
            self.substring_pooled_data[layer].clear()
            self.attention_layer_data[layer].clear()
            
            if self.num_heads:
                for head in range(self.num_heads):
                    self.attention_head_data[layer][head].clear()
        
        self.attention_model_data.clear()
        
        # Force garbage collection
        torch.cuda.empty_cache() 