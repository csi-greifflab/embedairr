import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from pepe.embedders.components.embedding_processor import EmbeddingProcessor


class TestEmbeddingProcessor:
    """Test suite for EmbeddingProcessor class."""

    @pytest.fixture
    def sample_layers(self):
        """Sample layer indices for testing."""
        return [1, 2, 3]

    @pytest.fixture
    def sample_device(self):
        """Sample device for testing."""
        return torch.device("cpu")

    @pytest.fixture
    def sample_special_tokens(self):
        """Sample special tokens tensor."""
        return torch.tensor([0, 1, 2], dtype=torch.int8)

    @pytest.fixture
    def processor(self, sample_layers, sample_device):
        """Create a basic EmbeddingProcessor instance."""
        return EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=False,
            discard_padding=False,
            streaming_output=False,
            io_dispatcher=None,
            special_tokens=None,
            num_heads=8,
        )

    @pytest.fixture
    def streaming_processor(self, sample_layers, sample_device):
        """Create a streaming EmbeddingProcessor instance."""
        mock_dispatcher = Mock()
        return EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=False,
            discard_padding=False,
            streaming_output=True,
            io_dispatcher=mock_dispatcher,
            special_tokens=None,
            num_heads=8,
        )

    @pytest.fixture
    def sample_representations(self):
        """Sample representations tensor."""
        return {
            1: torch.randn(2, 10, 768),  # batch_size=2, seq_len=10, hidden_size=768
            2: torch.randn(2, 10, 768),
            3: torch.randn(2, 10, 768),
        }

    @pytest.fixture
    def sample_logits(self):
        """Sample logits tensor."""
        return {
            0: torch.randn(2, 10, 1000),  # batch_size=2, seq_len=10, vocab_size=1000
            1: torch.randn(2, 10, 1000),
            2: torch.randn(2, 10, 1000),
        }

    @pytest.fixture
    def sample_attention_matrices(self):
        """Sample attention matrices tensor."""
        return torch.randn(3, 2, 8, 10, 10)  # layers=3, batch=2, heads=8, seq_len=10, seq_len=10

    @pytest.fixture
    def sample_pooling_mask(self):
        """Sample pooling mask tensor."""
        return torch.ones(2, 10, dtype=torch.bool)  # batch_size=2, seq_len=10

    @pytest.fixture
    def sample_batch_labels(self):
        """Sample batch labels."""
        return ["seq1", "seq2"]

    def test_initialization(self, sample_layers, sample_device):
        """Test EmbeddingProcessor initialization."""
        processor = EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=True,
            discard_padding=True,
            streaming_output=True,
            io_dispatcher=Mock(),
            special_tokens=torch.tensor([0, 1, 2]),
            num_heads=12,
        )
        
        assert processor.layers == sample_layers
        assert processor.precision == "float32"
        assert processor.device == sample_device
        assert processor.flatten is True
        assert processor.discard_padding is True
        assert processor.streaming_output is True
        assert processor.io_dispatcher is not None
        assert processor.special_tokens is not None
        assert processor.num_heads == 12
        
        # Check data structures initialization
        assert all(layer in processor.logits_data for layer in sample_layers)
        assert all(layer in processor.mean_pooled_data for layer in sample_layers)
        assert all(layer in processor.per_token_data for layer in sample_layers)
        assert all(layer in processor.substring_pooled_data for layer in sample_layers)
        assert all(layer in processor.attention_layer_data for layer in sample_layers)
        assert all(layer in processor.attention_head_data for layer in sample_layers)

    def test_precision_to_dtype(self, processor):
        """Test precision conversion to dtype."""
        # Test float16
        assert processor._precision_to_dtype("float16", "torch") == torch.float16
        assert processor._precision_to_dtype("16", "torch") == torch.float16
        assert processor._precision_to_dtype("half", "torch") == torch.float16
        assert processor._precision_to_dtype("float16", "numpy") == np.float16
        
        # Test float32
        assert processor._precision_to_dtype("float32", "torch") == torch.float32
        assert processor._precision_to_dtype("32", "torch") == torch.float32
        assert processor._precision_to_dtype("full", "torch") == torch.float32
        assert processor._precision_to_dtype("float32", "numpy") == np.float32
        
        # Test invalid precision
        with pytest.raises(ValueError):
            processor._precision_to_dtype("invalid", "torch")

    def test_mask_special_tokens(self, processor):
        """Test special token masking."""
        input_tensor = torch.tensor([0, 1, 2, 3, 4, 5])
        special_tokens = torch.tensor([0, 1, 2])
        
        # Test with provided special tokens
        mask = processor.mask_special_tokens(input_tensor, special_tokens)
        expected = torch.tensor([False, False, False, True, True, True])
        assert torch.equal(mask, expected)
        
        # Test with default special tokens
        mask = processor.mask_special_tokens(input_tensor)
        expected = torch.tensor([False, False, False, True, True, True])
        assert torch.equal(mask, expected)

    def test_to_numpy(self, processor):
        """Test tensor to numpy conversion."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = processor.to_numpy(tensor)
        
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_prepare_tensor(self, processor):
        """Test tensor preparation."""
        data_list = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
        
        # Test without flattening
        result = processor.prepare_tensor(data_list, flatten=False)
        expected_shape = (2, 2, 2)
        assert result.shape == expected_shape
        
        # Test with flattening
        result = processor.prepare_tensor(data_list, flatten=True)
        expected_shape = (2, 4)
        assert result.shape == expected_shape

    def test_extract_logits_batch_mode(self, processor, sample_logits):
        """Test logits extraction in batch mode."""
        offset = 0
        
        processor.extract_logits(sample_logits, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            assert len(processor.logits_data[layer]) > 0
            assert processor.logits_data[layer][0].shape == sample_logits[layer - 1].shape

    def test_extract_logits_streaming_mode(self, streaming_processor, sample_logits):
        """Test logits extraction in streaming mode."""
        offset = 0
        
        streaming_processor.extract_logits(sample_logits, offset)
        
        # Check that data was sent to dispatcher
        assert streaming_processor.io_dispatcher.enqueue.called
        call_args = streaming_processor.io_dispatcher.enqueue.call_args_list
        
        # Should have one call per layer
        assert len(call_args) == len(streaming_processor.layers)
        
        # Check call parameters
        for i, call in enumerate(call_args):
            kwargs = call[1]
            assert kwargs['output_type'] == 'logits'
            assert kwargs['layer'] == streaming_processor.layers[i]
            assert kwargs['head'] is None
            assert kwargs['offset'] == offset

    def test_extract_mean_pooled_batch_mode(self, processor, sample_representations, sample_pooling_mask, sample_batch_labels):
        """Test mean pooled extraction in batch mode."""
        offset = 0
        
        processor.extract_mean_pooled(sample_representations, sample_batch_labels, sample_pooling_mask, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            assert len(processor.mean_pooled_data[layer]) > 0
            stored_tensor = processor.mean_pooled_data[layer][0]
            assert stored_tensor.shape[0] == len(sample_batch_labels)
            assert stored_tensor.shape[1] == sample_representations[layer].shape[2]  # embedding dim

    def test_extract_mean_pooled_streaming_mode(self, streaming_processor, sample_representations, sample_pooling_mask, sample_batch_labels):
        """Test mean pooled extraction in streaming mode."""
        offset = 0
        
        streaming_processor.extract_mean_pooled(sample_representations, sample_batch_labels, sample_pooling_mask, offset)
        
        # Check that data was sent to dispatcher
        assert streaming_processor.io_dispatcher.enqueue.called
        call_args = streaming_processor.io_dispatcher.enqueue.call_args_list
        
        # Should have one call per layer
        assert len(call_args) == len(streaming_processor.layers)
        
        # Check call parameters
        for i, call in enumerate(call_args):
            kwargs = call[1]
            assert kwargs['output_type'] == 'mean_pooled'
            assert kwargs['layer'] == streaming_processor.layers[i]
            assert kwargs['head'] is None
            assert kwargs['offset'] == offset

    def test_extract_per_token_batch_mode(self, processor, sample_representations, sample_batch_labels):
        """Test per-token extraction in batch mode."""
        offset = 0
        
        processor.extract_per_token(sample_representations, sample_batch_labels, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            assert len(processor.per_token_data[layer]) > 0
            stored_tensor = processor.per_token_data[layer][0]
            assert stored_tensor.shape[0] == len(sample_batch_labels)
            assert stored_tensor.shape[1] == sample_representations[layer].shape[1]  # seq_len
            assert stored_tensor.shape[2] == sample_representations[layer].shape[2]  # embedding dim

    def test_extract_per_token_flattened(self, sample_layers, sample_device, sample_representations, sample_batch_labels):
        """Test per-token extraction with flattening."""
        processor = EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=True,
            discard_padding=False,
            streaming_output=False,
            io_dispatcher=None,
            special_tokens=None,
            num_heads=8,
        )
        
        offset = 0
        processor.extract_per_token(sample_representations, sample_batch_labels, offset)
        
        # Check that data was flattened
        for layer in processor.layers:
            stored_tensor = processor.per_token_data[layer][0]
            assert stored_tensor.shape[0] == len(sample_batch_labels)
            # Should be flattened: seq_len * embedding_dim
            expected_flat_dim = sample_representations[layer].shape[1] * sample_representations[layer].shape[2]
            assert stored_tensor.shape[1] == expected_flat_dim

    def test_extract_substring_pooled_batch_mode(self, processor, sample_representations, sample_pooling_mask):
        """Test substring pooled extraction in batch mode."""
        offset = 0
        
        processor.extract_substring_pooled(sample_representations, sample_pooling_mask, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            assert len(processor.substring_pooled_data[layer]) > 0
            stored_tensor = processor.substring_pooled_data[layer][0]
            assert stored_tensor.shape[0] == len(sample_pooling_mask)
            assert stored_tensor.shape[1] == sample_representations[layer].shape[2]  # embedding dim

    def test_extract_attention_head_batch_mode(self, processor, sample_attention_matrices, sample_batch_labels):
        """Test attention head extraction in batch mode."""
        offset = 0
        
        processor.extract_attention_head(sample_attention_matrices, sample_batch_labels, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            for head in range(processor.num_heads):
                assert len(processor.attention_head_data[layer][head]) > 0
                stored_tensor = processor.attention_head_data[layer][head][0]
                assert stored_tensor.shape[0] == len(sample_batch_labels)

    def test_extract_attention_head_no_heads(self, sample_layers, sample_device, sample_attention_matrices, sample_batch_labels):
        """Test attention head extraction when num_heads is None."""
        processor = EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=False,
            discard_padding=False,
            streaming_output=False,
            io_dispatcher=None,
            special_tokens=None,
            num_heads=None,
        )
        
        offset = 0
        
        with patch('pepe.embedders.components.embedding_processor.logger') as mock_logger:
            processor.extract_attention_head(sample_attention_matrices, sample_batch_labels, offset)
            mock_logger.warning.assert_called_once()

    def test_extract_attention_layer_batch_mode(self, processor, sample_attention_matrices, sample_batch_labels):
        """Test attention layer extraction in batch mode."""
        offset = 0
        
        processor.extract_attention_layer(sample_attention_matrices, sample_batch_labels, offset)
        
        # Check that data was stored correctly
        for layer in processor.layers:
            assert len(processor.attention_layer_data[layer]) > 0
            stored_tensor = processor.attention_layer_data[layer][0]
            assert stored_tensor.shape[0] == len(sample_batch_labels)

    def test_extract_attention_model_batch_mode(self, processor, sample_attention_matrices, sample_batch_labels):
        """Test attention model extraction in batch mode."""
        offset = 0
        
        processor.extract_attention_model(sample_attention_matrices, sample_batch_labels, offset)
        
        # Check that data was stored correctly
        assert len(processor.attention_model_data) > 0
        stored_tensor = processor.attention_model_data[0]
        assert stored_tensor.shape[0] == len(sample_batch_labels)

    def test_get_extraction_method(self, processor):
        """Test getting extraction methods."""
        assert processor.get_extraction_method("logits") == processor.extract_logits
        assert processor.get_extraction_method("mean_pooled") == processor.extract_mean_pooled
        assert processor.get_extraction_method("per_token") == processor.extract_per_token
        assert processor.get_extraction_method("substring_pooled") == processor.extract_substring_pooled
        assert processor.get_extraction_method("attention_head") == processor.extract_attention_head
        assert processor.get_extraction_method("attention_layer") == processor.extract_attention_layer
        assert processor.get_extraction_method("attention_model") == processor.extract_attention_model
        assert processor.get_extraction_method("invalid") is None

    def test_get_output_data(self, processor):
        """Test getting output data."""
        # Add some sample data
        processor.logits_data[1] = [torch.randn(2, 10, 1000)]
        processor.mean_pooled_data[1] = [torch.randn(2, 768)]
        
        assert processor.get_output_data("logits") == processor.logits_data
        assert processor.get_output_data("mean_pooled") == processor.mean_pooled_data
        assert processor.get_output_data("per_token") == processor.per_token_data
        assert processor.get_output_data("substring_pooled") == processor.substring_pooled_data
        assert processor.get_output_data("attention_head") == processor.attention_head_data
        assert processor.get_output_data("attention_layer") == processor.attention_layer_data
        assert processor.get_output_data("attention_model") == processor.attention_model_data
        assert processor.get_output_data("invalid") == {}

    def test_clear_data(self, processor):
        """Test clearing all data."""
        # Add some sample data
        processor.logits_data[1] = [torch.randn(2, 10, 1000)]
        processor.mean_pooled_data[1] = [torch.randn(2, 768)]
        processor.per_token_data[1] = [torch.randn(2, 10, 768)]
        processor.substring_pooled_data[1] = [torch.randn(2, 768)]
        processor.attention_layer_data[1] = [torch.randn(2, 10, 10)]
        processor.attention_head_data[1][0] = [torch.randn(2, 10, 10)]
        processor.attention_model_data = [torch.randn(2, 10, 10)]
        
        # Clear data
        processor.clear_data()
        
        # Check that all data is cleared
        for layer in processor.layers:
            assert len(processor.logits_data[layer]) == 0
            assert len(processor.mean_pooled_data[layer]) == 0
            assert len(processor.per_token_data[layer]) == 0
            assert len(processor.substring_pooled_data[layer]) == 0
            assert len(processor.attention_layer_data[layer]) == 0
            
            if processor.num_heads:
                for head in range(processor.num_heads):
                    assert len(processor.attention_head_data[layer][head]) == 0
        
        assert len(processor.attention_model_data) == 0

    def test_discard_padding_mode(self, sample_layers, sample_device, sample_representations, sample_batch_labels):
        """Test per-token extraction with discard_padding mode."""
        processor = EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=False,
            discard_padding=True,
            streaming_output=False,
            io_dispatcher=None,
            special_tokens=None,
            num_heads=8,
        )
        
        offset = 0
        
        with patch('pepe.embedders.components.embedding_processor.logger') as mock_logger:
            processor.extract_per_token(sample_representations, sample_batch_labels, offset)
            mock_logger.warning.assert_called_once()

    def test_streaming_mode_comprehensive(self, sample_layers, sample_device):
        """Test comprehensive streaming mode functionality."""
        mock_dispatcher = Mock()
        processor = EmbeddingProcessor(
            layers=sample_layers,
            precision="float32",
            device=sample_device,
            flatten=False,
            discard_padding=False,
            streaming_output=True,
            io_dispatcher=mock_dispatcher,
            special_tokens=None,
            num_heads=8,
        )
        
        # Test data
        sample_representations = {
            1: torch.randn(2, 10, 768),
            2: torch.randn(2, 10, 768),
            3: torch.randn(2, 10, 768),
        }
        sample_logits = {
            0: torch.randn(2, 10, 1000),
            1: torch.randn(2, 10, 1000),
            2: torch.randn(2, 10, 1000),
        }
        sample_attention_matrices = torch.randn(3, 2, 8, 10, 10)
        sample_pooling_mask = torch.ones(2, 10, dtype=torch.bool)
        sample_batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Test all extraction methods
        processor.extract_logits(sample_logits, offset)
        processor.extract_mean_pooled(sample_representations, sample_batch_labels, sample_pooling_mask, offset)
        processor.extract_per_token(sample_representations, sample_batch_labels, offset)
        processor.extract_substring_pooled(sample_representations, sample_pooling_mask, offset)
        processor.extract_attention_head(sample_attention_matrices, sample_batch_labels, offset)
        processor.extract_attention_layer(sample_attention_matrices, sample_batch_labels, offset)
        processor.extract_attention_model(sample_attention_matrices, sample_batch_labels, offset)
        
        # Check that all methods called the dispatcher
        assert mock_dispatcher.enqueue.call_count > 0
        
        # Check that no data was stored locally
        for layer in processor.layers:
            assert len(processor.logits_data[layer]) == 0
            assert len(processor.mean_pooled_data[layer]) == 0
            assert len(processor.per_token_data[layer]) == 0
            assert len(processor.substring_pooled_data[layer]) == 0
            assert len(processor.attention_layer_data[layer]) == 0
            for head in range(processor.num_heads):
                assert len(processor.attention_head_data[layer][head]) == 0
        
        assert len(processor.attention_model_data) == 0


if __name__ == "__main__":
    pytest.main([__file__]) 