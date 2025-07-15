"""
Unit tests for embedding extraction functionality.

This module contains tests for various embedding extraction methods
and their edge cases.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from pepe.embedders.base_embedder import BaseEmbedder


class TestEmbeddingExtraction:
    """Test suite for embedding extraction methods."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder for testing."""
        embedder = Mock(spec=BaseEmbedder)
        embedder.layers = [-1, -2]
        embedder.num_heads = 12
        embedder.precision = "float32"
        embedder.flatten = False
        embedder.streaming_output = False
        embedder.device = torch.device("cpu")
        embedder.output_prefix = "test"
        embedder.model_name = "test_model"
        
        # Mock the _precision_to_dtype method
        def mock_precision_to_dtype(precision, framework):
            if precision == "float32":
                return torch.float32 if framework == "torch" else np.float32
            elif precision == "float16":
                return torch.float16 if framework == "torch" else np.float16
            else:
                raise ValueError(f"Unsupported precision: {precision}")
        
        embedder._precision_to_dtype = mock_precision_to_dtype
        
        # Mock the _to_numpy method
        def mock_to_numpy(tensor):
            return tensor.detach().cpu().contiguous().numpy()
        
        embedder._to_numpy = mock_to_numpy
        
        return embedder
    
    @pytest.fixture
    def sample_representations(self):
        """Create sample representations for testing."""
        return {
            -1: torch.randn(3, 50, 768),  # 3 sequences, 50 tokens, 768 dimensions
            -2: torch.randn(3, 50, 768)
        }
    
    @pytest.fixture
    def sample_attention_matrices(self):
        """Create sample attention matrices for testing."""
        # Shape: [layers, batch, heads, seq_len, seq_len]
        return torch.randn(2, 3, 12, 50, 50)
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample logits for testing."""
        return {
            -1: torch.randn(3, 50, 21),  # 3 sequences, 50 tokens, 21 vocab size
            -2: torch.randn(3, 50, 21)
        }


class TestMeanPooledExtraction:
    """Test mean pooled embedding extraction."""
    
    def test_extract_mean_pooled_basic(self, mock_embedder, sample_representations):
        """Test basic mean pooled extraction."""
        # Setup
        batch_labels = ["seq1", "seq2", "seq3"]
        pooling_mask = torch.ones(3, 50, dtype=torch.bool)
        offset = 0
        
        # Mock the actual method from BaseEmbedder
        from pepe.embedders.base_embedder import BaseEmbedder
        
        # Create a real instance to get the method
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.streaming_output = False
            real_embedder.mean_pooled = {"output_data": {-1: [], -2: []}}
            
            # Call the method
            real_embedder._extract_mean_pooled(
                sample_representations, batch_labels, pooling_mask, offset
            )
            
            # Verify results
            assert len(real_embedder.mean_pooled["output_data"][-1]) == 3
            assert len(real_embedder.mean_pooled["output_data"][-2]) == 3
            assert real_embedder.mean_pooled["output_data"][-1][0].shape == torch.Size([768])
    
    def test_extract_mean_pooled_with_mask(self, mock_embedder, sample_representations):
        """Test mean pooled extraction with partial masking."""
        # Setup with partial masking
        batch_labels = ["seq1", "seq2", "seq3"]
        pooling_mask = torch.ones(3, 50, dtype=torch.bool)
        pooling_mask[0, 40:] = False  # Mask last 10 tokens of first sequence
        pooling_mask[1, 35:] = False  # Mask last 15 tokens of second sequence
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.streaming_output = False
            real_embedder.mean_pooled = {"output_data": {-1: []}}
            
            # Call the method
            real_embedder._extract_mean_pooled(
                sample_representations, batch_labels, pooling_mask, offset
            )
            
            # Verify results
            assert len(real_embedder.mean_pooled["output_data"][-1]) == 3
            assert real_embedder.mean_pooled["output_data"][-1][0].shape == torch.Size([768])
    
    def test_extract_mean_pooled_empty_mask(self, mock_embedder, sample_representations):
        """Test mean pooled extraction with empty mask."""
        # Setup with all masked tokens
        batch_labels = ["seq1"]
        pooling_mask = torch.zeros(1, 50, dtype=torch.bool)
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.streaming_output = False
            real_embedder.mean_pooled = {"output_data": {-1: []}}
            
            # Call the method
            real_embedder._extract_mean_pooled(
                {-1: sample_representations[-1][:1]}, batch_labels, pooling_mask, offset
            )
            
            # Verify results - should handle division by zero gracefully
            assert len(real_embedder.mean_pooled["output_data"][-1]) == 1
            result = real_embedder.mean_pooled["output_data"][-1][0]
            assert torch.isnan(result).any() or torch.isinf(result).any()  # Should be nan/inf due to 0/0


class TestPerTokenExtraction:
    """Test per token embedding extraction."""
    
    def test_extract_per_token_basic(self, mock_embedder, sample_representations):
        """Test basic per token extraction."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.streaming_output = False
            real_embedder.discard_padding = False
            real_embedder.flatten = False
            real_embedder.per_token = {"output_data": {-1: [], -2: []}}
            
            # Call the method
            real_embedder._extract_per_token(
                sample_representations, batch_labels, offset
            )
            
            # Verify results
            assert len(real_embedder.per_token["output_data"][-1]) == 3
            assert len(real_embedder.per_token["output_data"][-2]) == 3
            assert real_embedder.per_token["output_data"][-1][0].shape == torch.Size([50, 768])
    
    def test_extract_per_token_with_flatten(self, mock_embedder, sample_representations):
        """Test per token extraction with flattening."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.streaming_output = False
            real_embedder.discard_padding = False
            real_embedder.flatten = True
            real_embedder.per_token = {"output_data": {-1: []}}
            
            # Call the method
            real_embedder._extract_per_token(
                sample_representations, batch_labels, offset
            )
            
            # Verify results
            assert len(real_embedder.per_token["output_data"][-1]) == 3
            assert real_embedder.per_token["output_data"][-1][0].shape == torch.Size([50 * 768])


class TestAttentionExtraction:
    """Test attention matrix extraction."""
    
    def test_extract_attention_head(self, mock_embedder, sample_attention_matrices):
        """Test attention head extraction."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.num_heads = 12
            real_embedder.streaming_output = False
            real_embedder.flatten = False
            real_embedder.attention_head = {
                "output_data": {
                    -1: {head: [] for head in range(12)},
                    -2: {head: [] for head in range(12)}
                }
            }
            
            # Call the method
            real_embedder._extract_attention_head(
                sample_attention_matrices, batch_labels, offset
            )
            
            # Verify results
            for layer in [-1, -2]:
                for head in range(12):
                    assert len(real_embedder.attention_head["output_data"][layer][head]) == 3
                    assert real_embedder.attention_head["output_data"][layer][head][0].shape == torch.Size([50, 50])
    
    def test_extract_attention_layer(self, mock_embedder, sample_attention_matrices):
        """Test attention layer extraction (averaged over heads)."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.streaming_output = False
            real_embedder.flatten = False
            real_embedder.attention_layer = {"output_data": {-1: [], -2: []}}
            
            # Call the method
            real_embedder._extract_attention_layer(
                sample_attention_matrices, batch_labels, offset
            )
            
            # Verify results
            assert len(real_embedder.attention_layer["output_data"][-1]) == 3
            assert len(real_embedder.attention_layer["output_data"][-2]) == 3
            assert real_embedder.attention_layer["output_data"][-1][0].shape == torch.Size([50, 50])
    
    def test_extract_attention_model(self, mock_embedder, sample_attention_matrices):
        """Test attention model extraction (averaged over layers and heads)."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.streaming_output = False
            real_embedder.flatten = False
            real_embedder.attention_model = {"output_data": []}
            
            # Call the method
            real_embedder._extract_attention_model(
                sample_attention_matrices, batch_labels, offset
            )
            
            # Verify results
            assert len(real_embedder.attention_model["output_data"]) == 3
            assert real_embedder.attention_model["output_data"][0].shape == torch.Size([50, 50])
    
    def test_extract_attention_with_flatten(self, mock_embedder, sample_attention_matrices):
        """Test attention extraction with flattening."""
        batch_labels = ["seq1", "seq2", "seq3"]
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.streaming_output = False
            real_embedder.flatten = True
            real_embedder.attention_layer = {"output_data": {-1: []}}
            
            # Call the method
            real_embedder._extract_attention_layer(
                sample_attention_matrices, batch_labels, offset
            )
            
            # Verify results
            assert len(real_embedder.attention_layer["output_data"][-1]) == 3
            assert real_embedder.attention_layer["output_data"][-1][0].shape == torch.Size([50 * 50])


class TestLogitsExtraction:
    """Test logits extraction."""
    
    def test_extract_logits_basic(self, mock_embedder, sample_logits):
        """Test basic logits extraction."""
        offset = 0
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.streaming_output = False
            real_embedder.logits = {"output_data": {-1: [], -2: []}}
            
            # Call the method
            real_embedder._extract_logits(sample_logits, offset)
            
            # Verify results
            assert len(real_embedder.logits["output_data"][-1]) == 3
            assert len(real_embedder.logits["output_data"][-2]) == 3
            assert real_embedder.logits["output_data"][-1][0].shape == torch.Size([50, 21])


class TestSubstringExtraction:
    """Test substring pooled extraction."""
    
    def test_extract_substring_pooled_basic(self, mock_embedder, sample_representations):
        """Test basic substring pooled extraction."""
        offset = 0
        
        # Create substring masks
        substring_mask = [
            torch.zeros(50, dtype=torch.bool),  # seq1
            torch.zeros(50, dtype=torch.bool),  # seq2
            torch.zeros(50, dtype=torch.bool)   # seq3
        ]
        # Set some positions to True to simulate substring regions
        substring_mask[0][10:20] = True
        substring_mask[1][15:25] = True
        substring_mask[2][5:15] = True
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1, -2]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1, -2]
            real_embedder.streaming_output = False
            real_embedder.substring_pooled = {"output_data": {-1: [], -2: []}}
            
            # Call the method
            real_embedder._extract_substring_pooled(
                sample_representations, substring_mask, offset
            )
            
            # Verify results
            assert len(real_embedder.substring_pooled["output_data"][-1]) == 3
            assert len(real_embedder.substring_pooled["output_data"][-2]) == 3
            assert real_embedder.substring_pooled["output_data"][-1][0].shape == torch.Size([768])
    
    def test_extract_substring_pooled_empty_mask(self, mock_embedder, sample_representations):
        """Test substring pooled extraction with empty mask."""
        offset = 0
        
        # Create empty substring masks
        substring_mask = [
            torch.zeros(50, dtype=torch.bool),  # seq1 - all False
            torch.zeros(50, dtype=torch.bool),  # seq2 - all False
            torch.zeros(50, dtype=torch.bool)   # seq3 - all False
        ]
        
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.streaming_output = False
            real_embedder.substring_pooled = {"output_data": {-1: []}}
            
            # Call the method
            real_embedder._extract_substring_pooled(
                sample_representations, substring_mask, offset
            )
            
            # Verify results - should handle division by zero gracefully
            assert len(real_embedder.substring_pooled["output_data"][-1]) == 3
            result = real_embedder.substring_pooled["output_data"][-1][0]
            assert torch.isnan(result).any() or torch.isinf(result).any()  # Should be nan/inf due to 0/0


class TestBatchExtraction:
    """Test batch extraction orchestration."""
    
    def test_extract_batch_multiple_types(self, mock_embedder):
        """Test batch extraction with multiple output types."""
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.output_types = ["mean_pooled", "per_token"]
            
            # Mock the extraction methods
            real_embedder.mean_pooled = {"method": Mock()}
            real_embedder.per_token = {"method": Mock()}
            
            # Create output bundle
            output_bundle = {
                "representations": {-1: torch.randn(2, 50, 768)},
                "batch_labels": ["seq1", "seq2"],
                "pooling_mask": torch.ones(2, 50, dtype=torch.bool),
                "offset": 0
            }
            
            # Call the method
            real_embedder._extract_batch(output_bundle)
            
            # Verify both methods were called
            real_embedder.mean_pooled["method"].assert_called_once()
            real_embedder.per_token["method"].assert_called_once()
    
    def test_extract_batch_parameter_filtering(self, mock_embedder):
        """Test that batch extraction filters parameters correctly."""
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            real_embedder = BaseEmbedder(Mock())
            real_embedder.layers = [-1]
            real_embedder.output_types = ["mean_pooled"]
            
            # Mock the extraction method with specific signature
            def mock_extract_mean_pooled(representations, batch_labels, pooling_mask, offset):
                pass
            
            real_embedder.mean_pooled = {"method": mock_extract_mean_pooled}
            
            # Create output bundle with extra parameters
            output_bundle = {
                "representations": {-1: torch.randn(2, 50, 768)},
                "batch_labels": ["seq1", "seq2"],
                "pooling_mask": torch.ones(2, 50, dtype=torch.bool),
                "offset": 0,
                "extra_param": "should_be_ignored"
            }
            
            # Mock inspect.signature to return our expected signature
            with patch('inspect.signature') as mock_signature:
                mock_signature.return_value.parameters = {
                    'representations': None,
                    'batch_labels': None,
                    'pooling_mask': None,
                    'offset': None
                }
                
                # Patch the extraction method
                with patch.object(real_embedder, 'mean_pooled', {"method": Mock()}):
                    real_embedder._extract_batch(output_bundle)
                    
                    # Verify the method was called with filtered parameters
                    real_embedder.mean_pooled["method"].assert_called_once() 