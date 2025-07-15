"""
Unit tests for BaseEmbedder class.

This module contains comprehensive tests for the BaseEmbedder class,
testing all its methods, configuration handling, and core functionality.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pepe.embedders.base_embedder import BaseEmbedder
from pepe.parse_arguments import parse_arguments


# Global fixtures for all test classes
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_fasta_path(temp_dir):
    """Create a sample FASTA file for testing."""
    fasta_path = os.path.join(temp_dir, "test.fasta")
    with open(fasta_path, "w") as f:
        f.write(">seq1\nACDEFGHIKLMNPQRSTVWY\n")
        f.write(">seq2\nMKVLSFGHIKLMNPQRSTVWY\n")
    return fasta_path

@pytest.fixture
def sample_substring_csv(temp_dir):
    """Create a sample substring CSV file for testing."""
    csv_path = os.path.join(temp_dir, "test_substring.csv")
    with open(csv_path, "w") as f:
        f.write("sequence_id,substring_aa\n")
        f.write("seq1,DEFGH\n")
        f.write("seq2,VLSFG\n")
    return csv_path

@pytest.fixture
def mock_args(temp_dir, sample_fasta_path, sample_substring_csv):
    """Create mock arguments for BaseEmbedder initialization."""
    args = Mock()
    args.fasta_path = sample_fasta_path
    args.model_name = "test_model"
    args.disable_special_tokens = False
    args.experiment_name = "test_experiment"
    args.output_path = temp_dir
    args.substring_path = sample_substring_csv
    args.context = 0
    args.layers = [[-1]]
    args.batch_size = 2
    args.max_length = 50
    args.device = "cpu"
    args.extract_embeddings = ["mean_pooled", "per_token"]
    args.discard_padding = False
    args.flatten = False
    args.streaming_output = False
    args.num_workers = 1
    args.flush_batches_after = 128
    args.precision = "float32"
    return args

@pytest.fixture
def mock_base_embedder(mock_args):
    """Create a mock BaseEmbedder instance."""
    # Mock the abstract methods
    with patch.multiple(
        BaseEmbedder,
        _load_data=Mock(return_value=(Mock(), 50)),
        _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
        _load_layers=Mock(return_value=[-1]),
        _compute_outputs=Mock(return_value=(None, {-1: torch.randn(2, 50, 768)}, None)),
        __abstractmethods__=set()
    ):
        embedder = BaseEmbedder(mock_args)
        # Set additional attributes after initialization
        embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY", "seq2": "MKVLSFGHIKLMNPQRSTVWY"}
        embedder.num_sequences = 2
        embedder.num_heads = 12
        embedder.num_layers = 12
        embedder.embedding_size = 768
        embedder.special_tokens = torch.tensor([0, 1, 2])
        embedder.data_loader = [(["seq1", "seq2"], ["str1", "str2"], torch.randint(0, 100, (2, 50)), None, [torch.ones(50, dtype=torch.bool), torch.ones(50, dtype=torch.bool)])]
        return embedder


class TestBaseEmbedder:
    """Test suite for BaseEmbedder class."""


class TestBaseEmbedderInitialization:
    """Test BaseEmbedder initialization and configuration."""
    
    def test_init_creates_output_directory(self, mock_args, temp_dir):
        """Test that initialization creates output directory."""
        output_path = os.path.join(temp_dir, "test_output")
        mock_args.output_path = output_path
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert os.path.exists(embedder.output_path)
    
    def test_init_model_name_from_path(self, mock_args):
        """Test model name extraction from file path."""
        mock_args.model_name = "/path/to/model.pt"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.model_name == "model"
    
    def test_init_model_name_from_url(self, mock_args):
        """Test model name extraction from URL."""
        mock_args.model_name = "username/model-name"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.model_name == "model-name"
    
    def test_init_device_selection_cpu(self, mock_args):
        """Test device selection for CPU."""
        mock_args.device = "cpu"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.device == torch.device("cpu")
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_init_device_selection_cuda(self, mock_cuda, mock_args):
        """Test device selection for CUDA."""
        mock_args.device = "cuda"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.device == torch.device("cuda")
    
    def test_init_output_prefix_from_experiment_name(self, mock_args):
        """Test output prefix is set from experiment name."""
        mock_args.experiment_name = "test_experiment"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.output_prefix == "test_experiment"
    
    def test_init_output_prefix_from_fasta_filename(self, mock_args):
        """Test output prefix is set from FASTA filename when no experiment name."""
        mock_args.experiment_name = None
        mock_args.fasta_path = "/path/to/sequences.fasta"
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(mock_args)
            assert embedder.output_prefix == "sequences"


class TestBaseEmbedderUtilityMethods:
    """Test BaseEmbedder utility methods."""
    
    def test_precision_to_dtype_torch_float16(self, mock_base_embedder):
        """Test precision conversion to torch float16."""
        dtype = mock_base_embedder._precision_to_dtype("float16", "torch")
        assert dtype == torch.float16
    
    def test_precision_to_dtype_torch_float32(self, mock_base_embedder):
        """Test precision conversion to torch float32."""
        dtype = mock_base_embedder._precision_to_dtype("float32", "torch")
        assert dtype == torch.float32
    
    def test_precision_to_dtype_numpy_float16(self, mock_base_embedder):
        """Test precision conversion to numpy float16."""
        dtype = mock_base_embedder._precision_to_dtype("float16", "numpy")
        assert dtype == np.float16
    
    def test_precision_to_dtype_numpy_float32(self, mock_base_embedder):
        """Test precision conversion to numpy float32."""
        dtype = mock_base_embedder._precision_to_dtype("float32", "numpy")
        assert dtype == np.float32
    
    def test_precision_to_dtype_invalid_precision(self, mock_base_embedder):
        """Test precision conversion with invalid precision."""
        with pytest.raises(ValueError, match="Unsupported precision"):
            mock_base_embedder._precision_to_dtype("invalid", "torch")
    
    def test_load_substrings_with_file(self, mock_base_embedder, sample_substring_csv):
        """Test loading substrings from CSV file."""
        substring_dict = mock_base_embedder._load_substrings(sample_substring_csv)
        expected = {"seq1": "DEFGH", "seq2": "VLSFG"}
        assert substring_dict == expected
    
    def test_load_substrings_without_file(self, mock_base_embedder):
        """Test loading substrings without file."""
        substring_dict = mock_base_embedder._load_substrings(None)
        assert substring_dict is None
    
    def test_make_output_filepath_basic(self, mock_base_embedder):
        """Test basic output filepath generation."""
        filepath = mock_base_embedder._make_output_filepath("embeddings", "/output", layer=-1)
        expected = "/output/test_experiment_test_model_embeddings_layer_-1.npy"
        assert filepath == expected
    
    def test_make_output_filepath_with_head(self, mock_base_embedder):
        """Test output filepath generation with head."""
        filepath = mock_base_embedder._make_output_filepath("attention", "/output", layer=-1, head=5)
        expected = "/output/test_experiment_test_model_attention_layer_-1_head_6.npy"
        assert filepath == expected
    
    def test_mask_special_tokens_with_tokens(self, mock_base_embedder):
        """Test masking special tokens with provided tokens."""
        input_tensor = torch.tensor([[0, 1, 5, 6, 2], [0, 3, 4, 5, 2]])
        special_tokens = torch.tensor([0, 1, 2])
        
        mask = mock_base_embedder._mask_special_tokens(input_tensor, special_tokens)
        expected = torch.tensor([[False, False, True, True, False], [False, True, True, True, False]])
        assert torch.equal(mask, expected)
    
    def test_mask_special_tokens_without_tokens(self, mock_base_embedder):
        """Test masking special tokens without provided tokens."""
        input_tensor = torch.tensor([[0, 1, 5, 6, 2], [0, 3, 4, 5, 2]])
        
        mask = mock_base_embedder._mask_special_tokens(input_tensor)
        expected = torch.tensor([[False, False, True, True, False], [False, True, True, True, False]])
        assert torch.equal(mask, expected)
    
    def test_to_numpy_conversion(self, mock_base_embedder):
        """Test tensor to numpy conversion."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        numpy_array = mock_base_embedder._to_numpy(tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (2, 2)
        assert np.array_equal(numpy_array, np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestBaseEmbedderOutputManagement:
    """Test BaseEmbedder output management methods."""
    
    def test_create_output_dirs(self, mock_base_embedder, temp_dir):
        """Test creation of output directories."""
        mock_base_embedder.output_path = temp_dir
        mock_base_embedder.output_types = ["embeddings", "attention"]
        
        mock_base_embedder._create_output_dirs()
        
        assert os.path.exists(os.path.join(temp_dir, "embeddings"))
        assert os.path.exists(os.path.join(temp_dir, "attention"))
    
    def test_export_sequence_indices(self, mock_base_embedder, temp_dir):
        """Test exporting sequence indices to CSV."""
        mock_base_embedder.output_path = temp_dir
        mock_base_embedder.fasta_path = "test.fasta"
        mock_base_embedder.sequence_labels = ["seq1", "seq2", "seq3"]
        
        mock_base_embedder.export_sequence_indices()
        
        idx_file = os.path.join(temp_dir, "test_idx.csv")
        assert os.path.exists(idx_file)
        
        with open(idx_file, "r") as f:
            content = f.read()
            assert "index,sequence_id" in content
            assert "0,seq1" in content
            assert "1,seq2" in content
            assert "2,seq3" in content


class TestBaseEmbedderEmbeddingExtraction:
    """Test BaseEmbedder embedding extraction methods."""
    
    def test_extract_mean_pooled_no_streaming(self, mock_base_embedder):
        """Test mean pooled embedding extraction without streaming."""
        mock_base_embedder.streaming_output = False
        mock_base_embedder.layers = [-1]
        
        # Mock data
        representations = {-1: torch.randn(2, 50, 768)}
        batch_labels = ["seq1", "seq2"]
        pooling_mask = torch.ones(2, 50, dtype=torch.bool)
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.mean_pooled = {"output_data": {-1: []}}
        
        mock_base_embedder._extract_mean_pooled(representations, batch_labels, pooling_mask, offset)
        
        assert len(mock_base_embedder.mean_pooled["output_data"][-1]) == 2
        assert mock_base_embedder.mean_pooled["output_data"][-1][0].shape == torch.Size([768])
    
    def test_extract_per_token_no_streaming(self, mock_base_embedder):
        """Test per token embedding extraction without streaming."""
        mock_base_embedder.streaming_output = False
        mock_base_embedder.layers = [-1]
        mock_base_embedder.discard_padding = False
        
        # Mock data
        representations = {-1: torch.randn(2, 50, 768)}
        batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.per_token = {"output_data": {-1: []}}
        
        mock_base_embedder._extract_per_token(representations, batch_labels, offset)
        
        assert len(mock_base_embedder.per_token["output_data"][-1]) == 2
        assert mock_base_embedder.per_token["output_data"][-1][0].shape == torch.Size([50, 768])
    
    def test_extract_per_token_with_flatten(self, mock_base_embedder):
        """Test per token embedding extraction with flattening."""
        mock_base_embedder.streaming_output = False
        mock_base_embedder.layers = [-1]
        mock_base_embedder.discard_padding = False
        mock_base_embedder.flatten = True
        
        # Mock data
        representations = {-1: torch.randn(2, 50, 768)}
        batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.per_token = {"output_data": {-1: []}}
        
        mock_base_embedder._extract_per_token(representations, batch_labels, offset)
        
        assert len(mock_base_embedder.per_token["output_data"][-1]) == 2
        assert mock_base_embedder.per_token["output_data"][-1][0].shape == torch.Size([50 * 768])
    
    def test_extract_attention_head_no_streaming(self, mock_base_embedder):
        """Test attention head extraction without streaming."""
        mock_base_embedder.streaming_output = False
        mock_base_embedder.layers = [-1]
        mock_base_embedder.num_heads = 12
        
        # Mock data - attention_matrices[layer-1, batch, head, seq, seq]
        attention_matrices = torch.randn(1, 2, 12, 50, 50)
        batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.attention_head = {"output_data": {-1: {head: [] for head in range(12)}}}
        
        mock_base_embedder._extract_attention_head(attention_matrices, batch_labels, offset)
        
        for head in range(12):
            assert len(mock_base_embedder.attention_head["output_data"][-1][head]) == 2
            assert mock_base_embedder.attention_head["output_data"][-1][head][0].shape == torch.Size([50, 50])
    
    def test_extract_attention_layer_no_streaming(self, mock_base_embedder):
        """Test attention layer extraction without streaming."""
        mock_base_embedder.streaming_output = False
        mock_base_embedder.layers = [-1]
        
        # Mock data - attention_matrices[layer-1, batch, head, seq, seq]
        attention_matrices = torch.randn(1, 2, 12, 50, 50)
        batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.attention_layer = {"output_data": {-1: []}}
        
        mock_base_embedder._extract_attention_layer(attention_matrices, batch_labels, offset)
        
        assert len(mock_base_embedder.attention_layer["output_data"][-1]) == 2
        assert mock_base_embedder.attention_layer["output_data"][-1][0].shape == torch.Size([50, 50])
    
    def test_extract_attention_model_no_streaming(self, mock_base_embedder):
        """Test attention model extraction without streaming."""
        mock_base_embedder.streaming_output = False
        
        # Mock data - attention_matrices[layer, batch, head, seq, seq]
        attention_matrices = torch.randn(12, 2, 12, 50, 50)
        batch_labels = ["seq1", "seq2"]
        offset = 0
        
        # Initialize output objects
        mock_base_embedder.attention_model = {"output_data": []}
        
        mock_base_embedder._extract_attention_model(attention_matrices, batch_labels, offset)
        
        assert len(mock_base_embedder.attention_model["output_data"]) == 2
        assert mock_base_embedder.attention_model["output_data"][0].shape == torch.Size([50, 50])


class TestBaseEmbedderErrorHandling:
    """Test BaseEmbedder error handling."""
    
    def test_safe_compute_oom_handling(self, mock_base_embedder):
        """Test safe compute with OOM error handling."""
        # Mock OOM error on first call, success on subsequent calls
        def mock_compute_outputs(model, toks, attention_mask, return_embeddings, return_contacts, return_logits):
            if toks.size(0) > 1:  # If batch size > 1, raise OOM
                raise torch.OutOfMemoryError("CUDA out of memory")
            else:
                return None, {-1: torch.randn(1, 50, 768)}, None
        
        mock_base_embedder._compute_outputs = mock_compute_outputs
        mock_base_embedder.return_embeddings = True
        mock_base_embedder.return_contacts = False
        mock_base_embedder.return_logits = False
        
        with patch('torch.cuda.empty_cache'):
            toks = torch.randint(0, 100, (2, 50))
            attention_mask = None
            
            result = mock_base_embedder._safe_compute(toks, attention_mask)
            
            assert result is not None
            logits, representations, attention_matrices = result
            assert representations is not None
            assert representations[-1].shape[0] == 2  # Should have 2 sequences after concatenation
    
    def test_safe_compute_oom_single_sample(self, mock_base_embedder):
        """Test safe compute with OOM on single sample."""
        def mock_compute_outputs(model, toks, attention_mask, return_embeddings, return_contacts, return_logits):
            raise torch.OutOfMemoryError("CUDA out of memory")
        
        mock_base_embedder._compute_outputs = mock_compute_outputs
        mock_base_embedder.return_embeddings = True
        mock_base_embedder.return_contacts = False
        mock_base_embedder.return_logits = False
        
        with patch('torch.cuda.empty_cache'):
            toks = torch.randint(0, 100, (1, 50))
            attention_mask = None
            
            with pytest.raises(torch.OutOfMemoryError):
                mock_base_embedder._safe_compute(toks, attention_mask)


class TestBaseEmbedderConfigurationEdgeCases:
    """Test BaseEmbedder configuration edge cases."""
    
    def test_get_output_types_multiple_same_types(self, mock_base_embedder):
        """Test get_output_types with duplicate types."""
        mock_args = Mock()
        mock_args.extract_embeddings = ["mean_pooled", "per_token", "mean_pooled"]
        
        output_types = mock_base_embedder._get_output_types(mock_args)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
        assert output_types.count("mean_pooled") == 1  # Should only appear once
    
    def test_get_output_types_all_types(self, mock_base_embedder):
        """Test get_output_types with all embedding types."""
        mock_args = Mock()
        mock_args.extract_embeddings = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        
        output_types = mock_base_embedder._get_output_types(mock_args)
        
        expected_types = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        
        for expected_type in expected_types:
            assert expected_type in output_types
    
    def test_get_substring_positions_basic(self, mock_base_embedder):
        """Test getting substring positions."""
        mock_base_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_base_embedder.substring_dict = {"seq1": "DEFGH"}
        
        start, end = mock_base_embedder.get_substring_positions("seq1", special_tokens=True, context=0)
        
        assert start == 3  # Position of "DEFGH" in sequence + 1 for special token
        assert end == 8   # End position + 1 for special token
    
    def test_get_substring_positions_with_context(self, mock_base_embedder):
        """Test getting substring positions with context."""
        mock_base_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_base_embedder.substring_dict = {"seq1": "DEFGH"}
        
        start, end = mock_base_embedder.get_substring_positions("seq1", special_tokens=True, context=2)
        
        assert start == 2  # Position - context + special token
        assert end == 9   # End position + context + special token
    
    def test_get_substring_positions_no_special_tokens(self, mock_base_embedder):
        """Test getting substring positions without special tokens."""
        mock_base_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_base_embedder.substring_dict = {"seq1": "DEFGH"}
        
        start, end = mock_base_embedder.get_substring_positions("seq1", special_tokens=False, context=0)
        
        assert start == 2  # Position of "DEFGH" in sequence
        assert end == 7   # End position 