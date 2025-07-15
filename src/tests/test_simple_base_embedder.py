"""
Simplified unit tests for BaseEmbedder class.

This module contains basic tests for the BaseEmbedder class methods
that can be tested without complex mocking.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from pepe.embedders.base_embedder import BaseEmbedder


class TestBaseEmbedderUtilityMethods:
    """Test BaseEmbedder utility methods."""
    
    def test_precision_to_dtype_torch_float16(self):
        """Test precision conversion to torch float16."""
        # Create a mock embedder just for testing the method
        mock_embedder = Mock()
        mock_embedder._precision_to_dtype = BaseEmbedder._precision_to_dtype.__get__(mock_embedder, BaseEmbedder)
        
        dtype = mock_embedder._precision_to_dtype("float16", "torch")
        assert dtype == torch.float16
    
    def test_precision_to_dtype_torch_float32(self):
        """Test precision conversion to torch float32."""
        mock_embedder = Mock()
        mock_embedder._precision_to_dtype = BaseEmbedder._precision_to_dtype.__get__(mock_embedder, BaseEmbedder)
        
        dtype = mock_embedder._precision_to_dtype("float32", "torch")
        assert dtype == torch.float32
    
    def test_precision_to_dtype_numpy_float16(self):
        """Test precision conversion to numpy float16."""
        mock_embedder = Mock()
        mock_embedder._precision_to_dtype = BaseEmbedder._precision_to_dtype.__get__(mock_embedder, BaseEmbedder)
        
        dtype = mock_embedder._precision_to_dtype("float16", "numpy")
        assert dtype == np.float16
    
    def test_precision_to_dtype_numpy_float32(self):
        """Test precision conversion to numpy float32."""
        mock_embedder = Mock()
        mock_embedder._precision_to_dtype = BaseEmbedder._precision_to_dtype.__get__(mock_embedder, BaseEmbedder)
        
        dtype = mock_embedder._precision_to_dtype("float32", "numpy")
        assert dtype == np.float32
    
    def test_precision_to_dtype_invalid_precision(self):
        """Test precision conversion with invalid precision."""
        mock_embedder = Mock()
        mock_embedder._precision_to_dtype = BaseEmbedder._precision_to_dtype.__get__(mock_embedder, BaseEmbedder)
        
        with pytest.raises(ValueError, match="Unsupported precision"):
            mock_embedder._precision_to_dtype("invalid", "torch")
    
    def test_load_substrings_with_file(self):
        """Test loading substrings from CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("sequence_id,substring_aa\n")
            f.write("seq1,DEFGH\n")
            f.write("seq2,VLSFG\n")
            csv_path = f.name
        
        try:
            # Test the method
            mock_embedder = Mock()
            mock_embedder._load_substrings = BaseEmbedder._load_substrings.__get__(mock_embedder, BaseEmbedder)
            
            substring_dict = mock_embedder._load_substrings(csv_path)
            expected = {"seq1": "DEFGH", "seq2": "VLSFG"}
            assert substring_dict == expected
        finally:
            os.unlink(csv_path)
    
    def test_load_substrings_without_file(self):
        """Test loading substrings without file."""
        mock_embedder = Mock()
        mock_embedder._load_substrings = BaseEmbedder._load_substrings.__get__(mock_embedder, BaseEmbedder)
        
        substring_dict = mock_embedder._load_substrings(None)
        assert substring_dict is None
    
    def test_mask_special_tokens_with_tokens(self):
        """Test masking special tokens with provided tokens."""
        mock_embedder = Mock()
        mock_embedder._mask_special_tokens = BaseEmbedder._mask_special_tokens.__get__(mock_embedder, BaseEmbedder)
        
        input_tensor = torch.tensor([[0, 1, 5, 6, 2], [0, 3, 4, 5, 2]])
        special_tokens = torch.tensor([0, 1, 2])
        
        mask = mock_embedder._mask_special_tokens(input_tensor, special_tokens)
        expected = torch.tensor([[False, False, True, True, False], [False, True, True, True, False]])
        assert torch.equal(mask, expected)
    
    def test_mask_special_tokens_without_tokens(self):
        """Test masking special tokens without provided tokens."""
        mock_embedder = Mock()
        mock_embedder._mask_special_tokens = BaseEmbedder._mask_special_tokens.__get__(mock_embedder, BaseEmbedder)
        
        input_tensor = torch.tensor([[0, 1, 5, 6, 2], [0, 3, 4, 5, 2]])
        
        mask = mock_embedder._mask_special_tokens(input_tensor)
        expected = torch.tensor([[False, False, True, True, False], [False, True, True, True, False]])
        assert torch.equal(mask, expected)
    
    def test_to_numpy_conversion(self):
        """Test tensor to numpy conversion."""
        mock_embedder = Mock()
        mock_embedder._to_numpy = BaseEmbedder._to_numpy.__get__(mock_embedder, BaseEmbedder)
        
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        numpy_array = mock_embedder._to_numpy(tensor)
        
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.shape == (2, 2)
        assert np.array_equal(numpy_array, np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    def test_make_output_filepath_basic(self):
        """Test basic output filepath generation."""
        mock_embedder = Mock()
        mock_embedder.output_prefix = "test_experiment"
        mock_embedder.model_name = "test_model"
        mock_embedder._make_output_filepath = BaseEmbedder._make_output_filepath.__get__(mock_embedder, BaseEmbedder)
        
        filepath = mock_embedder._make_output_filepath("embeddings", "/output", layer=-1)
        expected = "/output/test_experiment_test_model_embeddings_layer_-1.npy"
        assert filepath == expected
    
    def test_make_output_filepath_with_head(self):
        """Test output filepath generation with head."""
        mock_embedder = Mock()
        mock_embedder.output_prefix = "test_experiment"
        mock_embedder.model_name = "test_model"
        mock_embedder._make_output_filepath = BaseEmbedder._make_output_filepath.__get__(mock_embedder, BaseEmbedder)
        
        filepath = mock_embedder._make_output_filepath("attention", "/output", layer=-1, head=5)
        expected = "/output/test_experiment_test_model_attention_layer_-1_head_6.npy"
        assert filepath == expected
    
    def test_get_output_types_basic(self):
        """Test get_output_types method."""
        mock_embedder = Mock()
        mock_embedder._get_output_types = BaseEmbedder._get_output_types.__get__(mock_embedder, BaseEmbedder)
        
        mock_args = Mock()
        mock_args.extract_embeddings = ["mean_pooled", "per_token"]
        
        output_types = mock_embedder._get_output_types(mock_args)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
    
    def test_get_output_types_with_duplicates(self):
        """Test get_output_types with duplicate types."""
        mock_embedder = Mock()
        mock_embedder._get_output_types = BaseEmbedder._get_output_types.__get__(mock_embedder, BaseEmbedder)
        
        mock_args = Mock()
        mock_args.extract_embeddings = ["mean_pooled", "per_token", "mean_pooled"]
        
        output_types = mock_embedder._get_output_types(mock_args)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
        assert output_types.count("mean_pooled") == 1  # Should only appear once
    
    def test_get_output_types_all_types(self):
        """Test get_output_types with all embedding types."""
        mock_embedder = Mock()
        mock_embedder._get_output_types = BaseEmbedder._get_output_types.__get__(mock_embedder, BaseEmbedder)
        
        mock_args = Mock()
        mock_args.extract_embeddings = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        
        output_types = mock_embedder._get_output_types(mock_args)
        
        expected_types = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        
        for expected_type in expected_types:
            assert expected_type in output_types


class TestBaseEmbedderFileOperations:
    """Test BaseEmbedder file operations."""
    
    def test_create_output_dirs(self):
        """Test creation of output directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_embedder = Mock()
            mock_embedder.output_path = temp_dir
            mock_embedder.output_types = ["embeddings", "attention"]
            mock_embedder._create_output_dirs = BaseEmbedder._create_output_dirs.__get__(mock_embedder, BaseEmbedder)
            
            mock_embedder._create_output_dirs()
            
            assert os.path.exists(os.path.join(temp_dir, "embeddings"))
            assert os.path.exists(os.path.join(temp_dir, "attention"))
    
    def test_export_sequence_indices(self):
        """Test exporting sequence indices to CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_embedder = Mock()
            mock_embedder.output_path = temp_dir
            mock_embedder.fasta_path = "test.fasta"
            mock_embedder.sequence_labels = ["seq1", "seq2", "seq3"]
            mock_embedder.export_sequence_indices = BaseEmbedder.export_sequence_indices.__get__(mock_embedder, BaseEmbedder)
            
            mock_embedder.export_sequence_indices()
            
            idx_file = os.path.join(temp_dir, "test_idx.csv")
            assert os.path.exists(idx_file)
            
            with open(idx_file, "r") as f:
                content = f.read()
                assert "index,sequence_id" in content
                assert "0,seq1" in content
                assert "1,seq2" in content
                assert "2,seq3" in content


class TestBaseEmbedderConfigurationMethods:
    """Test BaseEmbedder configuration methods."""
    
    def test_get_substring_positions_basic(self):
        """Test getting substring positions."""
        mock_embedder = Mock()
        mock_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_embedder.substring_dict = {"seq1": "DEFGH"}
        mock_embedder.get_substring_positions = BaseEmbedder.get_substring_positions.__get__(mock_embedder, BaseEmbedder)
        
        start, end = mock_embedder.get_substring_positions("seq1", special_tokens=True, context=0)
        
        assert start == 3  # Position of "DEFGH" in sequence + 1 for special token
        assert end == 9   # End position + 1 for special token (length of "DEFGH" is 5, so 2+5+1=8+1=9)
    
    def test_get_substring_positions_with_context(self):
        """Test getting substring positions with context."""
        mock_embedder = Mock()
        mock_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_embedder.substring_dict = {"seq1": "DEFGH"}
        mock_embedder.get_substring_positions = BaseEmbedder.get_substring_positions.__get__(mock_embedder, BaseEmbedder)
        
        start, end = mock_embedder.get_substring_positions("seq1", special_tokens=True, context=2)
        
        assert start == 1  # Position - context + special token (max(2-2,0)+1=1)
        assert end == 9   # End position + context + special token (min(1+5+2,20)+1=9)
    
    def test_get_substring_positions_no_special_tokens(self):
        """Test getting substring positions without special tokens."""
        mock_embedder = Mock()
        mock_embedder.sequences = {"seq1": "ACDEFGHIKLMNPQRSTVWY"}
        mock_embedder.substring_dict = {"seq1": "DEFGH"}
        mock_embedder.get_substring_positions = BaseEmbedder.get_substring_positions.__get__(mock_embedder, BaseEmbedder)
        
        start, end = mock_embedder.get_substring_positions("seq1", special_tokens=False, context=0)
        
        assert start == 2  # Position of "DEFGH" in sequence
        assert end == 7   # End position


class TestBaseEmbedderTensorOperations:
    """Test BaseEmbedder tensor operations."""
    
    def test_prepare_tensor_no_flatten(self):
        """Test tensor preparation without flattening."""
        mock_embedder = Mock()
        mock_embedder._prepare_tensor = BaseEmbedder._prepare_tensor.__get__(mock_embedder, BaseEmbedder)
        
        data_list = [torch.randn(10, 768), torch.randn(10, 768)]
        result = mock_embedder._prepare_tensor(data_list, flatten=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10, 768)
    
    def test_prepare_tensor_with_flatten(self):
        """Test tensor preparation with flattening."""
        mock_embedder = Mock()
        mock_embedder._prepare_tensor = BaseEmbedder._prepare_tensor.__get__(mock_embedder, BaseEmbedder)
        
        data_list = [torch.randn(10, 768), torch.randn(10, 768)]
        result = mock_embedder._prepare_tensor(data_list, flatten=True)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10 * 768)
    
    def test_prepare_tensor_empty_list(self):
        """Test tensor preparation with empty list."""
        mock_embedder = Mock()
        mock_embedder._prepare_tensor = BaseEmbedder._prepare_tensor.__get__(mock_embedder, BaseEmbedder)
        
        data_list = []
        
        with pytest.raises(RuntimeError):
            mock_embedder._prepare_tensor(data_list, flatten=False)
    
    def test_prepare_tensor_single_item(self):
        """Test tensor preparation with single item."""
        mock_embedder = Mock()
        mock_embedder._prepare_tensor = BaseEmbedder._prepare_tensor.__get__(mock_embedder, BaseEmbedder)
        
        data_list = [torch.randn(10, 768)]
        result = mock_embedder._prepare_tensor(data_list, flatten=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 10, 768) 