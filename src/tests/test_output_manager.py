"""
Unit tests for OutputManager class.

This module contains comprehensive tests for the OutputManager class,
testing all its methods and functionality.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pepe.embedders.components.output_manager import OutputManager


# Global fixtures for all test classes
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def basic_output_manager(temp_dir):
    """Create a basic OutputManager instance for testing."""
    return OutputManager(
        output_path=temp_dir,
        model_name="test_model",
        output_prefix="test_prefix",
        output_types=["mean_pooled", "per_token"],
        precision="float32",
        streaming_output=False,
        flatten=False
    )

@pytest.fixture
def streaming_output_manager(temp_dir):
    """Create a streaming OutputManager instance for testing."""
    return OutputManager(
        output_path=temp_dir,
        model_name="test_model",
        output_prefix="test_prefix",
        output_types=["mean_pooled", "attention_head"],
        precision="float16",
        streaming_output=True,
        flatten=True
    )


class TestOutputManager:
    """Test suite for OutputManager class."""


class TestOutputManagerInitialization:
    """Test OutputManager initialization."""

    def test_basic_initialization(self, temp_dir):
        """Test basic OutputManager initialization."""
        manager = OutputManager(
            output_path=temp_dir,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["mean_pooled"],
            precision="float32"
        )
        
        assert manager.output_path == temp_dir
        assert manager.model_name == "test_model"
        assert manager.output_prefix == "test_prefix"
        assert manager.output_types == ["mean_pooled"]
        assert manager.precision == "float32"
        assert manager.streaming_output == False
        assert manager.flatten == False
        assert manager.memmap_registry == {}

    def test_initialization_with_options(self, temp_dir):
        """Test OutputManager initialization with all options."""
        manager = OutputManager(
            output_path=temp_dir,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["mean_pooled", "per_token", "attention_head"],
            precision="float16",
            streaming_output=True,
            flatten=True
        )
        
        assert manager.streaming_output == True
        assert manager.flatten == True
        assert manager.precision == "float16"
        assert len(manager.output_types) == 3


class TestOutputManagerPrecisionConversion:
    """Test precision conversion methods."""

    def test_precision_to_dtype_torch_float16(self, basic_output_manager):
        """Test precision conversion to torch float16."""
        dtype = basic_output_manager._precision_to_dtype("float16", "torch")
        assert dtype == torch.float16

    def test_precision_to_dtype_torch_float32(self, basic_output_manager):
        """Test precision conversion to torch float32."""
        dtype = basic_output_manager._precision_to_dtype("float32", "torch")
        assert dtype == torch.float32

    def test_precision_to_dtype_numpy_float16(self, basic_output_manager):
        """Test precision conversion to numpy float16."""
        dtype = basic_output_manager._precision_to_dtype("float16", "numpy")
        assert dtype == np.float16

    def test_precision_to_dtype_numpy_float32(self, basic_output_manager):
        """Test precision conversion to numpy float32."""
        dtype = basic_output_manager._precision_to_dtype("float32", "numpy")
        assert dtype == np.float32

    def test_precision_to_dtype_alternative_formats(self, basic_output_manager):
        """Test precision conversion with alternative format strings."""
        assert basic_output_manager._precision_to_dtype("16", "torch") == torch.float16
        assert basic_output_manager._precision_to_dtype("32", "torch") == torch.float32
        assert basic_output_manager._precision_to_dtype("half", "torch") == torch.float16
        assert basic_output_manager._precision_to_dtype("full", "torch") == torch.float32

    def test_precision_to_dtype_invalid_precision(self, basic_output_manager):
        """Test precision conversion with invalid precision."""
        with pytest.raises(ValueError, match="Unsupported precision"):
            basic_output_manager._precision_to_dtype("invalid", "torch")

    def test_precision_to_dtype_invalid_framework(self, basic_output_manager):
        """Test precision conversion with invalid framework."""
        # This should return None for invalid frameworks based on the implementation
        result = basic_output_manager._precision_to_dtype("float32", "invalid")
        assert result is None


class TestOutputManagerDirectoryOperations:
    """Test directory creation and management."""

    def test_create_output_directories(self, basic_output_manager):
        """Test creation of output directories."""
        basic_output_manager.create_output_directories()
        
        for output_type in basic_output_manager.output_types:
            expected_dir = os.path.join(basic_output_manager.output_path, output_type)
            assert os.path.exists(expected_dir)
            assert os.path.isdir(expected_dir)

    def test_create_output_directories_existing(self, basic_output_manager):
        """Test creation of output directories when they already exist."""
        # Create directories manually first
        for output_type in basic_output_manager.output_types:
            output_dir = os.path.join(basic_output_manager.output_path, output_type)
            os.makedirs(output_dir)
        
        # Should not raise an error
        basic_output_manager.create_output_directories()
        
        # Directories should still exist
        for output_type in basic_output_manager.output_types:
            expected_dir = os.path.join(basic_output_manager.output_path, output_type)
            assert os.path.exists(expected_dir)

    def test_create_output_directories_multiple_levels(self, temp_dir):
        """Test creation of nested output directories."""
        nested_path = os.path.join(temp_dir, "nested", "output")
        manager = OutputManager(
            output_path=nested_path,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["mean_pooled"],
            precision="float32"
        )
        
        manager.create_output_directories()
        
        expected_dir = os.path.join(nested_path, "mean_pooled")
        assert os.path.exists(expected_dir)


class TestOutputManagerFilePathGeneration:
    """Test file path generation methods."""

    def test_make_output_filepath_basic(self, basic_output_manager):
        """Test basic output filepath generation."""
        filepath = basic_output_manager.make_output_filepath("embeddings", "/output")
        expected = "/output/test_prefix_test_model_embeddings.npy"
        assert filepath == expected

    def test_make_output_filepath_with_layer(self, basic_output_manager):
        """Test output filepath generation with layer."""
        filepath = basic_output_manager.make_output_filepath("embeddings", "/output", layer=-1)
        expected = "/output/test_prefix_test_model_embeddings_layer_-1.npy"
        assert filepath == expected

    def test_make_output_filepath_with_head(self, basic_output_manager):
        """Test output filepath generation with head."""
        filepath = basic_output_manager.make_output_filepath("attention", "/output", layer=-1, head=5)
        expected = "/output/test_prefix_test_model_attention_layer_-1_head_6.npy"
        assert filepath == expected

    def test_make_output_filepath_with_layer_and_head(self, basic_output_manager):
        """Test output filepath generation with both layer and head."""
        filepath = basic_output_manager.make_output_filepath("attention", "/output", layer=-2, head=0)
        expected = "/output/test_prefix_test_model_attention_layer_-2_head_1.npy"
        assert filepath == expected

    def test_make_output_filepath_different_model_names(self, temp_dir):
        """Test filepath generation with different model names."""
        manager = OutputManager(
            output_path=temp_dir,
            model_name="different_model",
            output_prefix="test_prefix",
            output_types=["mean_pooled"],
            precision="float32"
        )
        
        filepath = manager.make_output_filepath("embeddings", "/output")
        expected = "/output/test_prefix_different_model_embeddings.npy"
        assert filepath == expected


class TestOutputManagerOutputTypes:
    """Test output type management."""

    def test_get_output_types_basic(self, basic_output_manager):
        """Test basic output type retrieval."""
        extract_embeddings = ["mean_pooled", "per_token"]
        output_types = basic_output_manager.get_output_types(extract_embeddings)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
        assert len(output_types) == 2

    def test_get_output_types_with_duplicates(self, basic_output_manager):
        """Test output type retrieval with duplicates."""
        extract_embeddings = ["mean_pooled", "per_token", "mean_pooled"]
        output_types = basic_output_manager.get_output_types(extract_embeddings)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
        assert output_types.count("mean_pooled") == 1
        assert len(output_types) == 2

    def test_get_output_types_all_types(self, basic_output_manager):
        """Test output type retrieval with all types."""
        extract_embeddings = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        output_types = basic_output_manager.get_output_types(extract_embeddings)
        
        expected_types = [
            "per_token", "mean_pooled", "substring_pooled", 
            "attention_head", "attention_layer", "attention_model", "logits"
        ]
        
        for expected_type in expected_types:
            assert expected_type in output_types
        assert len(output_types) == len(expected_types)

    def test_get_output_types_invalid_types(self, basic_output_manager):
        """Test output type retrieval with invalid types."""
        extract_embeddings = ["mean_pooled", "invalid_type", "per_token"]
        output_types = basic_output_manager.get_output_types(extract_embeddings)
        
        assert "mean_pooled" in output_types
        assert "per_token" in output_types
        assert "invalid_type" not in output_types
        assert len(output_types) == 2

    def test_get_output_types_empty_list(self, basic_output_manager):
        """Test output type retrieval with empty list."""
        extract_embeddings = []
        output_types = basic_output_manager.get_output_types(extract_embeddings)
        
        assert len(output_types) == 0


class TestOutputManagerTensorPreparation:
    """Test tensor preparation methods."""

    def test_prepare_tensor_no_flatten(self, basic_output_manager):
        """Test tensor preparation without flattening."""
        data_list = [torch.randn(10, 768), torch.randn(10, 768)]
        result = basic_output_manager._prepare_tensor(data_list, flatten=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10, 768)

    def test_prepare_tensor_with_flatten(self, basic_output_manager):
        """Test tensor preparation with flattening."""
        data_list = [torch.randn(10, 768), torch.randn(10, 768)]
        result = basic_output_manager._prepare_tensor(data_list, flatten=True)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10 * 768)

    def test_prepare_tensor_single_item(self, basic_output_manager):
        """Test tensor preparation with single item."""
        data_list = [torch.randn(10, 768)]
        result = basic_output_manager._prepare_tensor(data_list, flatten=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 10, 768)

    def test_prepare_tensor_empty_list(self, basic_output_manager):
        """Test tensor preparation with empty list."""
        data_list = []
        
        with pytest.raises(RuntimeError):
            basic_output_manager._prepare_tensor(data_list, flatten=False)

    def test_prepare_tensor_different_shapes(self, basic_output_manager):
        """Test tensor preparation with different tensor shapes."""
        data_list = [torch.randn(5, 512), torch.randn(5, 512)]
        result = basic_output_manager._prepare_tensor(data_list, flatten=False)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 5, 512)


class TestOutputManagerSequenceIndices:
    """Test sequence index export functionality."""

    def test_export_sequence_indices_basic(self, basic_output_manager):
        """Test basic sequence index export."""
        sequence_labels = ["seq1", "seq2", "seq3"]
        fasta_path = "test.fasta"
        
        basic_output_manager.export_sequence_indices(sequence_labels, fasta_path)
        
        idx_file = os.path.join(basic_output_manager.output_path, "test_idx.csv")
        assert os.path.exists(idx_file)
        
        with open(idx_file, "r") as f:
            content = f.read()
            assert "index,sequence_id" in content
            assert "0,seq1" in content
            assert "1,seq2" in content
            assert "2,seq3" in content

    def test_export_sequence_indices_with_path(self, basic_output_manager):
        """Test sequence index export with full path."""
        sequence_labels = ["seq1", "seq2"]
        fasta_path = "/path/to/sequences.fasta"
        
        basic_output_manager.export_sequence_indices(sequence_labels, fasta_path)
        
        idx_file = os.path.join(basic_output_manager.output_path, "sequences_idx.csv")
        assert os.path.exists(idx_file)

    def test_export_sequence_indices_empty_list(self, basic_output_manager):
        """Test sequence index export with empty list."""
        sequence_labels = []
        fasta_path = "test.fasta"
        
        basic_output_manager.export_sequence_indices(sequence_labels, fasta_path)
        
        idx_file = os.path.join(basic_output_manager.output_path, "test_idx.csv")
        assert os.path.exists(idx_file)
        
        with open(idx_file, "r") as f:
            content = f.read()
            assert "index,sequence_id" in content
            # Should only have header
            assert len(content.split("\n")) == 2  # header + empty line

    def test_export_sequence_indices_special_characters(self, basic_output_manager):
        """Test sequence index export with special characters."""
        sequence_labels = ["seq_1", "seq-2", "seq.3"]
        fasta_path = "test.fasta"
        
        basic_output_manager.export_sequence_indices(sequence_labels, fasta_path)
        
        idx_file = os.path.join(basic_output_manager.output_path, "test_idx.csv")
        assert os.path.exists(idx_file)
        
        with open(idx_file, "r") as f:
            content = f.read()
            assert "0,seq_1" in content
            assert "1,seq-2" in content
            assert "2,seq.3" in content


class TestOutputManagerCleanup:
    """Test cleanup functionality."""

    def test_cleanup_basic(self, basic_output_manager):
        """Test basic cleanup functionality."""
        # Add some mock memmap objects
        mock_memmap1 = Mock()
        mock_memmap2 = Mock()
        basic_output_manager.memmap_registry = {
            ("type1", -1, None): mock_memmap1,
            ("type2", -2, None): mock_memmap2
        }
        
        basic_output_manager.cleanup()
        
        # Check that close was called on memmap objects
        mock_memmap1.close.assert_called_once()
        mock_memmap2.close.assert_called_once()
        
        # Check that registry was cleared
        assert len(basic_output_manager.memmap_registry) == 0

    def test_cleanup_no_close_method(self, basic_output_manager):
        """Test cleanup with objects that don't have close method."""
        # Add objects without close method
        mock_obj = Mock()
        del mock_obj.close  # Remove close method
        basic_output_manager.memmap_registry = {
            ("type1", -1, None): mock_obj,
        }
        
        # Should not raise an error
        basic_output_manager.cleanup()
        
        # Registry should still be cleared
        assert len(basic_output_manager.memmap_registry) == 0

    def test_cleanup_empty_registry(self, basic_output_manager):
        """Test cleanup with empty registry."""
        basic_output_manager.memmap_registry = {}
        
        # Should not raise an error
        basic_output_manager.cleanup()
        
        assert len(basic_output_manager.memmap_registry) == 0

    def test_destructor_calls_cleanup(self, temp_dir):
        """Test that destructor calls cleanup."""
        with patch.object(OutputManager, 'cleanup') as mock_cleanup:
            manager = OutputManager(
                output_path=temp_dir,
                model_name="test_model",
                output_prefix="test_prefix",
                output_types=["mean_pooled"],
                precision="float32"
            )
            
            # Force garbage collection
            del manager
            
            # Cleanup should have been called
            mock_cleanup.assert_called_once()


class TestOutputManagerDiskSpacePreallocation:
    """Test disk space preallocation functionality."""

    def test_preallocate_disk_space_basic(self, basic_output_manager):
        """Test basic disk space preallocation."""
        shapes_info = {
            "mean_pooled": {
                "shape": (10, 768),
                "layers": [-1, -2]
            }
        }
        
        with patch('pepe.embedders.components.output_manager.check_disk_free_space'), \
             patch('pepe.embedders.components.output_manager.open_memmap') as mock_memmap:
            
            mock_memmap.return_value = Mock()
            
            registry = basic_output_manager.preallocate_disk_space(shapes_info)
            
            # Check that memmap was called for each layer
            assert mock_memmap.call_count == 2
            
            # Check registry structure
            assert ("mean_pooled", -1, None) in registry
            assert ("mean_pooled", -2, None) in registry

    def test_preallocate_disk_space_with_heads(self, temp_dir):
        """Test disk space preallocation with attention heads."""
        # Create OutputManager with attention_head in output_types
        manager = OutputManager(
            output_path=temp_dir,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["attention_head"],
            precision="float32"
        )
        
        shapes_info = {
            "attention_head": {
                "shape": (10, 50, 50),
                "layers": [-1],
                "heads": 12
            }
        }
        
        with patch('pepe.embedders.components.output_manager.check_disk_free_space'), \
             patch('pepe.embedders.components.output_manager.open_memmap') as mock_memmap:
            
            mock_memmap.return_value = Mock()
            
            registry = manager.preallocate_disk_space(shapes_info)
            
            # Check that memmap was called for each head
            assert mock_memmap.call_count == 12
            
            # Check registry structure
            for head in range(12):
                assert ("attention_head", -1, head) in registry

    def test_preallocate_disk_space_model_level(self, temp_dir):
        """Test disk space preallocation for model-level outputs."""
        # Create OutputManager with attention_model in output_types
        manager = OutputManager(
            output_path=temp_dir,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["attention_model"],
            precision="float32"
        )
        
        shapes_info = {
            "attention_model": {
                "shape": (10, 50, 50)
            }
        }
        
        with patch('pepe.embedders.components.output_manager.check_disk_free_space'), \
             patch('pepe.embedders.components.output_manager.open_memmap') as mock_memmap:
            
            mock_memmap.return_value = Mock()
            
            registry = manager.preallocate_disk_space(shapes_info)
            
            # Check that memmap was called once
            assert mock_memmap.call_count == 1
            
            # Check registry structure
            assert ("attention_model", None, None) in registry

    def test_preallocate_disk_space_missing_shape_info(self, basic_output_manager):
        """Test disk space preallocation with missing shape info."""
        shapes_info = {
            "other_type": {
                "shape": (10, 768),
                "layers": [-1]
            }
        }
        
        with patch('pepe.embedders.components.output_manager.check_disk_free_space'), \
             patch('pepe.embedders.components.output_manager.open_memmap') as mock_memmap:
            
            mock_memmap.return_value = Mock()
            
            registry = basic_output_manager.preallocate_disk_space(shapes_info)
            
            # Should not call memmap for missing output types
            assert mock_memmap.call_count == 0
            assert len(registry) == 0

    @patch('pepe.embedders.components.output_manager.check_disk_free_space')
    def test_preallocate_disk_space_insufficient_space(self, mock_check_space, basic_output_manager):
        """Test disk space preallocation with insufficient space."""
        shapes_info = {
            "mean_pooled": {
                "shape": (10, 768),
                "layers": [-1]
            }
        }
        
        # Mock check_disk_free_space to raise SystemExit
        mock_check_space.side_effect = SystemExit("Insufficient disk space")
        
        with pytest.raises(SystemExit):
            basic_output_manager.preallocate_disk_space(shapes_info)


class TestOutputManagerDataExport:
    """Test data export functionality."""

    def test_export_data_basic(self, basic_output_manager):
        """Test basic data export."""
        # Set up mock memmap registry
        mock_memmap = Mock()
        mock_memmap.__setitem__ = Mock()
        basic_output_manager.memmap_registry = {
            ("embeddings", -1, None): mock_memmap
        }
        
        data = np.random.randn(10, 768)
        basic_output_manager.export_data("embeddings", -1, None, data)
        
        # Check that data was written to memmap
        mock_memmap.__setitem__.assert_called_once()
        mock_memmap.flush.assert_called_once()

    def test_export_data_with_head(self, basic_output_manager):
        """Test data export with attention head."""
        # Set up mock memmap registry
        mock_memmap = Mock()
        mock_memmap.__setitem__ = Mock()
        basic_output_manager.memmap_registry = {
            ("attention", -1, 5): mock_memmap
        }
        
        data = np.random.randn(10, 50, 50)
        basic_output_manager.export_data("attention", -1, 5, data)
        
        # Check that data was written to memmap
        mock_memmap.__setitem__.assert_called_once()
        mock_memmap.flush.assert_called_once()

    def test_export_data_missing_key(self, basic_output_manager):
        """Test data export with missing key."""
        # Empty memmap registry
        basic_output_manager.memmap_registry = {}
        
        data = np.random.randn(10, 768)
        
        # Should not raise an error but should log a warning
        basic_output_manager.export_data("embeddings", -1, None, data)

    def test_export_data_different_shapes(self, basic_output_manager):
        """Test data export with different data shapes."""
        # Set up mock memmap registry
        mock_memmap = Mock()
        mock_memmap.__setitem__ = Mock()
        basic_output_manager.memmap_registry = {
            ("embeddings", -1, None): mock_memmap
        }
        
        # Test with different shapes
        data_shapes = [(5, 512), (20, 1024), (1, 768)]
        
        for shape in data_shapes:
            data = np.random.randn(*shape)
            basic_output_manager.export_data("embeddings", -1, None, data)
            
            # Check that data was written
            mock_memmap.__setitem__.assert_called()
            mock_memmap.flush.assert_called()


class TestOutputManagerBatchExport:
    """Test batch export functionality."""

    def test_export_to_disk_basic(self, basic_output_manager):
        """Test basic batch export to disk."""
        # Create mock output data
        output_data = {
            "mean_pooled": {
                "output_data": {
                    -1: [torch.randn(10, 768), torch.randn(10, 768)]
                }
            }
        }
        
        with patch('numpy.save') as mock_save:
            basic_output_manager.export_to_disk(output_data, [-1])
            
            # Check that numpy.save was called
            mock_save.assert_called_once()

    def test_export_to_disk_with_heads(self, basic_output_manager):
        """Test batch export with attention heads."""
        # Update output types to include attention_head
        basic_output_manager.output_types = ["attention_head"]
        
        output_data = {
            "attention_head": {
                "output_data": {
                    -1: {
                        0: [torch.randn(10, 50, 50), torch.randn(10, 50, 50)],
                        1: [torch.randn(10, 50, 50), torch.randn(10, 50, 50)]
                    }
                }
            }
        }
        
        with patch('numpy.save') as mock_save:
            basic_output_manager.export_to_disk(output_data, [-1], num_heads=2)
            
            # Check that numpy.save was called for each head
            assert mock_save.call_count == 2

    def test_export_to_disk_missing_data(self, basic_output_manager):
        """Test batch export with missing data."""
        # Create empty output data
        output_data = {}
        
        with patch('numpy.save') as mock_save:
            basic_output_manager.export_to_disk(output_data, [-1])
            
            # Should not save anything
            mock_save.assert_not_called()

    def test_export_to_disk_missing_heads(self, basic_output_manager):
        """Test batch export with missing num_heads parameter."""
        # Update output types to include attention_head
        basic_output_manager.output_types = ["attention_head"]
        
        output_data = {
            "attention_head": {
                "output_data": {
                    -1: {
                        0: [torch.randn(10, 50, 50)]
                    }
                }
            }
        }
        
        with patch('numpy.save') as mock_save:
            basic_output_manager.export_to_disk(output_data, [-1], num_heads=None)
            
            # Should not save anything due to missing num_heads
            mock_save.assert_not_called()

    def test_export_to_disk_model_level(self, basic_output_manager):
        """Test batch export for model-level outputs."""
        # Update output types to include attention_model
        basic_output_manager.output_types = ["attention_model"]
        
        output_data = {
            "attention_model": {
                "output_data": [torch.randn(10, 50, 50), torch.randn(10, 50, 50)]
            }
        }
        
        with patch('numpy.save') as mock_save:
            basic_output_manager.export_to_disk(output_data, [-1])
            
            # Check that numpy.save was called once
            mock_save.assert_called_once()

    def test_export_to_disk_with_flatten(self, temp_dir):
        """Test batch export with flattening enabled."""
        manager = OutputManager(
            output_path=temp_dir,
            model_name="test_model",
            output_prefix="test_prefix",
            output_types=["per_token"],
            precision="float32",
            flatten=True
        )
        
        output_data = {
            "per_token": {
                "output_data": {
                    -1: [torch.randn(10, 768), torch.randn(10, 768)]
                }
            }
        }
        
        with patch('numpy.save') as mock_save:
            manager.export_to_disk(output_data, [-1])
            
            # Check that numpy.save was called
            mock_save.assert_called_once()
            
            # Check that the saved data was flattened
            call_args = mock_save.call_args
            saved_data = call_args[0][1]  # Second argument is the data
            assert len(saved_data.shape) == 2  # Should be flattened to 2D
            assert saved_data.shape[1] == 10 * 768  # Should be flattened dimension 