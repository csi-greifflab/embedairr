"""
Tests for StreamingIO class.

This module contains comprehensive tests for the StreamingIO class,
including file operations, streaming functionality, and error handling.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import csv

from pepe.embedders.components.streaming_io import StreamingIO


class TestStreamingIO:
    """Test suite for StreamingIO class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "output")
        self.output_prefix = "test_sequence"
        self.model_name = "test_model"
        self.output_types = ["mean_pooled", "per_token", "logits"]
        self.precision = "float32"
        self.fasta_path = os.path.join(self.temp_dir, "test.fasta")
        
        # Create test FASTA file
        with open(self.fasta_path, "w") as f:
            f.write(">seq1\nACGT\n>seq2\nTGCA\n")
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        self.streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            fasta_path=self.fasta_path,
            layers=[1, 2],
            num_heads=8,
            flatten=False,
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test StreamingIO initialization."""
        assert self.streaming_io.output_path == self.output_path
        assert self.streaming_io.output_prefix == self.output_prefix
        assert self.streaming_io.model_name == self.model_name
        assert self.streaming_io.output_types == self.output_types
        assert self.streaming_io.precision == self.precision
        assert self.streaming_io.fasta_path == self.fasta_path
        assert self.streaming_io.layers == [1, 2]
        assert self.streaming_io.num_heads == 8
        assert self.streaming_io.flatten is False
        assert self.streaming_io.checkpoint_dir == self.output_path
        assert self.streaming_io.io_dispatcher is None
        assert self.streaming_io.memmap_registry == {}
    
    def test_precision_to_dtype(self):
        """Test precision string to numpy dtype conversion."""
        # Test float16 variations
        streaming_io_16 = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="float16",
        )
        assert streaming_io_16._precision_to_dtype() == np.float16
        
        streaming_io_16b = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="16",
        )
        assert streaming_io_16b._precision_to_dtype() == np.float16
        
        streaming_io_half = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="half",
        )
        assert streaming_io_half._precision_to_dtype() == np.float16
        
        # Test float32 variations
        assert self.streaming_io._precision_to_dtype() == np.float32
        
        streaming_io_32 = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="32",
        )
        assert streaming_io_32._precision_to_dtype() == np.float32
        
        streaming_io_full = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="full",
        )
        assert streaming_io_full._precision_to_dtype() == np.float32
        
        # Test invalid precision
        streaming_io_invalid = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision="invalid",
        )
        with pytest.raises(ValueError, match="Unsupported precision"):
            streaming_io_invalid._precision_to_dtype()
    
    def test_make_output_filepath(self):
        """Test output file path generation."""
        output_dir = os.path.join(self.output_path, "test_output")
        
        # Test basic path
        filepath = self.streaming_io.make_output_filepath("mean_pooled", output_dir)
        expected = os.path.join(output_dir, "test_sequence_test_model_mean_pooled.npy")
        assert filepath == expected
        
        # Test with layer
        filepath = self.streaming_io.make_output_filepath("per_token", output_dir, layer=5)
        expected = os.path.join(output_dir, "test_sequence_test_model_per_token_layer_5.npy")
        assert filepath == expected
        
        # Test with layer and head
        filepath = self.streaming_io.make_output_filepath("attention_head", output_dir, layer=3, head=2)
        expected = os.path.join(output_dir, "test_sequence_test_model_attention_head_layer_3_head_3.npy")
        assert filepath == expected
    
    def test_create_output_dirs(self):
        """Test output directory creation."""
        # Remove existing directories
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if os.path.exists(output_type_path):
                shutil.rmtree(output_type_path)
        
        # Create directories
        self.streaming_io.create_output_dirs()
        
        # Verify directories were created
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            assert os.path.exists(output_type_path)
            assert os.path.isdir(output_type_path)
    
    @patch('pepe.embedders.components.streaming_io.check_disk_free_space')
    @patch('pepe.embedders.components.streaming_io.open_memmap')
    def test_preallocate_disk_space(self, mock_open_memmap, mock_check_disk):
        """Test disk space preallocation."""
        # Mock memory map arrays
        mock_memmap = Mock()
        mock_open_memmap.return_value = mock_memmap
        
        # Create test output data registry
        output_data_registry = {
            "mean_pooled": {
                "output_data": {1: None, 2: None},
                "output_dir": os.path.join(self.output_path, "mean_pooled"),
            },
            "per_token": {
                "output_data": {1: None, 2: None},
                "output_dir": os.path.join(self.output_path, "per_token"),
            },
            "logits": {
                "output_data": None,
                "output_dir": os.path.join(self.output_path, "logits"),
            }
        }
        
        # Test parameters
        num_sequences = 100
        max_length = 512
        embedding_size = 768
        
        # Call preallocate_disk_space
        result = self.streaming_io.preallocate_disk_space(
            output_data_registry, num_sequences, max_length, embedding_size
        )
        
        # Verify disk space check was called
        mock_check_disk.assert_called_once()
        
        # Verify memory maps were created
        expected_calls = len(self.streaming_io.layers) * 2 + 1  # 2 layer-based + 1 model-level
        assert mock_open_memmap.call_count == expected_calls
        
        # Verify registry was populated
        assert len(result) == expected_calls
        assert self.streaming_io.memmap_registry == result
    
    @patch('pepe.embedders.components.streaming_io.MultiIODispatcher')
    def test_initialize_streaming_io(self, mock_dispatcher_class):
        """Test streaming I/O initialization."""
        # Set up mock
        mock_dispatcher = Mock()
        mock_dispatcher_class.return_value = mock_dispatcher
        mock_dispatcher.get_resume_info.return_value = None
        
        # Set up streaming output
        self.streaming_io.streaming_output = True
        self.streaming_io.memmap_registry = {("test", None, None): Mock()}
        
        # Initialize streaming I/O
        self.streaming_io.initialize_streaming_io()
        
        # Verify dispatcher was created
        mock_dispatcher_class.assert_called_once_with(
            self.streaming_io.memmap_registry,
            num_workers=1,
            flush_bytes_limit=64 * 1024 * 1024,
            heavy_output_type="per_token",
            checkpoint_dir=self.output_path,
        )
        
        # Verify dispatcher was set
        assert self.streaming_io.io_dispatcher == mock_dispatcher
    
    def test_get_queue_fullness(self):
        """Test queue fullness checking."""
        # Test without dispatcher
        assert self.streaming_io.get_queue_fullness() == 0.0
        
        # Test with dispatcher
        mock_dispatcher = Mock()
        mock_dispatcher.queue_fullness.return_value = 0.5
        self.streaming_io.io_dispatcher = mock_dispatcher
        
        assert self.streaming_io.get_queue_fullness() == 0.5
        mock_dispatcher.queue_fullness.assert_called_once()
    
    def test_stop_streaming(self):
        """Test stopping streaming I/O."""
        # Test without dispatcher
        self.streaming_io.stop_streaming()  # Should not raise
        
        # Test with dispatcher
        mock_dispatcher = Mock()
        self.streaming_io.io_dispatcher = mock_dispatcher
        
        self.streaming_io.stop_streaming()
        
        mock_dispatcher.stop.assert_called_once()
        assert self.streaming_io.io_dispatcher is None
    
    @patch('numpy.save')
    def test_export_to_disk(self, mock_np_save):
        """Test export to disk functionality."""
        # Create mock prepare_tensor function
        def mock_prepare_tensor(data, flatten):
            return np.array([1, 2, 3])
        
        # Create test output data registry
        output_data_registry = {
            "mean_pooled": {
                "output_data": {1: np.array([1, 2, 3]), 2: np.array([4, 5, 6])},
                "output_dir": os.path.join(self.output_path, "mean_pooled"),
            },
            "per_token": {
                "output_data": {1: np.array([1, 2, 3]), 2: np.array([4, 5, 6])},
                "output_dir": os.path.join(self.output_path, "per_token"),
            },
            "logits": {
                "output_data": np.array([1, 2, 3]),
                "output_dir": os.path.join(self.output_path, "logits"),
            }
        }
        
        # Export to disk
        self.streaming_io.export_to_disk(output_data_registry, mock_prepare_tensor)
        
        # Verify save was called for each output
        expected_calls = len(self.streaming_io.layers) * 2 + 1  # 2 layer-based + 1 model-level
        assert mock_np_save.call_count == expected_calls
    
    def test_export_sequence_indices(self):
        """Test sequence index export."""
        sequence_labels = ["seq1", "seq2", "seq3"]
        
        # Export sequence indices
        self.streaming_io.export_sequence_indices(sequence_labels)
        
        # Verify file was created
        expected_file = os.path.join(self.output_path, "test_idx.csv")
        assert os.path.exists(expected_file)
        
        # Verify file contents
        with open(expected_file, "r") as f:
            reader = csv.reader(f)
            lines = list(reader)
            
        assert lines[0] == ["index", "sequence_id"]
        assert lines[1] == ["0", "seq1"]
        assert lines[2] == ["1", "seq2"]
        assert lines[3] == ["2", "seq3"]
    
    def test_export_sequence_indices_no_fasta(self):
        """Test sequence index export without FASTA path."""
        # Create StreamingIO without FASTA path
        streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            fasta_path=None,
        )
        
        sequence_labels = ["seq1", "seq2", "seq3"]
        
        # Export should not create file
        streaming_io.export_sequence_indices(sequence_labels)
        
        # Verify no file was created
        expected_file = os.path.join(self.output_path, "test_idx.csv")
        assert not os.path.exists(expected_file)
    
    def test_cleanup_checkpoint(self):
        """Test checkpoint cleanup."""
        # Create a fake checkpoint file
        checkpoint_file = os.path.join(self.output_path, "global_checkpoint.json")
        with open(checkpoint_file, "w") as f:
            f.write('{"test": "data"}')
        
        assert os.path.exists(checkpoint_file)
        
        # Clean up checkpoint
        self.streaming_io.cleanup_checkpoint()
        
        # Verify file was removed
        assert not os.path.exists(checkpoint_file)
    
    def test_cleanup_checkpoint_no_file(self):
        """Test checkpoint cleanup when no file exists."""
        checkpoint_file = os.path.join(self.output_path, "global_checkpoint.json")
        assert not os.path.exists(checkpoint_file)
        
        # Should not raise error
        self.streaming_io.cleanup_checkpoint()
    
    def test_cleanup_checkpoint_permission_error(self):
        """Test checkpoint cleanup with permission error."""
        # Create a fake checkpoint file
        checkpoint_file = os.path.join(self.output_path, "global_checkpoint.json")
        with open(checkpoint_file, "w") as f:
            f.write('{"test": "data"}')
        
        # Mock os.remove to raise permission error
        with patch('os.remove', side_effect=PermissionError("Permission denied")):
            # Should not raise error, just log warning
            self.streaming_io.cleanup_checkpoint()
    
    def test_get_resume_info(self):
        """Test getting resume information."""
        # Test without dispatcher
        assert self.streaming_io.get_resume_info() is None
        
        # Test with dispatcher
        mock_dispatcher = Mock()
        expected_info = {"test": "info"}
        mock_dispatcher.get_resume_info.return_value = expected_info
        self.streaming_io.io_dispatcher = mock_dispatcher
        
        result = self.streaming_io.get_resume_info()
        assert result == expected_info
        mock_dispatcher.get_resume_info.assert_called_once()
    
    def test_streaming_output_configuration(self):
        """Test streaming output configuration."""
        # Test with streaming enabled
        streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            streaming_output=True,
            num_workers=4,
            flush_batches_after=128 * 1024 * 1024,
        )
        
        assert streaming_io.streaming_output is True
        assert streaming_io.num_workers == 4
        assert streaming_io.flush_batches_after == 128 * 1024 * 1024
        
        # Test with streaming disabled
        streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            streaming_output=False,
            num_workers=4,
        )
        
        assert streaming_io.streaming_output is False
        assert streaming_io.num_workers == 1  # Should be forced to 1
    
    def test_flattening_configuration(self):
        """Test flattening configuration."""
        # Test with flattening enabled
        streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            flatten=True,
        )
        
        assert streaming_io.flatten is True
        
        # Test with flattening disabled
        streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            flatten=False,
        )
        
        assert streaming_io.flatten is False 