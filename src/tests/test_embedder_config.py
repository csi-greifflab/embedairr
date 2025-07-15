"""
Tests for EmbedderConfig class.

This module contains comprehensive tests for the EmbedderConfig class,
including configuration processing, validation, and utility methods.
"""

import os
import tempfile
import shutil
import torch
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import csv
from types import SimpleNamespace

from pepe.embedders.components.embedder_config import EmbedderConfig


class TestEmbedderConfig:
    """Test suite for EmbedderConfig class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.fasta_path = os.path.join(self.temp_dir, "test.fasta")
        self.substring_path = os.path.join(self.temp_dir, "substrings.csv")
        self.output_path = os.path.join(self.temp_dir, "output")
        
        # Create test FASTA file
        with open(self.fasta_path, "w") as f:
            f.write(">seq1\nACGT\n>seq2\nTGCA\n")
        
        # Create test substring file
        with open(self.substring_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_id", "substring"])
            writer.writerow(["seq1", "ACG"])
            writer.writerow(["seq2", "TGC"])
        
        # Create basic args namespace
        self.args = SimpleNamespace(
            fasta_path=self.fasta_path,
            model_name="test_model",
            output_path=self.output_path,
            experiment_name=None,
            extract_embeddings=["mean_pooled", "per_token"],
            layers=[[1, 2]],
            batch_size=32,
            max_length=512,
            device="cpu",
            precision="float32",
            streaming_output=False,
            num_workers=4,
            flush_batches_after=64,
            disable_special_tokens=False,
            discard_padding=True,
            flatten=False,
            substring_path=self.substring_path,
            context=0,
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_basic(self):
        """Test basic EmbedderConfig initialization."""
        config = EmbedderConfig(self.args)
        
        assert config.fasta_path == self.fasta_path
        assert config.model_link == "test_model"
        assert config.model_name == "test_model"
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.device == torch.device("cpu")
        assert config.precision == "float32"
        assert config.streaming_output is False
        assert config.num_workers == 1  # Should be 1 when streaming is disabled
        assert config.output_types == ["mean_pooled", "per_token"]
        assert config.return_embeddings is True
        assert config.return_contacts is False
        assert config.return_logits is False
    
    def test_model_name_processing_huggingface(self):
        """Test model name processing for Hugging Face models."""
        self.args.model_name = "facebook/esm2_t12_35M_UR50D"
        config = EmbedderConfig(self.args)
        assert config.model_name == "esm2_t12_35M_UR50D"
    
    def test_model_name_processing_local_file(self):
        """Test model name processing for local files."""
        model_file = os.path.join(self.temp_dir, "test_model.pt")
        with open(model_file, "w") as f:
            f.write("dummy")
        
        self.args.model_name = model_file
        config = EmbedderConfig(self.args)
        assert config.model_name == "test_model"
    
    def test_model_name_processing_custom(self):
        """Test model name processing for custom models."""
        self.args.model_name = "custom:my_model"
        config = EmbedderConfig(self.args)
        assert config.model_name == "custom:my_model"
    
    def test_output_prefix_from_experiment_name(self):
        """Test output prefix generation from experiment name."""
        self.args.experiment_name = "my_experiment"
        config = EmbedderConfig(self.args)
        assert config.output_prefix == "my_experiment"
    
    def test_output_prefix_from_fasta_path(self):
        """Test output prefix generation from FASTA path."""
        self.args.experiment_name = None
        config = EmbedderConfig(self.args)
        assert config.output_prefix == "test"
    
    def test_layers_processing_none(self):
        """Test layers processing when None is specified."""
        self.args.layers = [None]
        config = EmbedderConfig(self.args)
        assert config.layers is None
    
    def test_layers_processing_nested(self):
        """Test layers processing with nested lists."""
        self.args.layers = [[1, 2], [3, 4]]
        config = EmbedderConfig(self.args)
        assert config.layers == [1, 2, 3, 4]
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_device_processing_cuda_available(self, mock_cuda):
        """Test device processing when CUDA is available."""
        self.args.device = "cuda"
        config = EmbedderConfig(self.args)
        assert config.device == torch.device("cuda")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_device_processing_cuda_unavailable(self, mock_cuda):
        """Test device processing when CUDA is unavailable."""
        self.args.device = "cuda"
        config = EmbedderConfig(self.args)
        assert config.device == torch.device("cpu")
    
    def test_output_types_processing(self):
        """Test output types processing."""
        self.args.extract_embeddings = ["mean_pooled", "attention_head", "logits"]
        config = EmbedderConfig(self.args)
        assert set(config.output_types) == {"mean_pooled", "attention_head", "logits"}
    
    def test_return_flags_computation(self):
        """Test computation of return flags."""
        # Test embeddings
        self.args.extract_embeddings = ["mean_pooled", "per_token"]
        config = EmbedderConfig(self.args)
        assert config.return_embeddings is True
        assert config.return_contacts is False
        assert config.return_logits is False
        
        # Test attention
        self.args.extract_embeddings = ["attention_head", "attention_layer"]
        config = EmbedderConfig(self.args)
        assert config.return_embeddings is False
        assert config.return_contacts is True
        assert config.return_logits is False
        
        # Test logits
        self.args.extract_embeddings = ["logits"]
        config = EmbedderConfig(self.args)
        assert config.return_embeddings is False
        assert config.return_contacts is False
        assert config.return_logits is True
    
    def test_streaming_configuration_enabled(self):
        """Test streaming configuration when enabled."""
        self.args.streaming_output = True
        self.args.num_workers = 4
        config = EmbedderConfig(self.args)
        assert config.streaming_output is True
        assert config.num_workers == 4
        assert config.max_in_flight == 8
        assert config.flush_batches_after == 64 * 1024 * 1024
    
    def test_streaming_configuration_disabled(self):
        """Test streaming configuration when disabled."""
        self.args.streaming_output = False
        self.args.num_workers = 4
        config = EmbedderConfig(self.args)
        assert config.streaming_output is False
        assert config.num_workers == 1  # Should be forced to 1
    
    def test_substring_loading_success(self):
        """Test successful substring loading."""
        config = EmbedderConfig(self.args)
        assert config.substring_dict is not None
        assert len(config.substring_dict) == 2
        assert config.substring_dict["seq1"] == "ACG"
        assert config.substring_dict["seq2"] == "TGC"
    
    def test_substring_loading_none(self):
        """Test substring loading when path is None."""
        self.args.substring_path = None
        config = EmbedderConfig(self.args)
        assert config.substring_dict is None
    
    def test_substring_loading_file_not_found(self):
        """Test substring loading when file doesn't exist."""
        self.args.substring_path = "/nonexistent/file.csv"
        with pytest.raises(FileNotFoundError):
            EmbedderConfig(self.args)
    
    def test_precision_to_dtype_numpy_float32(self):
        """Test precision to dtype conversion for numpy float32."""
        config = EmbedderConfig(self.args)
        assert config.precision_to_dtype("numpy") == np.float32
    
    def test_precision_to_dtype_numpy_float16(self):
        """Test precision to dtype conversion for numpy float16."""
        self.args.precision = "float16"
        config = EmbedderConfig(self.args)
        assert config.precision_to_dtype("numpy") == np.float16
    
    def test_precision_to_dtype_torch_float32(self):
        """Test precision to dtype conversion for torch float32."""
        config = EmbedderConfig(self.args)
        assert config.precision_to_dtype("torch") == torch.float32
    
    def test_precision_to_dtype_torch_float16(self):
        """Test precision to dtype conversion for torch float16."""
        self.args.precision = "float16"
        config = EmbedderConfig(self.args)
        assert config.precision_to_dtype("torch") == torch.float16
    
    def test_precision_to_dtype_invalid_framework(self):
        """Test precision to dtype conversion with invalid framework."""
        config = EmbedderConfig(self.args)
        with pytest.raises(ValueError, match="Unsupported framework"):
            config.precision_to_dtype("invalid")
    
    def test_precision_to_dtype_invalid_precision(self):
        """Test precision to dtype conversion with invalid precision."""
        self.args.precision = "invalid"
        with pytest.raises(ValueError, match="Invalid precision"):
            EmbedderConfig(self.args)
    
    def test_validation_negative_batch_size(self):
        """Test validation with negative batch size."""
        self.args.batch_size = -1
        with pytest.raises(ValueError, match="Batch size must be positive"):
            EmbedderConfig(self.args)
    
    def test_validation_zero_max_length(self):
        """Test validation with zero max length."""
        self.args.max_length = 0
        with pytest.raises(ValueError, match="Max length must be positive"):
            EmbedderConfig(self.args)
    
    def test_validation_invalid_precision(self):
        """Test validation with invalid precision."""
        self.args.precision = "invalid"
        with pytest.raises(ValueError, match="Invalid precision"):
            EmbedderConfig(self.args)
    
    def test_validation_no_output_types(self):
        """Test validation with no output types."""
        self.args.extract_embeddings = []
        with pytest.raises(ValueError, match="At least one output type must be specified"):
            EmbedderConfig(self.args)
    
    def test_validation_invalid_streaming_workers(self):
        """Test validation with invalid streaming workers."""
        self.args.streaming_output = True
        self.args.num_workers = 0
        with pytest.raises(ValueError, match="Number of workers must be positive"):
            EmbedderConfig(self.args)
    
    def test_validation_missing_fasta_file(self):
        """Test validation with missing FASTA file."""
        self.args.fasta_path = "/nonexistent/file.fasta"
        with pytest.raises(ValueError, match="FASTA file not found"):
            EmbedderConfig(self.args)
    
    def test_validation_missing_substring_file(self):
        """Test validation with missing substring file."""
        self.args.substring_path = "/nonexistent/file.csv"
        with pytest.raises(ValueError, match="Substring file not found"):
            EmbedderConfig(self.args)
    
    def test_get_output_directory(self):
        """Test get_output_directory method."""
        config = EmbedderConfig(self.args)
        output_dir = config.get_output_directory("mean_pooled")
        expected_dir = os.path.join(config.output_path, "mean_pooled")
        assert output_dir == expected_dir
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    def test_get_device_info(self, mock_device_count, mock_cuda_available):
        """Test get_device_info method."""
        config = EmbedderConfig(self.args)
        device_info = config.get_device_info()
        
        assert device_info["device"] == "cpu"
        assert device_info["cuda_available"] is True
        assert device_info["cuda_device_count"] == 2
    
    def test_get_streaming_info(self):
        """Test get_streaming_info method."""
        self.args.streaming_output = True
        self.args.num_workers = 4
        config = EmbedderConfig(self.args)
        streaming_info = config.get_streaming_info()
        
        assert streaming_info["streaming_output"] is True
        assert streaming_info["num_workers"] == 4
        assert streaming_info["max_in_flight"] == 8
        assert streaming_info["flush_batches_after"] == 64 * 1024 * 1024
    
    def test_get_configuration_summary(self):
        """Test get_configuration_summary method."""
        config = EmbedderConfig(self.args)
        summary = config.get_configuration_summary()
        
        assert summary["model_name"] == "test_model"
        assert summary["batch_size"] == 32
        assert summary["max_length"] == 512
        assert summary["output_types"] == ["mean_pooled", "per_token"]
        assert summary["return_embeddings"] is True
        assert summary["return_contacts"] is False
        assert summary["return_logits"] is False
        assert summary["has_substrings"] is True
        assert summary["num_substrings"] == 2
    
    def test_log_configuration(self):
        """Test log_configuration method."""
        config = EmbedderConfig(self.args)
        with patch('pepe.embedders.components.embedder_config.logger') as mock_logger:
            config.log_configuration()
            assert mock_logger.info.called
            # Check that configuration values are logged
            calls = mock_logger.info.call_args_list
            assert any("model_name: test_model" in str(call) for call in calls)
    
    def test_repr(self):
        """Test __repr__ method."""
        config = EmbedderConfig(self.args)
        repr_str = repr(config)
        assert "EmbedderConfig" in repr_str
        assert "model_name='test_model'" in repr_str
        assert "device='cpu'" in repr_str
    
    def test_output_directory_creation(self):
        """Test that output directory is created during initialization."""
        config = EmbedderConfig(self.args)
        assert os.path.exists(config.output_path)
        assert os.path.isdir(config.output_path)
    
    def test_checkpoint_directory_setup(self):
        """Test that checkpoint directory is set up correctly."""
        config = EmbedderConfig(self.args)
        assert config.checkpoint_dir == config.output_path
    
    def test_precision_aliases(self):
        """Test that precision aliases work correctly."""
        # Test float16 aliases
        for precision in ["float16", "16", "half"]:
            self.args.precision = precision
            config = EmbedderConfig(self.args)
            assert config.precision_to_dtype("numpy") == np.float16
        
        # Test float32 aliases  
        for precision in ["float32", "32", "full"]:
            self.args.precision = precision
            config = EmbedderConfig(self.args)
            assert config.precision_to_dtype("numpy") == np.float32
    
    def test_output_types_deduplication(self):
        """Test that duplicate output types are removed."""
        self.args.extract_embeddings = ["mean_pooled", "mean_pooled", "per_token"]
        config = EmbedderConfig(self.args)
        assert config.output_types == ["mean_pooled", "per_token"]
    
    def test_complex_configuration(self):
        """Test complex configuration with multiple features."""
        self.args.extract_embeddings = ["mean_pooled", "attention_head", "logits"]
        self.args.streaming_output = True
        self.args.num_workers = 8
        self.args.layers = [[1, 2], [3, 4, 5]]
        self.args.flatten = True
        self.args.precision = "float16"
        
        config = EmbedderConfig(self.args)
        
        assert config.output_types == ["mean_pooled", "attention_head", "logits"]
        assert config.return_embeddings is True
        assert config.return_contacts is True
        assert config.return_logits is True
        assert config.streaming_output is True
        assert config.num_workers == 8
        assert config.layers == [1, 2, 3, 4, 5]
        assert config.flatten is True
        assert config.precision == "float16"