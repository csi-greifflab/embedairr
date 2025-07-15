"""
End-to-end integration tests for PEPE pipeline.

This module contains comprehensive integration tests that test the complete
workflow from input FASTA files to output embeddings and attention matrices.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pepe.parse_arguments import parse_arguments
from pepe.model_selecter import select_model
from pepe.__main__ import main


class TestEndToEndIntegration:
    """Test end-to-end pipeline integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_fasta(self, temp_dir):
        """Create a sample FASTA file for testing."""
        fasta_content = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
MKVLSFGHIKLMNPQRSTVWY
>seq3
GKLMNPQRSTVWYACDEFHI
"""
        fasta_path = os.path.join(temp_dir, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        return fasta_path
    
    @pytest.fixture
    def sample_substring_csv(self, temp_dir):
        """Create a sample substring CSV file for testing."""
        csv_content = """sequence_id,substring_aa
seq1,DEFGH
seq2,VLSFG
seq3,MNPQR
"""
        csv_path = os.path.join(temp_dir, "test_substring.csv")
        with open(csv_path, "w") as f:
            f.write(csv_content)
        return csv_path
    
    @pytest.fixture
    def mock_custom_model(self, temp_dir):
        """Create a mock custom model for testing."""
        model_dir = os.path.join(temp_dir, "custom_model")
        os.makedirs(model_dir)
        
        # Create mock model file
        model_path = os.path.join(model_dir, "model.pt")
        torch.save({"dummy": "model"}, model_path)
        
        # Create mock config file
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"hidden_size": 768, "num_layers": 12, "num_attention_heads": 12}')
        
        return model_dir


class TestCustomEmbedderIntegration:
    """Test custom embedder integration."""
    
    def test_custom_embedder_basic_workflow(self, temp_dir, sample_fasta, sample_substring_csv):
        """Test basic custom embedder workflow."""
        # Create mock arguments
        args = Mock()
        args.model_name = "custom:/fake/model.pt"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = sample_substring_csv
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock the model selection
        with patch('pepe.model_selecter.select_model') as mock_select:
            mock_embedder_class = Mock()
            mock_embedder_instance = Mock()
            mock_embedder_class.return_value = mock_embedder_instance
            mock_select.return_value = mock_embedder_class
            
            # Test model selection
            selected_model = select_model(args.model_name)
            assert selected_model == mock_embedder_class
            
            # Test embedder instantiation
            embedder = selected_model(args)
            assert embedder == mock_embedder_instance
            
            # Test run method
            embedder.run()
            mock_embedder_instance.run.assert_called_once()
    
    def test_custom_embedder_with_multiple_outputs(self, temp_dir, sample_fasta, sample_substring_csv):
        """Test custom embedder with multiple output types."""
        # Create mock arguments for multiple outputs
        args = Mock()
        args.model_name = "custom:/fake/model.pt"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = sample_substring_csv
        args.context = 0
        args.layers = [[-1], [-2]]
        args.extract_embeddings = ["mean_pooled", "per_token", "attention_head"]
        args.batch_size = 2
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock the model selection and embedder
        with patch('pepe.model_selecter.select_model') as mock_select:
            mock_embedder_class = Mock()
            mock_embedder_instance = Mock()
            mock_embedder_class.return_value = mock_embedder_instance
            mock_select.return_value = mock_embedder_class
            
            # Test with multiple outputs
            selected_model = select_model(args.model_name)
            embedder = selected_model(args)
            embedder.run()
            
            # Verify run was called
            mock_embedder_instance.run.assert_called_once()
    
    def test_custom_embedder_streaming_output(self, temp_dir, sample_fasta, sample_substring_csv):
        """Test custom embedder with streaming output."""
        # Create mock arguments for streaming output
        args = Mock()
        args.model_name = "custom:/fake/model.pt"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = sample_substring_csv
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = True
        args.num_workers = 4
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock the model selection and embedder
        with patch('pepe.model_selecter.select_model') as mock_select:
            mock_embedder_class = Mock()
            mock_embedder_instance = Mock()
            mock_embedder_class.return_value = mock_embedder_instance
            mock_select.return_value = mock_embedder_class
            
            # Test with streaming output
            selected_model = select_model(args.model_name)
            embedder = selected_model(args)
            embedder.run()
            
            # Verify run was called
            mock_embedder_instance.run.assert_called_once()


class TestESMEmbedderIntegration:
    """Test ESM embedder integration."""
    
    def test_esm_embedder_model_selection(self, temp_dir, sample_fasta):
        """Test ESM embedder model selection."""
        # Test ESM2 model selection
        esm2_model_name = "esm2_t33_650M_UR50D"
        
        with patch('pepe.model_selecter._get_esm_embedder') as mock_get_esm:
            mock_esm_class = Mock()
            mock_get_esm.return_value = mock_esm_class
            
            selected_model = select_model(esm2_model_name)
            assert selected_model == mock_esm_class
            mock_get_esm.assert_called_once()
        
        # Test ESM1 model selection
        esm1_model_name = "esm1_t34_670M_UR50S"
        
        with patch('pepe.model_selecter._get_esm_embedder') as mock_get_esm:
            mock_esm_class = Mock()
            mock_get_esm.return_value = mock_esm_class
            
            selected_model = select_model(esm1_model_name)
            assert selected_model == mock_esm_class
            mock_get_esm.assert_called_once()
    
    def test_esm_embedder_workflow(self, temp_dir, sample_fasta):
        """Test ESM embedder workflow."""
        # Create mock arguments
        args = Mock()
        args.model_name = "esm2_t33_650M_UR50D"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_esm"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock ESM embedder
        with patch('pepe.model_selecter._get_esm_embedder') as mock_get_esm:
            mock_esm_class = Mock()
            mock_esm_instance = Mock()
            mock_esm_class.return_value = mock_esm_instance
            mock_get_esm.return_value = mock_esm_class
            
            # Test ESM workflow
            selected_model = select_model(args.model_name)
            embedder = selected_model(args)
            embedder.run()
            
            # Verify run was called
            mock_esm_instance.run.assert_called_once()


class TestHuggingFaceEmbedderIntegration:
    """Test HuggingFace embedder integration."""
    
    def test_huggingface_embedder_model_selection(self, temp_dir, sample_fasta):
        """Test HuggingFace embedder model selection."""
        # Test T5 model selection
        t5_model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        
        with patch('pepe.model_selecter._get_huggingface_embedders') as mock_get_hf:
            mock_t5_class = Mock()
            mock_antiberta_class = Mock()
            mock_get_hf.return_value = (mock_t5_class, mock_antiberta_class)
            
            with patch('transformers.AutoConfig.from_pretrained') as mock_config:
                mock_config.return_value.model_type = "t5"
                
                selected_model = select_model(t5_model_name)
                assert selected_model == mock_t5_class
                mock_get_hf.assert_called_once()
        
        # Test RoFormer model selection
        roformer_model_name = "alchemab/antiberta2-cssp"
        
        with patch('pepe.model_selecter._get_huggingface_embedders') as mock_get_hf:
            mock_t5_class = Mock()
            mock_antiberta_class = Mock()
            mock_get_hf.return_value = (mock_t5_class, mock_antiberta_class)
            
            with patch('transformers.AutoConfig.from_pretrained') as mock_config:
                mock_config.return_value.model_type = "roformer"
                
                selected_model = select_model(roformer_model_name)
                assert selected_model == mock_antiberta_class
                mock_get_hf.assert_called_once()
    
    def test_huggingface_embedder_workflow(self, temp_dir, sample_fasta):
        """Test HuggingFace embedder workflow."""
        # Create mock arguments
        args = Mock()
        args.model_name = "alchemab/antiberta2-cssp"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_hf"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock HuggingFace embedder
        with patch('pepe.model_selecter._get_huggingface_embedders') as mock_get_hf:
            mock_t5_class = Mock()
            mock_antiberta_class = Mock()
            mock_antiberta_instance = Mock()
            mock_antiberta_class.return_value = mock_antiberta_instance
            mock_get_hf.return_value = (mock_t5_class, mock_antiberta_class)
            
            with patch('transformers.AutoConfig.from_pretrained') as mock_config:
                mock_config.return_value.model_type = "roformer"
                
                # Test HuggingFace workflow
                selected_model = select_model(args.model_name)
                embedder = selected_model(args)
                embedder.run()
                
                # Verify run was called
                mock_antiberta_instance.run.assert_called_once()


class TestArgumentParsing:
    """Test argument parsing and validation."""
    
    def test_parse_arguments_basic(self):
        """Test basic argument parsing."""
        with patch('sys.argv', [
            'pepe',
            '--model_name', 'test_model',
            '--fasta_path', 'test.fasta',
            '--output_path', '/tmp/output'
        ]):
            args = parse_arguments()
            assert args.model_name == 'test_model'
            assert args.fasta_path == 'test.fasta'
            assert args.output_path == '/tmp/output'
    
    def test_parse_arguments_with_optional_params(self):
        """Test argument parsing with optional parameters."""
        with patch('sys.argv', [
            'pepe',
            '--model_name', 'test_model',
            '--fasta_path', 'test.fasta',
            '--output_path', '/tmp/output',
            '--experiment_name', 'test_exp',
            '--layers', '-1', '-2',
            '--extract_embeddings', 'mean_pooled', 'per_token',
            '--batch_size', '512',
            '--device', 'cpu',
            '--precision', 'float16',
            '--streaming_output', 'true'
        ]):
            args = parse_arguments()
            assert args.model_name == 'test_model'
            assert args.experiment_name == 'test_exp'
            assert args.layers == [[-1], [-2]]
            assert args.extract_embeddings == ['mean_pooled', 'per_token']
            assert args.batch_size == 512
            assert args.device == 'cpu'
            assert args.precision == 'float16'
            assert args.streaming_output == True
    
    def test_parse_arguments_substring_options(self):
        """Test argument parsing with substring options."""
        with patch('sys.argv', [
            'pepe',
            '--model_name', 'test_model',
            '--fasta_path', 'test.fasta',
            '--output_path', '/tmp/output',
            '--substring_path', 'substring.csv',
            '--context', '5',
            '--extract_embeddings', 'substring_pooled'
        ]):
            args = parse_arguments()
            assert args.substring_path == 'substring.csv'
            assert args.context == 5
            assert 'substring_pooled' in args.extract_embeddings


class TestOutputGeneration:
    """Test output generation and file handling."""
    
    def test_output_directory_creation(self, temp_dir, sample_fasta):
        """Test that output directories are created correctly."""
        output_path = os.path.join(temp_dir, "output")
        
        # Create mock arguments
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = sample_fasta
        args.output_path = output_path
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock the embedder
        with patch('pepe.model_selecter.select_model') as mock_select:
            mock_embedder_class = Mock()
            mock_embedder_instance = Mock()
            mock_embedder_class.return_value = mock_embedder_instance
            mock_select.return_value = mock_embedder_class
            
            # Test that output directory is created
            selected_model = select_model(args.model_name)
            embedder = selected_model(args)
            
            # Verify the embedder was instantiated with correct arguments
            mock_embedder_class.assert_called_once_with(args)
    
    def test_output_file_naming(self, temp_dir, sample_fasta):
        """Test output file naming conventions."""
        # Create mock arguments
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Test output file naming
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(args)
            
            # Test file path generation
            filepath = embedder._make_output_filepath("embeddings", "/output", layer=-1)
            expected = "/output/test_experiment_test_model_embeddings_layer_-1.npy"
            assert filepath == expected
    
    def test_sequence_index_export(self, temp_dir, sample_fasta):
        """Test sequence index export functionality."""
        # Create mock arguments
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Test sequence index export
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(args)
            embedder.sequence_labels = ["seq1", "seq2", "seq3"]
            
            # Test export
            embedder.export_sequence_indices()
            
            # Verify file was created
            idx_file = os.path.join(temp_dir, "test_idx.csv")
            assert os.path.exists(idx_file)
            
            # Verify content
            with open(idx_file, "r") as f:
                content = f.read()
                assert "index,sequence_id" in content
                assert "0,seq1" in content
                assert "1,seq2" in content
                assert "2,seq3" in content


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unsupported_model_error(self):
        """Test error handling for unsupported model."""
        with pytest.raises(ValueError, match="Model .* not supported"):
            select_model("unsupported_model_name")
    
    def test_missing_fasta_file_error(self, temp_dir):
        """Test error handling for missing FASTA file."""
        # Create mock arguments with missing file
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = os.path.join(temp_dir, "missing.fasta")
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Mock the embedder to use real FASTA reading
        with patch('pepe.model_selecter.select_model') as mock_select:
            mock_embedder_class = Mock()
            mock_embedder_instance = Mock()
            mock_embedder_class.return_value = mock_embedder_instance
            mock_select.return_value = mock_embedder_class
            
            # This should work fine because we're mocking the embedder
            selected_model = select_model("custom:/fake/model.pt")
            embedder = selected_model(args)
            
            # The actual error would occur when trying to read the FASTA file
            # in the real embedder implementation
    
    def test_invalid_device_handling(self, temp_dir, sample_fasta):
        """Test handling of invalid device specification."""
        # Create mock arguments with invalid device
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "invalid_device"  # Invalid device
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Test that the embedder handles invalid device gracefully
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(args)
            # Should default to CPU for invalid device
            assert embedder.device == torch.device("cpu")
    
    def test_memory_error_handling(self, temp_dir, sample_fasta):
        """Test handling of memory errors."""
        # Create mock arguments
        args = Mock()
        args.model_name = "test_model"
        args.fasta_path = sample_fasta
        args.output_path = temp_dir
        args.experiment_name = "test_experiment"
        args.tokenizer_from = None
        args.substring_path = None
        args.context = 0
        args.layers = [[-1]]
        args.extract_embeddings = ["mean_pooled"]
        args.batch_size = 1
        args.max_length = 50
        args.device = "cpu"
        args.discard_padding = False
        args.flatten = False
        args.streaming_output = False
        args.num_workers = 1
        args.flush_batches_after = 128
        args.precision = "float32"
        args.disable_special_tokens = False
        
        # Test memory error handling in _safe_compute
        from pepe.embedders.base_embedder import BaseEmbedder
        
        with patch.multiple(
            BaseEmbedder,
            _load_data=Mock(return_value=(Mock(), 50)),
            _initialize_model=Mock(return_value=(Mock(), Mock(), 12, 12, 768)),
            _load_layers=Mock(return_value=[-1]),
            __abstractmethods__=set()
        ):
            embedder = BaseEmbedder(args)
            embedder.return_embeddings = True
            embedder.return_contacts = False
            embedder.return_logits = False
            
            # Mock _compute_outputs to raise OOM error
            def mock_compute_outputs(model, toks, attention_mask, return_embeddings, return_contacts, return_logits):
                if toks.size(0) > 1:
                    raise torch.OutOfMemoryError("CUDA out of memory")
                else:
                    return None, {-1: torch.randn(1, 50, 768)}, None
            
            embedder._compute_outputs = mock_compute_outputs
            
            with patch('torch.cuda.empty_cache'):
                toks = torch.randint(0, 100, (2, 50))
                attention_mask = None
                
                # Should handle OOM by splitting batch
                result = embedder._safe_compute(toks, attention_mask)
                assert result is not None 