"""
Unit tests for utility functions in pepe.utils module.

This module contains tests for utility functions including FASTA parsing,
token validation, batch processing, and data handling.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

import pepe.utils


class TestFastaUtils:
    """Test FASTA file processing utilities."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_fasta_to_dict_basic(self, temp_dir):
        """Test basic FASTA file parsing."""
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
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 3
        assert sequences["seq1"] == "ACDEFGHIKLMNPQRSTVWY"
        assert sequences["seq2"] == "MKVLSFGHIKLMNPQRSTVWY"
        assert sequences["seq3"] == "GKLMNPQRSTVWYACDEFHI"
    
    def test_fasta_to_dict_multiline_sequences(self, temp_dir):
        """Test FASTA parsing with multiline sequences."""
        fasta_content = """>seq1
ACDEFGHIKLMNPQRSTVWY
GKLMNPQRSTVWYACDEFHI
>seq2
MKVLSFGHIKLMNPQRSTVWY
ACDEFGHIKLMNPQRSTVWY
"""
        fasta_path = os.path.join(temp_dir, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 2
        assert sequences["seq1"] == "ACDEFGHIKLMNPQRSTVWYGKLMNPQRSTVWYACDEFHI"
        assert sequences["seq2"] == "MKVLSFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    
    def test_fasta_to_dict_with_gaps(self, temp_dir):
        """Test FASTA parsing with gaps and special characters."""
        fasta_content = """>seq1
ACDEFG-HIKLMNPQRSTVWY
>seq2
MKVLSFG.HIKLMNPQRSTVWY
>seq3
GKLMNPQRSTVWYACDEFHI*
"""
        fasta_path = os.path.join(temp_dir, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 3
        assert sequences["seq1"] == "ACDEFG-HIKLMNPQRSTVWY"
        assert sequences["seq2"] == "MKVLSFG.HIKLMNPQRSTVWY"
        assert sequences["seq3"] == "GKLMNPQRSTVWYACDEFHI*"
    
    def test_fasta_to_dict_empty_file(self, temp_dir):
        """Test FASTA parsing with empty file."""
        fasta_path = os.path.join(temp_dir, "empty.fasta")
        with open(fasta_path, "w") as f:
            f.write("")
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 0
    
    def test_fasta_to_dict_missing_file(self, temp_dir):
        """Test FASTA parsing with missing file."""
        fasta_path = os.path.join(temp_dir, "missing.fasta")
        
        with pytest.raises(FileNotFoundError):
            pepe.utils.fasta_to_dict(fasta_path)
    
    def test_fasta_to_dict_no_sequences(self, temp_dir):
        """Test FASTA parsing with file containing no sequences."""
        fasta_content = "This is not a FASTA file\nJust some text"
        fasta_path = os.path.join(temp_dir, "no_sequences.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 0
    
    def test_fasta_to_dict_header_only(self, temp_dir):
        """Test FASTA parsing with header but no sequence."""
        fasta_content = """>seq1
>seq2
ACDEFGHIKLMNPQRSTVWY
"""
        fasta_path = os.path.join(temp_dir, "header_only.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 2
        assert sequences["seq1"] == ""
        assert sequences["seq2"] == "ACDEFGHIKLMNPQRSTVWY"


class TestTokenValidation:
    """Test token validation utilities."""
    
    def test_check_input_tokens_valid(self):
        """Test token validation with valid tokens."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWY"
        }
        model_name = "test_model"
        
        # Should not raise any exception
        pepe.utils.check_input_tokens(valid_tokens, sequences, model_name)
    
    def test_check_input_tokens_invalid(self):
        """Test token validation with invalid tokens."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWYXZB"  # Contains invalid tokens X, Z, B
        }
        model_name = "test_model"
        
        # Should raise SystemExit
        with pytest.raises(SystemExit):
            pepe.utils.check_input_tokens(valid_tokens, sequences, model_name)
    
    def test_check_input_tokens_empty_sequences(self):
        """Test token validation with empty sequences."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = {}
        model_name = "test_model"
        
        # Should not raise any exception
        pepe.utils.check_input_tokens(valid_tokens, sequences, model_name)
    
    def test_check_input_tokens_empty_sequence_string(self):
        """Test token validation with empty sequence string."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = {
            "seq1": "",
            "seq2": "ACDEFGHIKLMNPQRSTVWY"
        }
        model_name = "test_model"
        
        # Should not raise any exception
        pepe.utils.check_input_tokens(valid_tokens, sequences, model_name)
    
    def test_check_input_tokens_special_characters(self):
        """Test token validation with special characters."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY-.*")
        sequences = {
            "seq1": "ACDEFG-HIKLMNPQRSTVWY",
            "seq2": "MKVLSFG.HIKLMNPQRSTVWY*"
        }
        model_name = "test_model"
        
        # Should not raise any exception
        pepe.utils.check_input_tokens(valid_tokens, sequences, model_name)


class TestBatchSampling:
    """Test batch sampling utilities."""
    
    def test_token_budget_batch_sampler_basic(self):
        """Test basic token budget batch sampler."""
        # Create mock dataset
        dataset = [
            ("seq1", "ACDEFG", [1, 2, 3, 4, 5, 6], None),
            ("seq2", "MKVLSF", [7, 8, 9, 10, 11, 12], None),
            ("seq3", "GKLMNP", [13, 14, 15, 16, 17, 18], None),
            ("seq4", "QRSTVW", [19, 20, 21, 22, 23, 24], None),
        ]
        
        # Create batch sampler with budget of 12 tokens (2 sequences of 6 tokens each)
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=12)
        
        batches = list(sampler)
        
        assert len(batches) == 2  # 4 sequences / 2 per batch = 2 batches
        assert batches[0] == [0, 1]  # First batch
        assert batches[1] == [2, 3]  # Second batch
    
    def test_token_budget_batch_sampler_odd_number(self):
        """Test batch sampler with odd number of sequences."""
        # Create mock dataset with 5 sequences
        dataset = [
            ("seq1", "ACDEFG", [1, 2, 3, 4, 5, 6], None),
            ("seq2", "MKVLSF", [7, 8, 9, 10, 11, 12], None),
            ("seq3", "GKLMNP", [13, 14, 15, 16, 17, 18], None),
            ("seq4", "QRSTVW", [19, 20, 21, 22, 23, 24], None),
            ("seq5", "YACDEF", [25, 26, 27, 28, 29, 30], None),
        ]
        
        # Create batch sampler with budget of 12 tokens (2 sequences of 6 tokens each)
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=12)
        
        batches = list(sampler)
        
        assert len(batches) == 3  # 5 sequences / 2 per batch = 3 batches (last has 1 sequence)
        assert batches[0] == [0, 1]  # First batch
        assert batches[1] == [2, 3]  # Second batch
        assert batches[2] == [4]     # Third batch (incomplete)
    
    def test_token_budget_batch_sampler_single_sequence(self):
        """Test batch sampler with single sequence per batch."""
        # Create mock dataset
        dataset = [
            ("seq1", "ACDEFGHIKLMNPQRSTVWY", [1] * 20, None),
            ("seq2", "MKVLSFGHIKLMNPQRSTVWY", [2] * 20, None),
        ]
        
        # Create batch sampler with budget of 20 tokens (1 sequence of 20 tokens)
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=20)
        
        batches = list(sampler)
        
        assert len(batches) == 2  # 2 sequences / 1 per batch = 2 batches
        assert batches[0] == [0]  # First batch
        assert batches[1] == [1]  # Second batch
    
    def test_token_budget_batch_sampler_large_budget(self):
        """Test batch sampler with large budget."""
        # Create mock dataset
        dataset = [
            ("seq1", "ACDEFG", [1, 2, 3, 4, 5, 6], None),
            ("seq2", "MKVLSF", [7, 8, 9, 10, 11, 12], None),
            ("seq3", "GKLMNP", [13, 14, 15, 16, 17, 18], None),
        ]
        
        # Create batch sampler with budget of 100 tokens (much larger than needed)
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=100)
        
        batches = list(sampler)
        
        assert len(batches) == 1  # All sequences fit in one batch
        assert batches[0] == [0, 1, 2]  # All sequences in first batch


class TestSequenceDataset:
    """Test sequence dataset utilities."""
    
    def test_sequence_dict_dataset_basic(self):
        """Test basic sequence dataset creation."""
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWY"
        }
        substring_dict = None
        context = 0
        
        dataset = pepe.utils.SequenceDictDataset(sequences, substring_dict, context)
        
        assert len(dataset) == 2
        assert dataset.data[0] == ("seq1", "ACDEFGHIKLMNPQRSTVWY")
        assert dataset.data[1] == ("seq2", "MKVLSFGHIKLMNPQRSTVWY")
    
    def test_sequence_dict_dataset_with_substring(self):
        """Test sequence dataset with substring dictionary."""
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWY"
        }
        substring_dict = {
            "seq1": "DEFGH",
            "seq2": "VLSFG"
        }
        context = 0
        
        dataset = pepe.utils.SequenceDictDataset(sequences, substring_dict, context)
        
        assert len(dataset) == 2
        assert dataset.substring_dict == substring_dict
        assert dataset.context == context
    
    def test_sequence_dict_dataset_indexing(self):
        """Test sequence dataset indexing."""
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWY"
        }
        substring_dict = None
        context = 0
        
        dataset = pepe.utils.SequenceDictDataset(sequences, substring_dict, context)
        
        # Test direct indexing
        assert dataset[0] == ("seq1", "ACDEFGHIKLMNPQRSTVWY")
        assert dataset[1] == ("seq2", "MKVLSFGHIKLMNPQRSTVWY")
    
    def test_sequence_dict_dataset_empty(self):
        """Test sequence dataset with empty sequences."""
        sequences = {}
        substring_dict = None
        context = 0
        
        dataset = pepe.utils.SequenceDictDataset(sequences, substring_dict, context)
        
        assert len(dataset) == 0
        assert dataset.data == []


class TestDiskSpaceCheck:
    """Test disk space checking utilities."""
    
    def test_check_disk_free_space_sufficient(self):
        """Test disk space check with sufficient space."""
        # Most systems should have more than 1KB free space
        required_bytes = 1024  # 1KB
        output_path = "/tmp"
        
        # Should not raise any exception
        pepe.utils.check_disk_free_space(output_path, required_bytes)
    
    def test_check_disk_free_space_insufficient(self):
        """Test disk space check with insufficient space."""
        # Use an extremely large number that no system would have
        required_bytes = 10**18  # 1 exabyte
        output_path = "/tmp"
        
        with pytest.raises(SystemExit):
            pepe.utils.check_disk_free_space(output_path, required_bytes)
    
    def test_check_disk_free_space_nonexistent_path(self):
        """Test disk space check with nonexistent path."""
        required_bytes = 1024
        output_path = "/nonexistent/path/that/does/not/exist"
        
        # Should handle the error gracefully
        with pytest.raises((FileNotFoundError, OSError)):
            pepe.utils.check_disk_free_space(output_path, required_bytes)


class TestMultiIODispatcher:
    """Test multi-threaded I/O dispatcher utilities."""
    
    def test_multi_io_dispatcher_initialization(self):
        """Test MultiIODispatcher initialization."""
        memmap_registry = {}
        flush_bytes_limit = 1024 * 1024  # 1MB
        heavy_output_type = "per_token"
        checkpoint_dir = "/tmp"
        
        dispatcher = pepe.utils.MultiIODispatcher(
            memmap_registry, flush_bytes_limit, heavy_output_type, checkpoint_dir
        )
        
        assert dispatcher.memmap_registry == memmap_registry
        assert dispatcher.flush_bytes_limit == flush_bytes_limit
        assert dispatcher.heavy_output_type == heavy_output_type
        assert dispatcher.checkpoint_dir == checkpoint_dir
    
    def test_multi_io_dispatcher_enqueue(self):
        """Test MultiIODispatcher enqueue functionality."""
        memmap_registry = {}
        flush_bytes_limit = 1024 * 1024  # 1MB
        heavy_output_type = "per_token"
        checkpoint_dir = "/tmp"
        
        dispatcher = pepe.utils.MultiIODispatcher(
            memmap_registry, flush_bytes_limit, heavy_output_type, checkpoint_dir
        )
        
        # Test enqueue with mock data
        output_type = "embeddings"
        layer = -1
        head = None
        offset = 0
        array = np.random.randn(10, 768).astype(np.float32)
        
        # Should not raise any exception
        dispatcher.enqueue(output_type, layer, head, offset, array)
    
    def test_multi_io_dispatcher_queue_fullness(self):
        """Test MultiIODispatcher queue fullness calculation."""
        memmap_registry = {}
        flush_bytes_limit = 1024 * 1024  # 1MB
        heavy_output_type = "per_token"
        checkpoint_dir = "/tmp"
        
        dispatcher = pepe.utils.MultiIODispatcher(
            memmap_registry, flush_bytes_limit, heavy_output_type, checkpoint_dir
        )
        
        # Initially should be empty
        fullness = dispatcher.queue_fullness()
        assert fullness >= 0.0
        assert fullness <= 1.0
    
    def test_multi_io_dispatcher_stop(self):
        """Test MultiIODispatcher stop functionality."""
        memmap_registry = {}
        flush_bytes_limit = 1024 * 1024  # 1MB
        heavy_output_type = "per_token"
        checkpoint_dir = "/tmp"
        
        dispatcher = pepe.utils.MultiIODispatcher(
            memmap_registry, flush_bytes_limit, heavy_output_type, checkpoint_dir
        )
        
        # Should not raise any exception
        dispatcher.stop()
    
    def test_multi_io_dispatcher_resume_info(self):
        """Test MultiIODispatcher resume info functionality."""
        memmap_registry = {}
        flush_bytes_limit = 1024 * 1024  # 1MB
        heavy_output_type = "per_token"
        checkpoint_dir = "/tmp"
        
        dispatcher = pepe.utils.MultiIODispatcher(
            memmap_registry, flush_bytes_limit, heavy_output_type, checkpoint_dir
        )
        
        # Should return None or resume information
        resume_info = dispatcher.get_resume_info()
        assert resume_info is None or isinstance(resume_info, dict)


class TestUtilityEdgeCases:
    """Test edge cases and error conditions in utilities."""
    
    def test_fasta_to_dict_large_file(self, temp_dir):
        """Test FASTA parsing with large file."""
        # Create a large FASTA file
        fasta_path = os.path.join(temp_dir, "large.fasta")
        with open(fasta_path, "w") as f:
            for i in range(1000):
                f.write(f">seq{i}\n")
                f.write("ACDEFGHIKLMNPQRSTVWY" * 10 + "\n")
        
        sequences = pepe.utils.fasta_to_dict(fasta_path)
        
        assert len(sequences) == 1000
        assert all(len(seq) == 200 for seq in sequences.values())
    
    def test_batch_sampler_edge_cases(self):
        """Test batch sampler with edge cases."""
        # Empty dataset
        dataset = []
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=100)
        batches = list(sampler)
        assert len(batches) == 0
        
        # Single sequence dataset
        dataset = [("seq1", "ACDEFG", [1, 2, 3, 4, 5, 6], None)]
        sampler = pepe.utils.TokenBudgetBatchSampler(dataset, token_budget=100)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0]
    
    def test_sequence_dataset_iteration(self):
        """Test sequence dataset iteration."""
        sequences = {
            "seq1": "ACDEFGHIKLMNPQRSTVWY",
            "seq2": "MKVLSFGHIKLMNPQRSTVWY",
            "seq3": "GKLMNPQRSTVWYACDEFHI"
        }
        
        dataset = pepe.utils.SequenceDictDataset(sequences, None, 0)
        
        # Test iteration
        items = list(dataset)
        assert len(items) == 3
        assert items[0] == ("seq1", "ACDEFGHIKLMNPQRSTVWY")
        assert items[1] == ("seq2", "MKVLSFGHIKLMNPQRSTVWY")
        assert items[2] == ("seq3", "GKLMNPQRSTVWYACDEFHI")
    
    def test_check_input_tokens_case_sensitivity(self):
        """Test token validation case sensitivity."""
        valid_tokens = set("ACDEFGHIKLMNPQRSTVWY")
        sequences = {
            "seq1": "acdefghiklmnpqrstvwy",  # lowercase
            "seq2": "ACDEFGHIKLMNPQRSTVWY"   # uppercase
        }
        model_name = "test_model"
        
        # Should raise SystemExit for lowercase tokens
        with pytest.raises(SystemExit):
            pepe.utils.check_input_tokens(valid_tokens, sequences, model_name) 