# Phase 6: Extract DataProcessor Class

## Overview
Extract data loading and preprocessing logic from BaseEmbedder into a dedicated DataProcessor class to improve separation of concerns and maintainability.

## Current Status: IN PROGRESS

## Objectives
- Extract data loading and preprocessing from BaseEmbedder
- Create a clean, testable data processing component
- Centralize FASTA loading, tokenization, and batching
- Reduce BaseEmbedder complexity

## Tasks

### 6.1 Analysis and Planning
- [x] Identify data processing methods in BaseEmbedder
- [x] Map data loading and preprocessing dependencies
- [x] Design DataProcessor class interface
- [x] Plan integration strategy

### 6.2 DataProcessor Class Implementation
- [x] Create `src/pepe/embedders/components/data_processor.py`
- [x] Implement FASTA loading and parsing
- [x] Add tokenization and batching logic
- [x] Include sequence validation and preprocessing
- [x] Add comprehensive type hints and documentation

### 6.3 Testing
- [x] Create comprehensive test suite
- [x] Test FASTA loading and parsing
- [x] Test tokenization and batching
- [x] Test sequence validation
- [x] Test error conditions and edge cases

### 6.4 Integration with BaseEmbedder
- [x] Import DataProcessor in BaseEmbedder
- [x] Modify BaseEmbedder to use DataProcessor
- [x] Remove old data processing methods from BaseEmbedder
- [x] Update component initialization

### 6.5 Validation and Cleanup
- [x] Run existing tests to ensure no regression
- [x] Verify functionality with sample data
- [x] Update documentation
- [x] Clean up unused imports

## Success Criteria
- [ ] DataProcessor class handles all data loading and preprocessing
- [ ] All existing tests pass
- [ ] BaseEmbedder is further reduced in size
- [ ] Clean separation between model logic and data processing
- [ ] No performance degradation

## Dependencies
- Phase 1: Understanding & Testing (COMPLETED)
- Phase 2: Extract Output Management (COMPLETED)
- Phase 3: Extract Embedding Processors (COMPLETED)
- Phase 4: Extract I/O Operations (COMPLETED)
- Phase 5: Extract EmbedderConfig (COMPLETED)

## Next Phase
Phase 7: Refactor BaseEmbedder to use all new components