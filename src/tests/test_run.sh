#!/usr/bin/env bash
pepe \
    --experiment_name "test" \
    --model_name "examples/custom_model/example_protein_model" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --substring_path "src/tests/test_files/test_substring.csv" \
    --extract_embeddings "per_token" mean_pooled substring_pooled attention_head \
    --streaming_output true \
    --device cpu \
    --layers -2 -1