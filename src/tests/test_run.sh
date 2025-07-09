pepe \
    --experiment_name "test" \
    --model_name "/home/jahn/embedairr/examples/custom_model/example_protein_model" \
    --fasta_path "src/tests/test_files/test.fasta" \
    --output_path "src/tests/test_files/test_output" \
    --substring_path "src/tests/test_files/test_cdr3.csv" \
    --extract_embeddings per_token mean_pooled substring_pooled attention_head \
    --batch_writing true \
    --device cpu