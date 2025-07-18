import sys
import os
from pepe.__main__ import parse_arguments
from pepe.model_selecter import select_model

# Set environment to non-interactive to avoid prompts
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'


def test_model(model_name, test_name):
    """Test a specific model with the given parameters."""
    print(f"\n=== Testing {test_name}: {model_name} ===")
    
    # Configure arguments similar to test_run.py
    sys.argv = [
        "pepe",
        "--experiment_name",
        f"test_{test_name}",
        "--model_name",
        model_name,
        "--fasta_path",
        "src/tests/test_files/test.fasta",
        "--output_path",
        f"src/tests/test_files/test_output_{test_name}",
        "--substring_path",
        "src/tests/test_files/test_substring.csv",
        "--extract_embeddings",
        "mean_pooled",
        "per_token",
        "--streaming_output",
        "false",
        "--device",
        "cpu",
    ]

    try:
        args = parse_arguments()
        
        # Check if output directory exists and creates it if it's missing
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        # Test model selection
        embedder_class = select_model(args.model_name)
        print(f"Selected embedder class: {embedder_class.__name__}")
        
        # Try to initialize the embedder
        embedder = embedder_class(args)
        print("Embedder initialized successfully")
        
        # Check model configuration
        if hasattr(embedder, 'model') and hasattr(embedder.model, 'config'):
            config = embedder.model.config
            print(f"Model type: {config.model_type}")
            print(f"Number of layers: {config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 'N/A'}")
            print(f"Hidden size: {config.hidden_size if hasattr(config, 'hidden_size') else 'N/A'}")
        
        # Run the embedder
        embedder.run()
        print("All outputs saved successfully.")
        
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Test the two specific models
    test_model("facebook/esm2_t33_650M_UR50D", "esm2_facebook")
    test_model("hugohrban/progen2-small", "progen2")


if __name__ == "__main__":
    main() 