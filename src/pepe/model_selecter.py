from pepe.embedders.custom_embedder import CustomEmbedder
from pepe.embedders.huggingface_embedder import HuggingFaceEmbedder
from transformers import AutoConfig
import os

class ModelSelecter:
    def __init__(self):
        pass

    def get_embedder(self, model_name):
        """
        Get the appropriate embedder for a given model name.
        
        Args:
            model_name (str): The name/path of the model
            
        Returns:
            An embedder instance
        """
        # Check if it's a local model path
        if os.path.exists(model_name):
            return CustomEmbedder(model_name)
        
        # For now, we'll use HuggingFaceEmbedder for all HuggingFace models
        # TODO: Add support for other embedders when they're available
        
        # For HuggingFace models, check the model type and use appropriate embedder
        try:
            # Try to get model config to determine model type
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model_type = getattr(config, 'model_type', None)
            
            # Handle specific model types
            if model_type in ['bert', 'roberta', 'distilbert', 'albert', 'electra', 'deberta', 'deberta-v2']:
                return HuggingFaceEmbedder(model_name)
            elif model_type in ['gpt2', 'gpt_neo', 'gpt_neox', 'llama', 'mistral', 'phi', 'gemma', 'qwen2']:
                return HuggingFaceEmbedder(model_name)
            elif model_type in ['t5', 'mt5', 'ul2', 'flan-t5']:
                return HuggingFaceEmbedder(model_name)
            else:
                # Fallback: attempt to use HuggingFaceEmbedder for unknown model types
                # This will let the HuggingFaceEmbedder try to infer the model loading
                print(f"Unknown model type '{model_type}' for model '{model_name}'. Attempting fallback with HuggingFaceEmbedder.")
                return HuggingFaceEmbedder(model_name)
                
        except Exception as e:
            print(f"Could not determine model type for '{model_name}': {e}")
            # Final fallback: try HuggingFaceEmbedder anyway
            return HuggingFaceEmbedder(model_name)
    
    def _is_sentence_transformer(self, model_name):
        """Check if a model is a sentence transformer model"""
        sentence_transformer_indicators = [
            'sentence-transformers/',
            'all-MiniLM',
            'all-mpnet',
            'multi-qa',
            'msmarco',
            'paraphrase',
        ]
        
        return any(indicator in model_name for indicator in sentence_transformer_indicators)
