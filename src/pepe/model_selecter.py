from pepe.embedders.custom_embedder import CustomEmbedder

import os


def _is_huggingface_model(model_name):
    """Check if a model name/path refers to a HuggingFace model."""
    # Skip if it's clearly a local PyTorch file
    if model_name.endswith((".pt", ".pth")) or model_name.startswith("custom:"):
        return False
    
    # Check if it's a local directory with HuggingFace model files
    if os.path.exists(model_name) and os.path.isdir(model_name):
        # Check for common HuggingFace model files
        hf_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
        if any(os.path.exists(os.path.join(model_name, f)) for f in hf_files):
            return True
    
    # Try to load config to see if it's a valid HuggingFace model
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_name)
        return True
    except:
        return False


def _get_esm_embedder():
    """Lazy import of ESM embedder to avoid loading heavy dependencies."""
    from pepe.embedders.esm_embedder import ESMEmbedder

    return ESMEmbedder


def _get_huggingface_embedders():
    """Lazy import of HuggingFace embedders to avoid loading heavy dependencies."""
    from pepe.embedders.huggingface_embedder import T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder

    return T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder


def select_model(model_name):
    if "esm2" in model_name.lower():
        return _get_esm_embedder()
    elif "esm1" in model_name.lower():
        return _get_esm_embedder()
    # elif "antiberta2" in model_name.lower() and model_name.startswith("alchemab"):
    #    T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
    #    return Antiberta2Embedder
    # elif "t5" in model_name.lower() and model_name.startswith("Rostlab"):
    #    T5Embedder, Antiberta2Embedder = _get_huggingface_embedders()
    #    return T5Embedder
    elif (
        model_name.endswith(".pt")
        or model_name.endswith(".pth")
        or model_name.startswith("custom:")
        or (
            os.path.exists(model_name)
            and (os.path.isfile(model_name) or os.path.isdir(model_name))
        )
    ):
        return CustomEmbedder
    elif "/" in model_name or _is_huggingface_model(model_name):
        # Assume it's a Hugging Face model (username/model-name format or local model)
        # Try to determine the architecture automatically
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name)
            model_type = config.model_type.lower()

            # Use specific embedders for known architectures
            if model_type in ["t5", "mt5"]:
                T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder = _get_huggingface_embedders()
                return T5Embedder
            elif model_type in ["roformer"]:
                T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder = _get_huggingface_embedders()
                return Antiberta2Embedder
            else:
                # For all other architectures, use the generic embedder
                T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder = _get_huggingface_embedders()
                return GenericHuggingFaceEmbedder
        except Exception as e:
            # If we can't load the config, try the generic embedder as a fallback
            error_msg = str(e)
            if "Unrecognized model" in error_msg or "model_type" in error_msg:
                raise ValueError(
                    f"Model {model_name} appears to be a Keras/TensorFlow model or has an unsupported architecture. EmbedAIRR currently supports PyTorch models only. Consider using a PyTorch version or converting the model."
                )
            else:
                # Try generic embedder as fallback
                try:
                    T5Embedder, Antiberta2Embedder, GenericHuggingFaceEmbedder = _get_huggingface_embedders()
                    return GenericHuggingFaceEmbedder
                except Exception as fallback_e:
                    raise ValueError(
                        f"Could not determine model architecture for {model_name} and generic fallback failed. Original error: {e}. Fallback error: {fallback_e}"
                    )
    else:
        raise ValueError(f"Model {model_name} not supported")


supported_models = [
    # ESM models
    "esm1_t34_670M_UR50S",
    "esm1_t34_670M_UR50D",
    "esm1_t34_670M_UR100",
    "esm1_t12_85M_UR50S",
    "esm1_t6_43M_UR50S",
    "esm1b_t33_650M_UR50S",
    #'esm_msa1_t12_100M_UR50S',
    #'esm_msa1b_t12_100M_UR50S',
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
    #'esm_if1_gvp4_t16_142M_UR50',
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
    # Pre-defined Hugging Face models
    "Rostlab/prot_t5_xl_half_uniref50-enc",
    "Rostlab/ProstT5",
    "alchemab/antiberta2-cssp",
    "alchemab/antiberta2",
    # Custom models examples:
    # - PyTorch models: "/path/to/model.pt", "/path/to/model_directory/", "custom:/path/to/model.pt"
    # - Hugging Face models: "username/model-name", "./local_hf_model"
    # - See documentation for details on custom model requirements
]
