from embedairr.esm_embedder import ESM1Embedder, ESM2Embedder
from embedairr.antiberta2_embedder import Antiberta2Embedder
from embedairr.prott5_embedder import T5Embedder


def select_model(model_name):
    if "esm2" in model_name.lower():
        return ESM2Embedder
    elif "esm1" in model_name.lower():
        return ESM1Embedder
    elif "antiberta2" in model_name.lower() and model_name.startswith("alchemab"):
        return Antiberta2Embedder
    elif "t5" in model_name.lower() and model_name.startswith("Rostlab"):
        return T5Embedder
    else:
        raise ValueError(f"Model {model_name} not supported")
