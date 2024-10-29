from embedairr.esm2_embedder import ESM2Embedder
from embedairr.antiberta2_embedder import Antiberta2Embedder


def select_model(model_name):
    embedding_models = {
        "esm2_t33_650m_UR50d": ESM2Embedder,
        "esm2": ESM2Embedder,
        "ab2": Antiberta2Embedder,
        "antiberta2": Antiberta2Embedder,
    }
    if model_name.lower() in embedding_models:
        return embedding_models[model_name]
    else:
        raise ValueError(f"Model {model_name} not supported")
