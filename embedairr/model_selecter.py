from embedairr.esm2_embedder import ESM2Embedder


def select_model(model_name):
    if model_name in ["esm2_t33_650M_UR50D", "esm2"]:
        return ESM2Embedder
    else:
        raise ValueError(f"Model {model_name} not supported")
