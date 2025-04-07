from embedairr.embedders.esm_embedder import ESMEmbedder
from embedairr.embedders.huggingface_embedder import T5Embedder, Antiberta2Embedder


def select_model(model_name):
    if "esm2" in model_name.lower():
        return ESMEmbedder
    elif "esm1" in model_name.lower():
        return ESMEmbedder
    elif "antiberta2" in model_name.lower() and model_name.startswith("alchemab"):
        return Antiberta2Embedder
    elif "t5" in model_name.lower() and model_name.startswith("Rostlab"):
        return T5Embedder
    else:
        raise ValueError(f"Model {model_name} not supported")


supported_models = [
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
    "Rostlab/prot_t5_xl_half_uniref50-enc",
    "Rostlab/ProstT5",
    "alchemab/antiberta2-cssp",
    "alchemab/antiberta2",
]
