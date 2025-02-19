def test():
    from embedairr.embedairr import parse_arguments
    from embedairr.model_selecter import select_model
    import sys

    sys.argv = [
        "embedairr.py",
        "--model_name",
        "ab2",
        "--fasta_path",
        "/doctorai/userdata/airr_atlas/data/sequences/test_500.fa",
        "--output_path",
        "data/embeddings/test/",
        "--layers",
        "-1",
        "--extract_embeddings",
        "pooled",
        "unpooled",
        "--extract_attention_matrices",
        "average_all",
        "average_layer",
        "all_heads",
    ]

    args = parse_arguments()
    model_name = args.model_name
    fasta_path = args.fasta_path
    output_path = args.output_path
    cdr3_path = args.cdr3_path
    context = args.context
    layers = list(map(int, args.layers.strip().split()))

    embedder = select_model(model_name)

    embedder = embedder(
        fasta_path, model_name, output_path, cdr3_path, context, layers, output_types
    )

    embedder.run()


test()

import torch

with torch.no_grad():
    for labels, strs, toks in embedder.data_loader:
        if embedder.device == torch.device("cuda"):
            toks = toks.to(device="cuda", non_blocking=True)
        outputs = embedder.model(
            toks, repr_layers=embedder.layers, return_contacts=embedder.return_contacts
        )
        # Extracting layer representations and moving them to CPU
        representations = {
            layer: t.to(device="cpu") for layer, t in outputs["representations"].items()
        }
        embedder.sequence_labels.extend(labels)
        embedder.extract_batch(outputs, representations, labels, strs)
        # print total progress
        print(
            f"{len(embedder.sequence_labels)} sequences of {len(embedder.sequences)} processed"
        )
        break

embedder.attention_matrices_average_layers
embedder.attention_matrices_average_layers = {
    layer: (
        embedder.attention_matrices_average_layers[layer]
        + [
            outputs["attentions"][i, layer - 1, :, 1:-1, 1:-1].mean(0).cpu()
            for i in range(len(labels))
        ]
    )
    for layer in range(1, embedder.num_layers + 1)
}
