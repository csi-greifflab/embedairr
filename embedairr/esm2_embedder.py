import torch
from esm import FastaBatchedDataset, pretrained
from embedairr.base_embedder import BaseEmbedder


class ESM2Embedder(BaseEmbedder):
    def __init__(
        self,
        fasta_path,
        model_name,
        output_path,
        cdr3_path,
        context,
        layers,
        output_types,
    ):
        super().__init__(
            fasta_path,
            model_name,
            output_path,
            cdr3_path,
            context,
            layers,
            output_types,
        )

        self.sequences = self.fasta_to_dict(fasta_path)
        self.model, self.alphabet, self.num_heads = self.initialize_model()
        self.layers = self.load_layers(layers)
        self.data_loader = self.load_data()
        self.set_output_objects()

    def initialize_model(self, model_name="esm2_t33_650M_UR50D"):
        """Initialize the model, tokenizer"""
        #  Loading the pretrained model and alphabet for tokenization
        print("Loading model...")
        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
        model.eval()  # Setting the model to evaluation mode
        num_heads = model.layers[0].self_attn.num_heads

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        return model, alphabet, num_heads

    def load_layers(self, layers):
        # Checking if the specified representation layers are valid
        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers for i in layers
        )
        layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1)
            for i in layers
        ]
        return layers

    def load_data(self, batch_size=30000):
        print("Loading and batching input sequences...")
        # Creating a dataset from the input fasta file
        dataset = FastaBatchedDataset.from_file(self.fasta_path)
        # Generating batch indices based on token count
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        print(f"Read {self.fasta_path} with {len(dataset)} sequences")
        return data_loader

    def extract_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_all_heads = {
            layer: {
                self.attention_matrices[layer][head].extend(
                    [
                        out["attentions"][i, layer, head, 1:-1, 1:-1].cpu()
                        for i in range(len(batch_labels))
                    ]
                )
                for head in range(self.num_heads)
            }
            for layer in self.layers
        }

    def extract_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_layers = {
            layer: self.attention_matrices_average_layers[layer].extend(
                [
                    out["attentions"][i, layer, :, 1:-1, 1:-1].mean(0).cpu()
                    for i in range(len(batch_labels))
                ]
            )
            for layer in self.layers
        }

    def extract_attention_matrices_average_all(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_all.extend(
            [
                out["attentions"][i, :, :, 1:-1, 1:-1].mean(dim=(0, 1)).cpu()
                for i in range(len(batch_labels))
            ]
        )

    def extract_cdr3_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_all_heads = {
                layer: {
                    self.cdr3_attention_matrices_all_heads[layer][head].extend(
                        [out["attentions"][i, layer, head, start:end, start:end].cpu()]
                    )
                    for head in range(self.num_heads)
                }
                for layer in self.layers
            }

    def extract_cdr3_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_average_layers = {
                layer: self.cdr3_attention_matrices_average_layers[layer].extend(
                    [out["attentions"][i, layer, :, start:end, start:end].mean(0).cpu()]
                )
                for layer in self.layers
            }

    def extract_cdr3_attention_matrices_average_all(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_average_all.extend(
                [
                    out["attentions"][i, :, :, start:end, start:end]
                    .mean(dim=(0, 1))
                    .cpu()
                ]
            )

    def embed(self):
        with torch.no_grad():
            for labels, strs, toks in self.data_loader:
                if self.device == torch.device("cuda"):
                    toks = toks.to(device="cuda", non_blocking=True)
                outputs = self.model(
                    toks, repr_layers=self.layers, return_contacts=self.return_contacts
                )
                # Extracting layer representations and moving them to CPU
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in outputs["representations"].items()
                }
                self.sequence_labels.extend(labels)
                self.extract_batch(outputs, representations, labels, strs)
                # print total progress
                print(
                    f"{len(self.sequence_labels)} sequences of {len(self.sequences)} processed"
                )
        print("Finished extracting embeddings")
