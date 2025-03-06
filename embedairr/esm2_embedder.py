import torch
import gc
from esm import FastaBatchedDataset, pretrained
from embedairr.base_embedder import BaseEmbedder

# torch.set_default_dtype(torch.float16)


class ESM2Embedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences, self.sequences_padded = self.fasta_to_dict(
            args.fasta_path, padding=True
        )
        self.model, self.alphabet, self.num_heads, self.num_layers = (
            self.initialize_model()
        )
        self.valid_tokens = set(self.alphabet.all_toks)
        self.check_input_tokens(self.valid_tokens, self.sequences)
        self.layers = self.load_layers(self.layers)
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
        num_layers = len(model.layers)

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print("No GPU available, using CPU")
        return model, alphabet, num_heads, num_layers

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

    def load_data(self):
        print("Loading and batching input sequences...")
        # Creating a dataset from the input fasta file
        dataset = FastaBatchedDataset(
            sequence_strs=self.sequences_padded.values(),
            sequence_labels=self.sequences_padded.keys(),
        )
        # Generating batch indices based on token count
        batches = dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        print(f"Read {self.fasta_path} with {len(dataset)} sequences")
        return data_loader

    def extract_embeddings(self, out, representations, batch_labels, batch_sequences):
        for layer in self.layers:
            self.embeddings[layer].extend(
                [
                    representations[layer][
                        i, 1 : len(batch_sequences[i].replace("<pad>", ""))
                    ].mean(0)
                    for i in range(len(batch_labels))
                ]
            )

    def extract_embeddings_unpooled(
        self, out, representations, batch_labels, batch_sequences
    ):
        if not self.discard_padding:
            for layer in self.layers:
                self.embeddings_unpooled[layer].extend(
                    [representations[layer][i][1:-1] for i in range(len(batch_labels))]
                )
        else:
            for layer in self.layers:
                self.embeddings_unpooled[layer].extend(
                    [
                        representations[layer][
                            i, 1 : len(batch_sequences[i].replace("<pad>", ""))
                        ]
                        for i in range(len(batch_labels))
                    ]
                )

    def extract_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                self.attention_matrices_all_heads[layer][head].extend(
                    [
                        out["attentions"][i, layer - 1, head, 1:-1, 1:-1].cpu()
                        for i in range(len(batch_labels))
                    ]
                )

    def extract_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        # Directly append new batch matrices to the existing lists
        for layer in self.layers:
            # Collect attention matrices for the current layer and average across heads
            # Append the new batch to the existing list
            self.attention_matrices_average_layers[layer].extend(
                [
                    out["attentions"][i, layer - 1, :, 1:-1, 1:-1]
                    .mean(0)
                    .to(torch.float16)
                    .cpu()
                    for i in range(len(batch_labels))
                ]
            )

    def extract_attention_matrices_average_all(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_all.extend(
            [
                out["attentions"][i, :, :, 1:-1, 1:-1]
                .mean(dim=(0, 1))
                .to(device="cpu", dtype=torch.float16)
                for i in range(len(batch_labels))
            ]
        )

    def extract_cdr3_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                for i, label in enumerate(batch_labels):
                    start, end = self.get_cdr3_positions(label)
                    self.cdr3_attention_matrices_all_heads[layer][head].extend(
                        [
                            out["attentions"][
                                i, layer - 1, head, start:end, start:end
                            ].cpu()
                        ]
                    )

    def extract_cdr3_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        for layer in self.layers:
            for i, label in enumerate(batch_labels):
                start, end = self.get_cdr3_positions(label)
                self.cdr3_attention_matrices_average_layers[layer].extend(
                    [
                        out["attentions"][i, layer - 1, :, start:end, start:end]
                        .mean(0)
                        .cpu()
                    ]
                )

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
                    layer: t.to(device="cpu", dtype=torch.float16)
                    for layer, t in outputs["representations"].items()
                }
                self.sequence_labels.extend(labels)
                self.extract_batch(outputs, representations, labels, strs)
                # print total progress
                print(
                    f"{self.model_name}: {len(self.sequence_labels)} sequences of {len(self.sequences)} processed",
                    end="\r",
                )
                gc.collect()

        print("Finished extracting embeddings")
