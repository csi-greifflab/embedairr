import torch
import time
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
        self.special_tokens = self.get_special_tokens()
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data()
        self.set_output_objects()

    def get_special_tokens(self):
        special_tokens = self.alphabet.all_special_tokens
        special_token_ids = torch.tensor(
            [self.alphabet.tok_to_idx[tok] for tok in special_tokens],
            device=self.device,
            dtype=torch.int8,
        )
        return special_token_ids

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
        # Creating a dataset from the input fasta file
        print("Tokenizing and batching sequences...")
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
        print("Data loaded")
        return data_loader

    def extract_embeddings(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        for layer in self.layers:
            self.embeddings[layer].extend(
                [
                    # representations[layer][
                    #    # i, 1 : len(batch_sequences[i].replace("<pad>", ""))
                    #    i, 1 : len(batch_sequences[i]) - batch_sequences[i].count("<pad>") * 5,
                    # ].mean(0)
                    (
                        (pooling_mask[i].unsqueeze(-1) * representations[layer][i]).sum(
                            0
                        )
                        / pooling_mask[i].unsqueeze(-1).sum(0)
                    ).cpu()
                    for i in range(len(batch_labels))
                ]
            )

    def extract_embeddings_unpooled(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        if not self.discard_padding:
            for layer in self.layers:
                self.embeddings_unpooled[layer].extend(
                    [
                        representations[layer][i][1:].cpu()
                        for i in range(len(batch_labels))
                    ]
                )
        else:
            for layer in self.layers:
                self.embeddings_unpooled[layer].extend(
                    [
                        representations[layer][
                            # i, 1 : len(batch_sequences[i].replace("<pad>", ""))
                            i,
                            1:,
                        ].cpu()  # remove CLS token
                        for i in range(len(batch_labels))
                    ]
                )

    def extract_attention_matrices_all_heads(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                self.attention_matrices_all_heads[layer][head].extend(
                    [
                        attention_matrices[i, layer - 1, head, 1:, 1:].to(
                            device="cpu", dtype=torch.float16
                        )  # remove CLS token
                        for i in range(len(batch_labels))
                    ]
                )

    def extract_attention_matrices_average_layer(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        # Directly append new batch matrices to the existing lists
        for layer in self.layers:
            # Collect attention matrices for the current layer and average across heads
            # Append the new batch to the existing list
            self.attention_matrices_average_layers[layer].extend(
                [
                    attention_matrices[i, layer - 1, :, 1:, 1:]  # remove CLS token
                    .mean(0)
                    .to(device="cpu", dtype=torch.float16)
                    for i in range(len(batch_labels))
                ]
            )

    def extract_attention_matrices_average_all(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        self.attention_matrices_average_all.extend(
            [
                attention_matrices[i, :, :, 1:, 1:]  # remove CLS token
                .mean(dim=(0, 1))
                .to(device="cpu", dtype=torch.float16)
                for i in range(len(batch_labels))
            ]
        )

    def extract_cdr3_attention_matrices_all_heads(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                for i, label in enumerate(batch_labels):
                    start, end = self.get_cdr3_positions(label)
                    self.cdr3_attention_matrices_all_heads[layer][head].extend(
                        [
                            attention_matrices[
                                i,
                                layer - 1,
                                head,
                                start + 1 : end + 1,
                                start + 1 : end + 1,
                            ].to(device="cpu", dtype=torch.float16)
                        ]
                    )

    def extract_cdr3_attention_matrices_average_layer(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        for layer in self.layers:
            for i, label in enumerate(batch_labels):
                start, end = self.get_cdr3_positions(label)
                self.cdr3_attention_matrices_average_layers[layer].extend(
                    [
                        attention_matrices[
                            i, layer - 1, :, start + 1 : end + 1, start + 1 : end + 1
                        ]
                        .mean(0)
                        .to(device="cpu", dtype=torch.float16)
                    ]
                )

    def extract_cdr3_attention_matrices_average_all(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_average_all.extend(
                [
                    attention_matrices[
                        i, :, :, start + 1 : end + 1, start + 1 : end + 1
                    ]
                    .mean(dim=(0, 1))
                    .to(device="cpu", dtype=torch.float16)
                ]
            )

    def embed(self):
        with torch.no_grad():
            for labels, strs, toks in self.data_loader:
                start_time = time.time()
                if self.device == torch.device("cuda"):
                    toks = toks.to(device="cuda", non_blocking=True)
                pooling_mask = self.mask_special_tokens(
                    toks, self.special_tokens
                )  # mask special tokens to avoid diluting signal when pooling embeddings
                outputs = self.model(
                    toks, repr_layers=self.layers, return_contacts=self.return_contacts
                )
                if self.return_contacts:
                    attention_matrices = outputs[
                        "attentions"
                    ]  # stack attention matrices across layers
                else:
                    attention_matrices = None
                # Extracting layer representations and moving them to CPU
                if self.return_embeddings:
                    representations = {
                        layer: t.to(dtype=torch.float16)
                        for layer, t in outputs["representations"].items()
                    }
                else:
                    representations = None
                self.sequence_labels.extend(labels)
                self.extract_batch(
                    attention_matrices, representations, labels, strs, pooling_mask
                )
                # print total progress
                end_time = time.time()
                sequences_per_second = len(labels) / (end_time - start_time)
                estimated_remaining_time = (
                    len(self.sequences) - len(self.sequence_labels)
                ) / sequences_per_second
                print(
                    f"{self.model_name}: {len(self.sequence_labels)} sequences of {len(self.sequences)} processed ({sequences_per_second:.2f} sequences/s). Estimated remaining time: {estimated_remaining_time:.2f} s"
                )

        print("Finished extracting embeddings")
