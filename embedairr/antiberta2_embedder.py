import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
from embedairr.base_embedder import BaseEmbedder
import time
import os

# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class Antiberta2Embedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences_gapped, self.sequences = self.fasta_to_dict(
            args.fasta_path, gaps=True
        )
        self.model, self.tokenizer, self.num_heads, self.num_layers = (
            self.initialize_model("alchemab/antiberta2-cssp")
        )
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        self.check_input_tokens(self.valid_tokens, self.sequences_gapped, gaps=True)
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data()
        self.sequences = {
            sequence_id: sequence_aa.replace(" ", "")
            for sequence_id, sequence_aa in self.sequences_gapped.items()
        }
        self.set_output_objects()

    def initialize_model(self, model_name="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device type: {device}")
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name).to(device)
        model.eval()
        num_heads = model.encoder.layer[0].attention.self.num_attention_heads
        num_layers = len(model.encoder.layer)
        return model, tokenizer, num_heads, num_layers

    def load_layers(self, layers):
        """Check if the specified representation layers are valid."""
        assert all(
            -(self.model.config.num_hidden_layers + 1)
            <= i
            <= self.model.config.num_hidden_layers
            for i in layers
        )
        layers = [
            (i + self.model.config.num_hidden_layers + 1)
            % (self.model.config.num_hidden_layers + 1)
            for i in layers
        ]
        return layers

    def load_data(self):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        print("Tokenizing and batching sequences...")
        tokens = self.tokenizer(
            list(self.sequences_gapped.values()),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length + 2,  # accomodate for special tokens,
        )

        # Extract input_ids and attention masks directly from the tokens
        input_ids = tokens["input_ids"]
        attention_masks = tokens["attention_mask"]

        # Create a dataset and a DataLoader
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        print("Finished tokenizing and batching sequences")

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
                    #    i, 1 : len(batch_sequences[i].replace("[SEP]", " "))
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
                    ]  # remove CLS token
                )
        else:
            for layer in self.layers:
                self.embeddings_unpooled[layer].extend(
                    [
                        representations[layer][i, 1:].cpu()  # remove CLS token
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
                        attention_matrices[
                            layer - 1, i, head, 1:, 1:
                        ]  # remove CLS token
                        .cpu()
                        .to(dtype=torch.float16)
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
        for layer in self.layers:
            self.attention_matrices_average_layers[layer].extend(
                [
                    attention_matrices[layer - 1, i, :, 1:, 1:]  # remove CLS token
                    .mean(0)
                    .to(device="cpu", dtype=torch.float16)  # average over heads
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
                # torch.stack(
                #    [
                #        attention_matrices[layer - 1, i, :, 1:-1, 1:-1]
                #        .mean(dim=0)
                #        .to(dtype=torch.float16)  # average over heads
                #        for layer in self.layers
                #    ]
                # ).mean(
                #    dim=0
                # ) # average over layers
                attention_matrices[:, i, :]
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
                                layer - 1,
                                i,
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
                            layer - 1, i, :, start + 1 : end, start + 1 : end + 1
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
                        layer - 1, i, :, start + 1 : end + 1, start + 1 : end + 1
                    ]
                    .mean(dim=(0, 1))
                    .to(device="cpu", dtype=torch.float16)
                    for layer in self.layers
                ]
            )

    def embed(self):
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                start_time = time.time()
                print(
                    f"Start embedding batch {batch_idx + 1} of {len(self.data_loader)}"
                )
                # wait 5 seconds
                labels = list(self.sequences.keys())[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                batch_sequences = list(self.sequences.values())[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                input_ids, attention_mask = [
                    b.to(self.device, non_blocking=True) for b in batch
                ]
                pooling_mask = self.mask_special_tokens(
                    input_ids, self.special_tokens
                )  # mask special tokens to avoid diluting signal when pooling embeddings
                # print(torch.cuda.memory_summary(device=torch.cuda.current_device()))
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=self.return_embeddings,
                    output_attentions=self.return_contacts,
                )
                if self.return_contacts:
                    attention_matrices = torch.stack(
                        outputs.attentions
                    )  # stack attention matrices across layers
                else:
                    attention_matrices = None
                if self.return_embeddings:
                    representations = {
                        layer: outputs.hidden_states[layer].to(
                            # device="cpu",
                            dtype=torch.float16
                        )
                        for layer in self.layers
                    }
                else:
                    representations = None
                self.sequence_labels.extend(labels)

                self.extract_batch(
                    attention_matrices,
                    representations,
                    labels,
                    batch_sequences,
                    pooling_mask,
                )
                end_time = time.time()
                sequences_per_second = self.batch_size / (end_time - start_time)
                estimated_time_remaining = (
                    len(self.sequences) - len(self.sequence_labels)
                ) / sequences_per_second
                print(
                    f"Processed {self.model_name}: {len(self.sequence_labels)} out of {len(self.sequences)} sequences ({sequences_per_second:.2f} sequences per second). Estimated time remaining: {estimated_time_remaining:.2f} seconds."
                )
        print("Finished extracting embeddings")
