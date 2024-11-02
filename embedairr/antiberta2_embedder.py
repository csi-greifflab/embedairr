import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
from embedairr.base_embedder import BaseEmbedder


class Antiberta2Embedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = self.fasta_to_dict(args.fasta_path, gaps=True)
        self.model, self.tokenizer, self.num_heads, self.num_layers = (
            self.initialize_model("alchemab/antiberta2-cssp")
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data(self.max_length, self.batch_size)
        self.sequences = {
            sequence_id: sequence_aa.replace(" ", "")
            for sequence_id, sequence_aa in self.sequences.items()
        }
        self.set_output_objects()

    def initialize_model(self, model_name="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def load_data(self, max_length, batch_size):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        input_ids = []
        attention_masks = []
        total_sequences = len(self.sequences)
        print("Start tokenization")
        for counter, sequence in enumerate(self.sequences.values()):
            tokens = self.tokenizer(
                sequence,
                truncation=False,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True,
                max_length=max_length,
            )
            input_ids.append(tokens["input_ids"])
            attention_masks.append(tokens["attention_mask"])
            # Calculate and print the percentage of completion
            percent_complete = ((counter + 1) / total_sequences) * 100
            # Check and print the progress at each 2% interval
            if (counter + 1) == total_sequences or int(percent_complete) % 2 == 0:
                # Ensures the message is printed once per interval and at 100% completion
                if (counter + 1) == total_sequences or (
                    int(percent_complete / 2)
                    != int(((counter) / total_sequences) * 100 / 2)
                ):
                    print(f"Progress: {percent_complete:.2f}%", end="\r")

        # Convert lists to tensors and create a dataset
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader

    def extract_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_all_heads = {
            layer: {
                head: (
                    self.attention_matrices_all_heads[layer][head]
                    + [
                        out.attentions[layer - 1][i, head, 1:-1, 1:-1].cpu()
                        for i in range(len(batch_labels))
                    ]
                )
                for head in range(self.num_heads)
            }
            for layer in range(1, self.num_layers + 1)
        }

    def extract_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_layers = {
            layer: (
                self.attention_matrices_average_layers[layer]
                + [
                    out.attentions[layer - 1][i, :, 1:-1, 1:-1]
                    .mean(0)
                    .cpu()  # average over heads
                    for i in range(len(batch_labels))
                ]
            )
            for layer in range(1, self.num_layers + 1)
        }

    def extract_attention_matrices_average_all(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_all + [
            torch.stack(
                [
                    out.attentions[layer][i, :, 1:-1, 1:-1].mean(
                        dim=0
                    )  # average over heads
                    for layer in range(self.num_layers)
                ]
            ).mean(
                dim=0
            )  # average over layers
            for i in range(len(batch_labels))
        ]

    def extract_cdr3_attention_matrices_all_heads(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_all_heads = {
                layer: {
                    head: (
                        self.cdr3_attention_matrices_all_heads[layer][head]
                        + [
                            out.attentions[layer - 1][
                                i, head, start:end, start:end
                            ].cpu()
                        ]
                    )
                    for head in range(self.num_heads)
                }
                for layer in range(1, self.num_layers + 1)
            }

    def extract_cdr3_attention_matrices_average_layer(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_average_layers = {
                layer: (
                    self.cdr3_attention_matrices_average_layers[layer]
                    + [
                        out.attentions[layer - 1][i, :, start:end, start:end]
                        .mean(0)
                        .cpu()
                    ]
                )
                for layer in range(1, self.num_layers + 1)
            }

    def extract_cdr3_attention_matrices_average_all(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_attention_matrices_average_all + [
                torch.stack(
                    [
                        out.attentions[layer][i, :, start:end, start:end].mean(dim=0)
                        for layer in range(self.num_layers)
                    ]
                ).mean(dim=0)
                for i in range(len(batch_labels))
            ]

    def embed(self):
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                labels = list(self.sequences.keys())[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                batch_sequences = list(self.sequences.values())[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                input_ids, attention_mask = [
                    b.to(self.device, non_blocking=True) for b in batch
                ]
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=self.return_contacts,
                )
                representations = {
                    layer: outputs.hidden_states[layer].to(device="cpu")
                    for layer in self.layers
                }
                self.sequence_labels.extend(labels)
                self.extract_batch(outputs, representations, labels, batch_sequences)
                print(
                    f"{len(self.sequence_labels)} sequences of {len(self.sequences)} processed"
                )
        print("Finished extracting embeddings")
