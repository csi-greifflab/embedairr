import torch
from transformers import RoFormerTokenizer, RoFormerModel
from torch.utils.data import DataLoader, TensorDataset
from embedairr.base_embedder import BaseEmbedder


class Antiberta2Embedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences_gapped = self.fasta_to_dict(args.fasta_path, gaps=True)
        self.model, self.tokenizer, self.num_heads, self.num_layers = (
            self.initialize_model("alchemab/antiberta2-cssp")
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
        # print device type
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
        tokens = self.tokenizer(
            list(self.sequences_gapped.values()),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            max_length=self.max_length + 2,
        )

        # Extract input_ids and attention masks directly from the tokens
        input_ids = tokens["input_ids"]
        attention_masks = tokens["attention_mask"]

        # Create a dataset and a DataLoader
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def extract_embeddings(self, out, representations, batch_labels, batch_sequences):
        self.embeddings = {
            layer: (
                self.embeddings[layer]
                + [
                    representations[layer][i, 1 : len(batch_sequences[i]) + 1].mean(0)
                    for i in range(len(batch_labels))
                ]
            )
            for layer in self.layers
        }

    def extract_embeddings_unpooled(
        self, out, representations, batch_labels, batch_sequences
    ):
        if not self.discard_padding:
            self.embeddings_unpooled = {
                layer: (
                    self.embeddings_unpooled[layer]
                    + [
                        representations[layer][i][1:-1]
                        for i in range(len(batch_labels))
                    ]
                )
                for layer in self.layers
            }
        else:
            self.embeddings_unpooled = {
                layer: (
                    self.embeddings_unpooled[layer]
                    + [
                        representations[layer][i, 1 : len(batch_sequences[i]) + 1]
                        for i in range(len(batch_labels))
                    ]
                )
                for layer in self.layers
            }

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
