import os
import csv
import torch


class BaseEmbedder:
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
        self.fasta_path = fasta_path
        self.model_name = model_name
        self.output_path = output_path
        self.cdr3_path = cdr3_path
        self.context = context
        self.layers = layers
        self.output_types = output_types
        self.cdr3_dict = self.load_cdr3(cdr3_path)
        self.batch_size = 30000  # TODO make this an argument
        self.max_length = 200  # TODO make this an argument
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.return_contacts = False
        for output_type in output_types:
            if "attention_matrix" in output_type:
                self.return_contacts = True
        self.extraction_methods = self.set_extraction_methods(output_types)

    def set_output_objects(self):
        """Initialize output objects."""
        self.sequence_labels = []
        self.embeddings = {layer: [] for layer in self.layers}
        self.embeddings_unpooled = {layer: [] for layer in self.layers}
        self.cdr3_extracted = {layer: [] for layer in self.layers}
        self.cdr3_extracted_unpooled = {layer: [] for layer in self.layers}
        self.cdr3_context_extracted = {layer: [] for layer in self.layers}
        self.cdr3_context_extracted_unpooled = {layer: [] for layer in self.layers}
        self.attention_matrices_all_heads = {
            layer: {head: [] for head in range(self.num_heads)} for layer in self.layers
        }
        self.attention_matrices_average_layers = {layer: [] for layer in self.layers}
        self.attention_matrices_average_all = []
        self.cdr3_attention_matrices_all_heads = {
            layer: {head: [] for head in range(self.num_heads)} for layer in self.layers
        }
        self.cdr3_attention_matrices_average_layers = {
            layer: [] for layer in self.layers
        }
        self.cdr3_attention_matrices_average_all = []

    def set_extraction_methods(self):
        extraction_methods_list = []
        extraction_methods = {
            "embeddings": self.extract_embeddings,
            "embeddings_unpooled": self.extract_embeddings_unpooled,
            "cdr3_extracted": self.extract_cdr3,
            "cdr3_extracted_unpooled": self.extract_cdr3_unpooled,
            "attention_matrices_all_heads": self.extract_attention_matrices_all_heads,
            "attention_matrices_average_layer": self.extract_attention_matrices_average_layer,
            "attention_matrices_average_all": self.extract_attention_matrices_average_all,
            "cdr3_attention_matrices_all_heads": self.extract_cdr3_attention_matrices_all_heads,
            "cdr3_attention_matrices_average_layer": self.extract_cdr3_attention_matrices_average_layer,
            "cdr3_attention_matrices_average_all": self.extract_cdr3_attention_matrices_average_all,
        }
        for output_type in self.output_types:
            # if output_type contains attention_matrix, call attention_matrix extraction method
            if output_type in extraction_methods:
                extraction_methods_list.append(extraction_methods[output_type])
            else:
                raise ValueError(f"Output type {output_type} not supported")
        return extraction_methods_list

    def fasta_to_dict(self, fasta_path, gaps=False):
        """Convert FASTA file into a dictionary."""
        print("Loading and batching input sequences...")

        seq_dict = dict()
        sequence_id = None
        sequence_aa = []

        def _flush_current_seq():
            nonlocal sequence_id, sequence_aa
            if sequence_id is None:
                return
            seq = "".join(sequence_aa)
            if gaps:
                seq_dict[sequence_id] = " ".join(seq)
            else:
                seq_dict[sequence_id] = seq
            sequence_id = None
            sequence_aa = []

        with open(fasta_path, "r") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line.startswith(">"):
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        sequence_id = line
                    else:
                        sequence_id = f"seqnum{line_idx:09d}"
                else:
                    sequence_aa.append(line)

        _flush_current_seq()

        return seq_dict

    def load_cdr3(self, cdr3_path):
        """Load CDR3 sequences and store in a dictionary."""
        if cdr3_path:
            with open(cdr3_path) as f:
                reader = csv.reader(f)
                cdr3_dict = {rows[0]: rows[1] for rows in reader}
            return cdr3_dict
        else:
            return None

    def embed(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def load_data(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def initialize_model(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def load_layers(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_cdr3_positions(self, label, context=0):
        """Get the start and end positions of the CDR3 sequence in the full sequence."""
        full_sequence = self.sequences[label]
        try:
            cdr3_sequence = self.cdr3_dict[label]
        except KeyError:
            SystemExit(f"No cdr3 sequence found for {label}")

        # remove '-' from cdr3_sequence
        cdr3_sequence = cdr3_sequence.replace("-", "")

        # get position of cdr3_sequence in sequence
        start = max(full_sequence.find(cdr3_sequence) - context, 0)
        end = min(start + len(cdr3_sequence) + context, len(full_sequence))
        return start, end

    def extract_batch(self, out, representations, batch_labels, batch_sequences):
        for method in self.extraction_methods:
            method(out, representations, batch_labels, batch_sequences)

    def extract_embeddings(self, out, representations, batch_labels, batch_sequences):
        self.embeddings = {
            layer: self.embeddings[layer].extend(
                [
                    representations[layer][i, 1 : len(batch_sequences[i]) + 1].mean(0)
                    for i in range(len(batch_labels))
                ]
            )
            for layer in self.layers
        }

    def extract_embeddings_unpooled(
        self, out, representations, batch_labels, batch_sequences
    ):
        self.embeddings_unpooled = {
            layer: self.embeddings_unpooled[layer].extend(
                [
                    representations[layer][i, 1 : len(batch_sequences[i]) + 1]
                    for i in range(len(batch_labels))
                ]
            )
            for layer in self.layers
        }

    def extract_cdr3(self, out, representations, batch_labels, batch_sequences):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_context_extracted = {
                layer: self.cdr3_context_extracted[layer].extend(
                    [representations[layer][i, start : end + 1].mean(0)]
                )
                for layer in self.layers
            }

    def extract_cdr3_unpooled(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_context_extracted_unpooled = {
                layer: self.cdr3_context_extracted_unpooled[layer].extend(
                    [representations[layer][i, start : end + 1]]
                )
                for layer in self.layers
            }

    def extract_attention_matrices_all_heads(self, out, batch_labels, batch_sequences):
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
        self, out, batch_labels, batch_sequences
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
        self, out, batch_labels, batch_sequences
    ):
        self.attention_matrices_average_all.extend(
            [
                out["attentions"][i, :, :, 1:-1, 1:-1].mean(dim=(0, 1)).cpu()
                for i in range(len(batch_labels))
            ]
        )

    def extract_cdr3_attention_matrices_all_heads(
        self, out, batch_labels, batch_sequences
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
        self, out, batch_labels, batch_sequences
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
        self, out, batch_labels, batch_sequences
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

    def export_to_disk(self):
        """Stack representations of each layer into a single tensor and save to output file."""
        output_name = os.path.splitext(os.path.basename(self.fasta_path))[
            0
        ]  # get filename without extension and path
        for output_type in self.output_types:
            if "attention_matrices" not in output_type:
                for layer in self.layers:
                    output_file_layer = os.path.join(
                        self.output_path,
                        self.model_name,
                        output_type,
                        f"{output_name}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    if "unpooled" not in output_type:
                        getattr(self, output_type)[layer] = torch.vstack(
                            getattr(self, output_type)[layer]
                        )
                    torch.save(getattr(self, output_type)[layer], output_file_layer)
                    print(
                        f"Saved {output_type} representation for layer {layer} to {output_file_layer}"
                    )
            elif "average_all" in output_type:
                output_file = os.path.join(
                    self.output_path,
                    self.model_name,
                    output_type,
                    f"{output_name}_{self.model_name}_{output_type}.pt",
                )
                torch.save(getattr(self, output_type), output_file)
                print(f"Saved {output_type} representation to {output_file}")
            elif "average_layer" in output_type:
                for layer in self.layers:
                    output_file = os.path.join(
                        self.output_path,
                        self.model_name,
                        output_type,
                        f"{output_name}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    torch.save(getattr(self, output_type)[layer], output_file)
                    print(
                        f"Saved {output_type} representation for layer {layer} to {output_file}"
                    )
            elif "all_heads" in output_type:
                for layer in self.layers:
                    for head in range(self.num_heads):
                        output_file = os.path.join(
                            self.output_path,
                            self.model_name,
                            output_type,
                            f"{output_name}_{self.model_name}_{output_type}_layer_{layer}_head_{head}.pt",
                        )
                        torch.save(getattr(self, output_type)[layer][head], output_file)
                        print(
                            f"Saved {output_type} representation for layer {layer} head {head} to {output_file}"
                        )

    def export_sequence_indices(self):
        """Save sequence indices to a CSV file."""
        output_name = os.path.basename(self.fasta_path).replace(".fa", "_idx.csv")
        output_file_idx = os.path.join(self.output_path, output_name)
        with open(output_file_idx, "w") as f:
            f.write("index,sequence_id\n")
            for i, label in enumerate(self.sequence_labels):
                f.write(f"{i},{label}\n")
        print(f"Saved sequence indices to {output_file_idx}")

    def create_output_dirs(self):
        for output_type in self.output_types:
            output_type_path = os.path.join(
                self.output_path, self.model_name, output_type
            )
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)

    def run(self):
        self.create_output_dirs()
        print("Created output directories")

        print("Start embedding extraction")
        self.embed()
        print("Finished embedding extraction")

        print("Saving embeddings...")
        self.export_to_disk()

        self.export_sequence_indices()
