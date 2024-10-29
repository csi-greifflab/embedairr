import os
import csv
import torch
from Bio import SeqIO


class BaseEmbedder:
    def __init__(
        self,
        fasta_path,
        model_name,
        output_path,
        cdr3_path,
        context,
        layers,
        pooling,
        output_types,
    ):
        self.fasta_path = fasta_path
        self.model_name = model_name
        self.output_path = output_path
        self.cdr3_path = cdr3_path
        self.context = context
        self.layers = layers
        self.pooling = pooling
        self.output_types = output_types
        self.cdr3_dict = self.load_cdr3(cdr3_path)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.sequence_labels = []
        self.embeddings = {layer: [] for layer in layers}
        self.embeddings_unpooled = {layer: [] for layer in layers}
        self.cdr3_extracted = {layer: [] for layer in layers}
        self.cdr3_extracted_unpooled = {layer: [] for layer in layers}
        self.cdr3_context_extracted = {layer: [] for layer in layers}
        self.cdr3_context_extracted_unpooled = {layer: [] for layer in layers}

    def fasta_to_dict(self, fasta_path, gaps=False):
        """Convert FASTA file into a dictionary."""
        print("Loading and batching input sequences...")

        seq_dict = dict()
        with open(fasta_path) as f:
            for record in SeqIO.parse(f, "fasta"):
                if gaps:
                    seq_dict[record.id] = " ".join(
                        str(record.seq)
                    )  # AA tokens for hugging face models must be space gapped
                else:
                    seq_dict[record.id] = str(record.seq)
                # print progress
                if len(seq_dict) % 1000 == 0:
                    print(f"{len(seq_dict)} sequences loaded", end="\r")
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

    def data_loader(self):
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

    def extract_batch(self, representations, batch_labels, batch_sequences):
        extraction_methods = {
            "embeddings": self.extract_embeddings,
            "embeddings_unpooled": self.extract_embeddings_unpooled,
            "cdr3_extracted": self.extract_cdr3,
            "cdr3_extracted_unpooled": self.extract_cdr3_unpooled,
            "cdr3_context_extracted": self.extract_cdr3_context,
            "cdr3_context_extracted_unpooled": self.extract_cdr3_context_unpooled,
        }
        for output_type in self.output_types:
            extraction_methods[output_type](
                representations, batch_labels, batch_sequences
            )

    def extract_embeddings(self, representations, batch_labels, batch_sequences):
        self.embeddings = {
            layer: self.embeddings[layer]
            + [
                representations[layer][i, 1 : len(batch_sequences[i]) + 1].mean(0)
                for i in range(len(batch_labels))
            ]
            for layer in self.layers
        }

    def extract_embeddings_unpooled(
        self, representations, batch_labels, batch_sequences
    ):
        self.embeddings_unpooled = {
            layer: self.embeddings_unpooled[layer]
            + [
                representations[layer][i, 1 : len(batch_sequences[i]) + 1]
                for i in range(len(batch_labels))
            ]
            for layer in self.layers
        }

    def extract_cdr3(self, representations, batch_labels, batch_sequences):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_extracted = {
                layer: self.cdr3_extracted[layer]
                + [representations[layer][i, start : end + 1].mean(0)]
                for layer in self.layers
            }

    def extract_cdr3_unpooled(self, representations, batch_labels, batch_sequences):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            self.cdr3_extracted_unpooled = {
                layer: self.cdr3_extracted_unpooled[layer]
                + [representations[layer][i, start : end + 1]]
                for layer in self.layers
            }

    def extract_cdr3_context(self, representations, batch_labels, batch_sequences):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_context_extracted = {
                layer: self.cdr3_context_extracted[layer]
                + [representations[layer][i, start : end + 1].mean(0)]
                for layer in self.layers
            }

    def extract_cdr3_context_unpooled(
        self, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_context_extracted_unpooled = {
                layer: self.cdr3_context_extracted_unpooled[layer]
                + [representations[layer][i, start : end + 1]]
                for layer in self.layers
            }

    def export_to_disk(self):
        """Stack representations of each layer into a single tensor and save to output file."""
        output_name = os.path.basename(self.fasta_path).replace(".fa", "")
        for output_type in self.output_types:
            for layer in self.layers:
                output_file_layer = os.path.join(
                    self.output_path,
                    self.model_name,
                    output_type,
                    f"{output_name}_layer_{layer}.pt",
                )
                if "unpooled" not in output_type:
                    getattr(self, output_type)[layer] = torch.vstack(
                        getattr(self, output_type)[layer]
                    )
                torch.save(getattr(self, output_type)[layer], output_file_layer)
                print(
                    f"Saved {output_type} representation for layer {layer} to {output_file_layer}"
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
