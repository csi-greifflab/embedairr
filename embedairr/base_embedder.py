import os
import csv
import torch
import sys
import gc
import re
from itertools import islice

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class BaseEmbedder:
    def __init__(self, args):
        self.fasta_path = args.fasta_path
        self.model_name = args.model_name
        self.output_path = args.output_path
        self.cdr3_path = args.cdr3_path
        self.context = args.context
        self.layers = list(map(int, args.layers.split()))
        self.cdr3_dict = self.load_cdr3(args.cdr3_path)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Transferred model to GPU")
        else:
            self.device = torch.device("cpu")
        self.output_types, self.extraction_methods = self.get_output_types(args)
        self.return_contacts = False
        for output_type in self.output_types:
            if "attention" in output_type:
                self.return_contacts = True
        self.discard_padding = args.discard_padding
        self.flatten = True
        if (args.extract_embeddings[0] == "false") & (args.extract_cdr3_embeddings[0] == "false"):
            self.return_embeddings = False
        else:
            self.return_embeddings = True

    def check_input_tokens(self, valid_tokens, sequences, gaps=False):
        print("Checking input sequences for invalid tokens...")
        for i, (label, sequence) in enumerate(sequences.items()):
            if gaps:
                sequence = sequence.split()
            else:
                sequence = re.findall(r'<.*?>|.', sequence)
            if not set(sequence).issubset(valid_tokens):
                raise ValueError(
                    f"Invalid tokens in sequence {label}. Please check the alphabet used by the model."
                )
            print(f'Processed {i} out of {len(sequences)} sequences', end="\r")

        print("\nNo invalid tokens in input sequences.")

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
            layer: {head: [] for head in range(self.num_heads)}
            for layer in range(1, self.num_layers + 1)
        }
        self.attention_matrices_average_layers = {
            layer: [] for layer in range(1, self.num_layers + 1)
        }
        self.attention_matrices_average_all = []
        self.cdr3_attention_matrices_all_heads = {
            layer: {head: [] for head in range(self.num_heads)}
            for layer in range(1, self.num_layers + 1)
        }
        self.cdr3_attention_matrices_average_layers = {
            layer: [] for layer in range(1, self.num_layers + 1)
        }
        self.cdr3_attention_matrices_average_all = []

    # When changes made here, also update base_embedder.py BaseEmbedder.extract_batch() method.
    def get_output_types(self, args):
        output_types = []
        extraction_methods_list = []

        options_mapping = {
            "pooled": ["embeddings", "cdr3_extracted"],
            "unpooled": ["embeddings_unpooled", "cdr3_extracted_unpooled"],
            "average_all": [
                "attention_matrices_average_all",
                "cdr3_attention_matrices_average_all",
            ],
            "average_layer": [
                "attention_matrices_average_layers",
                "cdr3_attention_matrices_average_layers",
            ],
            "all_heads": [
                "attention_matrices_all_heads",
                "cdr3_attention_matrices_all_heads",
            ],
        }

        for option, types in options_mapping.items():
            if option in args.extract_embeddings:
                output_types.append(types[0])
            if args.cdr3_path and option in args.extract_cdr3_embeddings:
                output_types.append(types[1])
            if option in args.extract_attention_matrices:
                output_types.append(types[0])
            if args.cdr3_path and option in args.extract_cdr3_attention_matrices:
                output_types.append(types[1])

        extraction_methods = {
            "embeddings": self.extract_embeddings,
            "embeddings_unpooled": self.extract_embeddings_unpooled,
            "cdr3_extracted": self.extract_cdr3,
            "cdr3_extracted_unpooled": self.extract_cdr3_unpooled,
            "attention_matrices_all_heads": self.extract_attention_matrices_all_heads,
            "attention_matrices_average_layers": self.extract_attention_matrices_average_layer,
            "attention_matrices_average_all": self.extract_attention_matrices_average_all,
            "cdr3_attention_matrices_all_heads": self.extract_cdr3_attention_matrices_all_heads,
            "cdr3_attention_matrices_average_layers": self.extract_cdr3_attention_matrices_average_layer,
            "cdr3_attention_matrices_average_all": self.extract_cdr3_attention_matrices_average_all,
        }
        for output_type in output_types:
            # if output_type contains attention_matrix, call attention_matrix extraction method
            if output_type in extraction_methods:
                extraction_methods_list.append(extraction_methods[output_type])
            else:
                raise ValueError(f"Output type {output_type} not supported")

        return output_types, extraction_methods_list

    def fasta_to_dict(self, fasta_path, gaps=False, padding=False):
        """Convert FASTA file into a dictionary."""
        print("Loading and batching input sequences...")

        seq_dict = dict()
        sequence_id = None
        sequence_aa = []
        import re

        def _clean_bracket_tokens(seq: str) -> str:
            """
            Remove spaces inside square or angle brackets.

            Example:
            "A B [ S E P ] C D" -> "A B [SEP] C D"
            """
            pattern = re.compile(r"([\[<])\s*([^]\>]+?)\s*([\]>])")
            return pattern.sub(
                lambda m: m.group(1) + m.group(2).replace(" ", "") + m.group(3), seq
            )

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
                        sequence_id = (
                            f"seqnum{line_idx:09d}"  # if no id, use line number
                        )
                else:
                    sequence_aa.append(line)

        _flush_current_seq()

        if gaps == False:
            longest_sequence = max(len(seq) for seq in seq_dict.values())
        else:
            # delete only spaces between "[" and "]" but keep all others
            for label, sequence in seq_dict.items():
                seq_dict[label] = _clean_bracket_tokens(sequence)
            longest_sequence = max(len(seq.split()) for seq in seq_dict.values())

        if self.max_length < longest_sequence:
            # raise warning
            print(
                f"Longest sequence with length {longest_sequence} is longer than the specified max_length {self.max_length} for padding"
            )
            self.max_length = longest_sequence
        if not padding:
            return seq_dict, None
        elif self.model_name == "esm2":
            padded_seq_dict = dict()
            for label, sequence in seq_dict.items():
                if len(sequence) < self.max_length:
                    padded_seq_dict[label] = sequence + "<pad>" * (
                        self.max_length - len(sequence)
                    )
                else:
                    padded_seq_dict[label] = sequence[: self.max_length]
            return seq_dict, padded_seq_dict

    def load_cdr3(self, cdr3_path):
        """Load CDR3 sequences and store in a dictionary."""
        if cdr3_path:
            with open(cdr3_path) as f:
                reader = csv.reader(f)
                cdr3_dict = {rows[0]: rows[1] for rows in reader}
                for i, (key, value) in enumerate(cdr3_dict.items()):
                    if i < 5:
                        print(f"{key} : {value}   cdr3 dict")
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

    def extract_batch(self, attention_matrices, representations, batch_labels, batch_sequences, pooling_mask = None):
        for method in self.extraction_methods:
            method(attention_matrices, representations, batch_labels, batch_sequences, pooling_mask)

    def extract_cdr3(self, out, representations, batch_labels, batch_sequences):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_extracted = {
                layer: (
                    self.cdr3_extracted[layer]
                    + [representations[layer][i, start : end + 1].mean(0)]
                )
                for layer in self.layers
            }

    def mask_special_tokens(self, input_tensor, special_tokens=None):
        """
        Create a boolean mask for special tokens in the input tensor.
        
        """
        special_tokens = None
        if special_tokens is not None: # Create a boolean mask: True where the value is not in special_tokens.
            mask = ~torch.isin(input_tensor, special_tokens)
        else: # Create a boolean mask: True where the value is not 0, 1, or 2.
            mask = (input_tensor != 0) & (input_tensor != 1) & (input_tensor != 2)
        # Convert and return the boolean mask to boolean type.
        return mask
    def extract_cdr3_unpooled(
        self, out, representations, batch_labels, batch_sequences
    ):
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label, context=self.context)
            self.cdr3_extracted_unpooled = {
                layer: (
                    self.cdr3_extracted_unpooled[layer]
                    + [representations[layer][i, start : end + 1]]
                )
                for layer in self.layers
            }
        # print(f" cdr3 tensor {self.cdr3_extracted_unpooled[1].shape}")

    def export_to_disk(self):
        """Stack representations of each layer into a single tensor and save to output file."""
        output_name = os.path.splitext(os.path.basename(self.fasta_path))[
            0
        ]  # get filename without extension and path
        for output_type in self.output_types:
            print(f"Saving {output_type} representations...")
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
                        stacked = True
                    if self.flatten and "unpooled" in output_type:
                        getattr(self, output_type)[layer] = torch.stack(
                            getattr(self, output_type)[layer], dim=0
                        ).flatten(start_dim=1)
                        stacked = True
                    elif not self.discard_padding and not stacked:
                        getattr(self, output_type)[layer] = torch.stack(
                            getattr(self, output_type)[layer]
                        )
                        stacked = True
                    torch.save((getattr(self, output_type)[layer]), output_file_layer)
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
                if self.flatten:
                    torch.save(
                        torch.stack(getattr(self, output_type), dim=0).flatten(
                            start_dim=1
                        ),
                        output_file,
                    )
                else:
                    torch.save(
                        torch.stack(getattr(self, output_type), dim=0), output_file
                    )
                print(f"Saved {output_type} representation to {output_file}")
            elif "average_layer" in output_type:
                for layer in self.layers:
                    output_file = os.path.join(
                        self.output_path,
                        self.model_name,
                        output_type,
                        f"{output_name}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    if self.flatten:
                        torch.save(
                            torch.stack(
                                getattr(self, output_type)[layer], dim=0
                            ).flatten(start_dim=1),
                            output_file,
                        )
                    else:
                        torch.save(
                            torch.stack(getattr(self, output_type)[layer]), output_file
                        )
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
                            f"{output_name}_{self.model_name}_{output_type}_layer_{layer}_head_{head + 1}.pt",
                        )
                        if self.flatten:
                            torch.save(
                                torch.stack(
                                    getattr(self, output_type)[layer][head], dim=0
                                ).flatten(start_dim=1),
                                output_file,
                            )
                        else:
                            torch.save(
                                torch.stack(getattr(self, output_type)[layer][head]),
                                output_file,
                            )
                        print(
                            f"Saved {output_type} representation for layer {layer} head {head + 1} to {output_file}"
                        )

    def export_sequence_indices(self):
        """Save sequence indices to a CSV file."""
        filename = os.path.basename(self.fasta_path)
        # replace file extension with _idx.csv regardless of pattern
        output_name = os.path.splitext(filename)[0] + "_idx.csv"
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
