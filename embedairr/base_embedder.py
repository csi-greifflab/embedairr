import os
import csv
import torch
import re
import numpy as np
import mmap


class BaseEmbedder:
    def __init__(self, args):
        self.fasta_path = args.fasta_path
        self.model_link = args.model_name
        self.disable_special_tokens = args.disable_special_tokens
        self.model_name = re.sub(r"^.*?/", "", self.model_link)
        self.output_path = os.path.join(args.output_path, self.model_name)
        # Check if output directory exists and creates it if it's missing
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not args.experiment_name:
            self.output_prefix = os.path.splitext(os.path.basename(self.fasta_path))[
                0
            ]  # get filename without extension and path
        else:
            self.output_prefix = args.experiment_name
        self.cdr3_path = args.cdr3_path
        self.context = args.context
        self.layers = list(map(int, args.layers.split()))
        self.cdr3_dict = self.load_cdr3(args.cdr3_path)
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.output_types = self.get_output_types(args)
        self.return_contacts = False
        for output_type in self.output_types:
            if "attention" in output_type:
                self.return_contacts = True
        self.discard_padding = args.discard_padding
        self.flatten = True
        if (args.extract_embeddings[0] == "false") & (
            args.extract_cdr3_embeddings[0] == "false"
        ):
            self.return_embeddings = False
        else:
            self.return_embeddings = True
        self.batch_writing = args.batch_writing

    def check_input_tokens(self, valid_tokens, sequences, gaps=False):
        print("Checking input sequences for invalid tokens...")
        for i, (label, sequence) in enumerate(sequences.items()):
            if gaps:
                sequence = sequence.split()
            else:
                sequence = re.findall(r"<.*?>|.", sequence)
            if not set(sequence).issubset(valid_tokens):
                # find invalid tokens
                invalid_tokens = set(sequence) - valid_tokens
                raise ValueError(
                    f"Invalid tokens in sequence {label}. Please check the alphabet used by the model."
                )
            print(f"Processed {i + 1} out of {len(sequences)} sequences", end="\r")

        print("\nNo invalid tokens in input sequences.")

    def set_output_objects(self):
        """Initialize output objects."""
        self.sequence_labels = []
        self.embeddings = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self.extract_embeddings,
            "output_dir": os.path.join(self.output_path, "embeddings"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.embeddings_unpooled = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self.extract_embeddings_unpooled,
            "output_dir": os.path.join(self.output_path, "embeddings_unpooled"),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.embedding_size,
            ),
        }
        self.cdr3_extracted = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self.extract_cdr3,
            "output_dir": os.path.join(self.output_path, "cdr3_extracted"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.attention_matrices_all_heads = {
            "output_data": {
                layer: {head: [] for head in range(self.num_heads)}
                for layer in self.layers
            },
            "method": self.extract_attention_matrices_all_heads,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_all_heads"
            ),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
            ),
            "shape_flattened": (
                self.num_sequences,
                (
                    self.max_length**2
                    if self.disable_special_tokens
                    else (self.max_length + 2) ** 2
                ),
            ),
        }
        self.attention_matrices_average_layers = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self.extract_attention_matrices_average_layer,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_average_layers"
            ),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
            ),
            "shape_flattened": (
                self.num_sequences,
                (
                    self.max_length**2
                    if self.disable_special_tokens
                    else (self.max_length + 2) ** 2
                ),
            ),
        }
        self.attention_matrices_average_all = {
            "output_data": [],
            "method": self.extract_attention_matrices_average_all,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_average_all"
            ),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
            ),
            "shape_flattened": (
                self.num_sequences,
                (
                    self.max_length**2
                    if self.disable_special_tokens
                    else (self.max_length + 2) ** 2
                ),
            ),
        }
        self.cdr3_attention_matrices_average_layers = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self.extract_cdr3_attention_matrices_average_layer,
            "output_dir": os.path.join(
                self.output_path,
                "cdr3_attention_matrices_average_layers",
            ),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
            ),
            "shape_flattened": (
                self.num_sequences,
                (
                    self.max_length**2
                    if self.disable_special_tokens
                    else (self.max_length + 2) ** 2
                ),
            ),
        }
        self.cdr3_attention_matrices_average_all = {
            "output_data": [],
            "method": self.extract_cdr3_attention_matrices_average_all,
            "output_dir": os.path.join(
                self.output_path, "cdr3_attention_matrices_average_all"
            ),
            "shape": (
                self.num_sequences,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
                self.max_length if self.disable_special_tokens else self.max_length + 2,
            ),
            "shape_flattened": (
                self.num_sequences,
                (
                    self.max_length**2
                    if self.disable_special_tokens
                    else (self.max_length + 2) ** 2
                ),
            ),
        }

    # When changes made here, also update base_embedder.py BaseEmbedder.set_output_objects() method.
    def get_output_types(self, args):
        output_types = []

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

        return output_types

    def preallocate_disk_space(self, output_type):
        dtype = torch.float16  # TODO make optional
        element_size = torch.tensor([], dtype=dtype).element_size()
        if "attention" in output_type:
            shape = getattr(self, output_type)["shape_flattened"]
            numel = np.prod(shape)
            expected_size = numel * element_size
            if "average_all" in output_type:
                output_file = os.path.join(
                    getattr(self, output_type)["output_dir"],
                    f"{self.output_prefix}_{self.model_name}_{output_type}.pt",
                )
                # Save it to disk
                with open(output_file, "wb") as f:
                    f.truncate(expected_size)
            elif "average_layer" in output_type:
                for layer in self.layers:
                    output_file_layer = os.path.join(
                        getattr(self, output_type)["output_dir"],
                        f"{self.output_prefix}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    # Save it to disk
                    with open(output_file_layer, "wb") as f:
                        f.truncate(expected_size)
            elif "all_heads" in output_type:
                for layer in self.layers:
                    for head in range(self.num_heads):
                        output_file = os.path.join(
                            getattr(self, output_type)["output_dir"],
                            f"{self.output_prefix}_{self.model_name}_{output_type}_layer_{layer}_head_{head + 1}.pt",
                        )
                        # Save it to disk
                        with open(output_file, "wb") as f:
                            f.truncate(expected_size)
        elif "cdr3_attention_matrices" not in output_type:
            shape = getattr(self, output_type)["shape"]
            numel = np.prod(shape)  # Total number of elements
            expected_size = numel * element_size
            for layer in self.layers:
                output_file_layer = os.path.join(
                    getattr(self, output_type)["output_dir"],
                    f"{self.output_prefix}_{self.model_name}_{output_type}_layer_{layer}.pt",
                )
                # Save it to disk
                with open(output_file_layer, "wb") as f:
                    f.truncate(expected_size)
        else:
            raise ValueError(
                f"Output type {output_type} not recognized. Please choose from: 'embeddings', 'embeddings_unpooled', 'attention_matrices_all_heads', 'attention_matrices_average_layers', 'attention_matrices_average_all', 'cdr3_extracted'"
            )
        print(f"Preallocated disk space for {output_type}")

    def fasta_to_dict(self, fasta_path, gaps=False, padding=False):
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
                seq_dict[sequence_id] = " ".join(
                    re.findall(r"\[.*?\]|.", seq)
                )  # split sequences by space except for special tokens in brackets
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

        if gaps:  # if gaps, split sequences by space
            longest_sequence = max(len(seq.split()) for seq in seq_dict.values())
        else:  # if no gaps, split sequences by amino acids and special tokens
            longest_sequence = max(
                len(re.findall(r"<.*?>|.", seq)) for seq in seq_dict.values()
            )

        if self.max_length < longest_sequence:
            # raise warning
            print(
                f"Longest sequence with length {longest_sequence} is longer than the specified max_length {self.max_length} for padding"
            )
            self.max_length = longest_sequence
        if not padding:
            return seq_dict, None
        else:
            padded_seq_dict = dict()
            for label, sequence in seq_dict.items():
                if len(sequence) < self.max_length:
                    padded_seq_dict[label] = sequence + "<pad>" * (
                        self.max_length - len(re.findall(r"<.*?>|.", sequence))
                    )
                else:
                    padded_seq_dict[label] = sequence
            return seq_dict, padded_seq_dict

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

    def extract_batch(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask=None,
        batch_idx=None,
    ):
        for output_type in self.output_types:
            getattr(self, output_type)["method"](
                attention_matrices,
                representations,
                batch_labels,
                batch_sequences,
                pooling_mask,
                batch_idx,
            )

    def mask_special_tokens(self, input_tensor, special_tokens=None):
        """
        Create a boolean mask for special tokens in the input tensor.

        """
        special_tokens = None
        if (
            special_tokens is not None
        ):  # Create a boolean mask: True where the value is not in special_tokens.
            mask = ~torch.isin(input_tensor, special_tokens)
        else:  # Create a boolean mask: True where the value is not 0, 1, or 2.
            mask = (input_tensor != 0) & (input_tensor != 1) & (input_tensor != 2)
        # Convert and return the boolean mask to boolean type.
        return mask

    def extract_embeddings(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            tensor = torch.stack(
                [
                    (
                        (pooling_mask[i].unsqueeze(-1) * representations[layer][i]).sum(
                            0
                        )
                        / pooling_mask[i].unsqueeze(-1).sum(0)
                    )
                    for i in range(len(batch_labels))
                ]
            )
            if self.batch_writing:
                output_file = os.path.join(
                    self.embeddings["output_dir"],
                    f"{self.output_prefix}_{self.model_name}_embeddings_layer_{layer}.pt",
                )
                self.write_batch_to_disk(output_file, tensor, batch_idx)
            else:
                self.embeddings["output_data"][layer].extend(tensor)

    def extract_embeddings_unpooled(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        if not self.discard_padding:
            for layer in self.layers:
                tensor = torch.stack(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )
                if self.batch_writing:
                    output_file = os.path.join(
                        self.embeddings_unpooled["output_dir"],
                        f"{self.output_prefix}_{self.model_name}_embeddings_unpooled_layer_{layer}.pt",
                    )
                    self.write_batch_to_disk(output_file, tensor, batch_idx)
                else:
                    self.embeddings_unpooled["output_data"][layer].extend(tensor)
        else:  # TODO remove padding tokens
            print("Feature not implemented yet")
            pass
            for layer in self.layers:
                self.embeddings_unpooled["output_data"][layer].extend(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )

    def extract_attention_matrices_all_heads(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                tensor = torch.stack(
                    [
                        attention_matrices[layer - 1, i, head]
                        for i in range(len(batch_labels))
                    ]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.batch_writing:
                    output_file = os.path.join(
                        self.attention_matrices_all_heads["output_dir"],
                        f"{self.output_prefix}_{self.model_name}_attention_matrices_all_heads_layer_{layer}_head_{head + 1}.pt",
                    )
                    self.write_batch_to_disk(output_file, tensor, batch_idx)
                else:
                    self.attention_matrices_all_heads["output_data"][layer][
                        head
                    ].extend(tensor)

    def extract_attention_matrices_average_layer(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            tensor = torch.stack(
                [
                    attention_matrices[layer - 1, i].mean(0)
                    for i in range(len(batch_labels))
                ]
            )
            if self.flatten:
                tensor = tensor.flatten(start_dim=1)
            if self.batch_writing:
                output_file = os.path.join(
                    self.attention_matrices_average_layers["output_dir"],
                    f"{self.output_prefix}_{self.model_name}_attention_matrices_average_layers_layer_{layer}.pt",
                )
                self.write_batch_to_disk(output_file, tensor, batch_idx)
            else:
                self.attention_matrices_average_layers["output_data"][layer].extend(
                    tensor
                )

    def extract_attention_matrices_average_all(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        tensor = torch.stack(
            [
                attention_matrices[:, i].mean(dim=(0, 1))
                for i in range(len(batch_labels))
            ]
        )
        if self.flatten:
            tensor = tensor.flatten(start_dim=1)
        if self.batch_writing:
            output_file = os.path.join(
                self.attention_matrices_average_all["output_dir"],
                f"{self.output_prefix}_{self.model_name}_attention_matrices_average_all.pt",
            )
            self.write_batch_to_disk(output_file, tensor, batch_idx)
        else:
            self.attention_matrices_average_all["output_data"].extend(tensor)

    def extract_cdr3_attention_matrices_average_all_heads(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            for head in range(self.num_heads):
                tensor = []
                for i, label in enumerate(batch_labels):
                    start, end = self.get_cdr3_positions(label)
                    tensor.extend(
                        attention_matrices[
                            layer - 1, i, head, start:end, start:end
                        ].mean(0)
                    )
                tensor = torch.stack(tensor)
                self.cdr3_attention_matrices_all_heads["output_data"][layer][
                    head
                ].extend(tensor)

    def extract_cdr3_attention_matrices_average_layer(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            tensor = []
            for i, label in enumerate(batch_labels):
                start, end = self.get_cdr3_positions(label)
                tensor.extend(
                    attention_matrices[layer - 1, i, :, start:end, start:end].mean(0)
                )
            tensor = torch.stack(tensor)
            self.cdr3_attention_matrices_average_layers["output_data"][layer].extend(
                tensor
            )

    def extract_cdr3_attention_matrices_average_all(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        tensor = []
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            tensor.extend(
                attention_matrices[:, i, :, start:end, start:end].mean(dim=(0, 1))
            )
        tensor = torch.stack(tensor)
        self.cdr3_attention_matrices_average_all["output_data"].extend(tensor)

    def extract_cdr3(
        self,
        attention_matrices,
        representations,
        batch_labels,
        batch_sequences,
        pooling_mask,
        batch_idx,
    ):
        for layer in self.layers:
            tensor = []
            for i, label in enumerate(batch_labels):
                start, end = self.get_cdr3_positions(label, context=self.context)
                tensor.extend(
                    (
                        pooling_mask[i, start:end].unsqueeze(-1)
                        * representations[layer][i, start:end]
                    ).sum(0)
                    / pooling_mask[i, start:end].unsqueeze(-1).sum(0)
                )
            tensor = torch.stack(tensor)
            if self.batch_writing:
                output_file = os.path.join(
                    self.cdr3_extracted["output_dir"],
                    f"{self.output_prefix}_{self.model_name}_cdr3_extracted_layer_{layer}.pt",
                )
                self.write_batch_to_disk(output_file, tensor, batch_idx)
            else:
                self.cdr3_extracted["output_data"][layer].extend(tensor)

    def write_batch_to_disk(self, file_path, tensor, batch_idx):
        with open(file_path, "r+b") as f:
            mmapped_file = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_WRITE)
            offset = batch_idx * tensor.element_size() * tensor.shape[0]
            mmapped_file.seek(offset)
            mmapped_file.write(tensor.numpy().tobytes())
            mmapped_file.flush()

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
                        output_type,
                        f"{output_name}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    if "unpooled" not in output_type:
                        getattr(self, output_type)[layer] = torch.vstack(
                            getattr(self, output_type)["output_data"][layer]
                        )
                        stacked = True
                    if self.flatten and "unpooled" in output_type:
                        getattr(self, output_type)["output_data"][layer] = torch.stack(
                            getattr(self, output_type)["output_data"][layer], dim=0
                        ).flatten(start_dim=1)
                        stacked = True
                    elif not self.discard_padding and not stacked:
                        getattr(self, output_type)["output_data"][layer] = torch.stack(
                            getattr(self, output_type)["output_data"][layer]
                        )
                        stacked = True
                    torch.save(
                        (getattr(self, output_type)["output_data"][layer]),
                        output_file_layer,
                    )
                    print(
                        f"Saved {output_type} representation for layer {layer} to {output_file_layer}"
                    )
            elif "average_all" in output_type:
                output_file = os.path.join(
                    self.output_path,
                    output_type,
                    f"{self.output_prefix}_{self.model_name}_{output_type}.pt",
                )
                if self.flatten:
                    torch.save(
                        torch.stack(
                            getattr(self, output_type)["output_data"], dim=0
                        ).flatten(start_dim=1),
                        output_file,
                    )
                else:
                    torch.save(
                        torch.stack(getattr(self, output_type)["output_data"], dim=0),
                        output_file,
                    )
                print(f"Saved {output_type} representation to {output_file}")
            elif "average_layer" in output_type:
                for layer in self.layers:
                    output_file = os.path.join(
                        self.output_path,
                        output_type,
                        f"{self.output_prefix}_{self.model_name}_{output_type}_layer_{layer}.pt",
                    )
                    if self.flatten:
                        torch.save(
                            torch.stack(
                                getattr(self, output_type)["output_data"][layer], dim=0
                            ).flatten(start_dim=1),
                            output_file,
                        )
                    else:
                        torch.save(
                            torch.stack(
                                getattr(self, output_type)["output_data"][layer]
                            ),
                            output_file,
                        )
                    print(
                        f"Saved {output_type} representation for layer {layer} to {output_file}"
                    )
            elif "all_heads" in output_type:
                for layer in self.layers:
                    for head in range(self.num_heads):
                        output_file = os.path.join(
                            self.output_path,
                            output_type,
                            f"{self.output_prefix}_{self.model_name}_{output_type}_layer_{layer}_head_{head + 1}.pt",
                        )
                        if self.flatten:
                            torch.save(
                                torch.stack(
                                    getattr(self, output_type)["output_data"][layer][
                                        head
                                    ],
                                    dim=0,
                                ).flatten(start_dim=1),
                                output_file,
                            )
                            # delete the tensor from memory
                            # del getattr(self, output_type)
                        else:
                            torch.save(
                                torch.stack(
                                    getattr(self, output_type)["output_data"][layer][
                                        head
                                    ]
                                ),
                                output_file,
                            )
                            # delete the tensor from memory
                            # del getattr(self, output_type)
                        print(
                            f"Saved {output_type} representation for layer {layer} head {head + 1} to {output_file}"
                        )

    def export_sequence_indices(self):
        """Save sequence indices to a CSV file."""
        input_file_name = os.path.basename(self.fasta_path)
        # replace file extension with _idx.csv regardless of pattern
        output_file_name = os.path.splitext(input_file_name)[0] + "_idx.csv"
        output_file_idx = os.path.join(self.output_path, output_file_name)
        with open(output_file_idx, "w") as f:
            f.write("index,sequence_id\n")
            for i, label in enumerate(self.sequence_labels):
                f.write(f"{i},{label}\n")
        print(f"Saved sequence indices to {output_file_idx}")

    def create_output_dirs(self):
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)

    def run(self):
        self.create_output_dirs()
        if self.batch_writing:
            print("Preallocating disk space...")
            for output_type in self.output_types:
                self.preallocate_disk_space(output_type)
            print("Preallocated disk space")
        print("Created output directories")

        print("Start embedding extraction")
        self.embed()
        print("Finished embedding extraction")

        print("Saving embeddings...")
        if not self.batch_writing:
            self.export_to_disk()

        self.export_sequence_indices()
