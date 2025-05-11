import os
import csv
import torch
import re
import numpy as np
from numpy.lib.format import open_memmap
import inspect
import gc
from embedairr.utils import IOFlushWorker
from alive_progress import alive_bar
import time


class BaseEmbedder:
    def __init__(self, args):
        self.ram_limit = args.ram_limit
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
        self.layers = (
            [j for i in args.layers for j in i] if args.layers != [None] else None
        )
        self.cdr3_dict = self._load_cdr3(args.cdr3_path) if args.cdr3_path else None
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.return_logits = args.extract_logits
        self.output_types = self._get_output_types(args)
        self.return_contacts = False
        for output_type in self.output_types:
            if "attention" in output_type:
                self.return_contacts = True
        self.discard_padding = args.discard_padding
        self.flatten = args.flatten
        if (args.extract_embeddings[0] == "false") & (
            args.extract_cdr3_embeddings[0] == "false"
        ):
            self.return_embeddings = False
        else:
            self.return_embeddings = True
        self.batch_writing = args.batch_writing
        self.num_workers = args.num_workers if self.batch_writing else 1
        self.max_in_flight = self.num_workers * 2
        self.flush_batches_after = args.flush_batches_after * 1024**2  # in bytes
        self.precision = args.precision
        # self.log_memory = args.log_memory # TODO implement memory logging

    def _precision_to_dtype(self, precision, framework):
        half_precision = ["float16", "16", "half"]
        full_precision = ["float32", "32", "full"]
        if precision in half_precision:
            if framework == "torch":
                return torch.float16
            elif framework == "numpy":
                return np.float16
        elif precision in full_precision:
            if framework == "torch":
                return torch.float32
            elif framework == "numpy":
                return np.float32
        else:
            raise ValueError(
                f"Unsupported precision: {precision}. Supported values are {half_precision} or {full_precision}."
            )

    def _set_output_objects(self):
        """Initialize output objects."""
        self.sequence_labels = []
        self.logits = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_logits,
            "output_dir": os.path.join(self.output_path, "logits"),
            "shape": (
                self.num_sequences,
                self.max_length,
            ),
        }
        self.embeddings = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_embeddings,
            "output_dir": os.path.join(self.output_path, "embeddings"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.embeddings_unpooled = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_embeddings_unpooled,
            "output_dir": os.path.join(self.output_path, "embeddings_unpooled"),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.embedding_size,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length * self.embedding_size,
                )
            ),
        }
        self.cdr3_extracted = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_cdr3,
            "output_dir": os.path.join(self.output_path, "cdr3_extracted"),
            "shape": (self.num_sequences, self.embedding_size),
        }
        self.attention_matrices_all_heads = {
            "output_data": {
                layer: {head: [] for head in range(self.num_heads)}
                for layer in self.layers
            },
            "method": self._extract_attention_matrices_all_heads,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_all_heads"
            ),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length**2,
                )
            ),
        }
        self.attention_matrices_average_layers = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_attention_matrices_average_layer,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_average_layers"
            ),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length**2,
                )
            ),
        }
        self.attention_matrices_average_all = {
            "output_data": [],
            "method": self._extract_attention_matrices_average_all,
            "output_dir": os.path.join(
                self.output_path, "attention_matrices_average_all"
            ),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length**2,
                )
            ),
        }
        self.cdr3_attention_matrices_average_layers = {
            "output_data": {layer: [] for layer in self.layers},
            "method": self._extract_cdr3_attention_matrices_average_layer,
            "output_dir": os.path.join(
                self.output_path,
                "cdr3_attention_matrices_average_layers",
            ),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length**2,
                )
            ),
        }
        self.cdr3_attention_matrices_average_all = {
            "output_data": [],
            "method": self._extract_cdr3_attention_matrices_average_all,
            "output_dir": os.path.join(
                self.output_path, "cdr3_attention_matrices_average_all"
            ),
            "shape": (
                (
                    self.num_sequences,
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,
                    self.max_length**2,
                )
            ),
        }

    # When changes made here, also update base_embedder.py BaseEmbedder.set_output_objects() method.
    def _get_output_types(self, args):
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

        if self.return_logits:
            output_types.append("logits")
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

    def _make_output_filepath(self, output_type, output_dir, layer=None, head=None):
        base = f"{self.output_prefix}_{self.model_name}_{output_type}"
        if layer is not None:
            base += f"_layer_{layer}"
        if head is not None:
            base += f"_head_{head + 1}"
        return os.path.join(output_dir, base + ".npy")

    def preallocate_disk_space(self):
        memmap_registry = {}
        for output_type in self.output_types:
            output_data = getattr(self, output_type)["output_data"]
            shape = getattr(self, output_type)["shape"]
            output_dir = getattr(self, output_type)["output_dir"]
            np_dtype = self._precision_to_dtype(
                self.precision, "numpy"
            )  # TODO: make configurable

            if isinstance(output_data, dict):
                for layer in self.layers:
                    if isinstance(output_data[layer], dict):  # e.g., all_heads
                        for head in range(self.num_heads):
                            file_path = self._make_output_filepath(
                                output_type, output_dir, layer, head
                            )
                            output_data[layer][head] = open_memmap(
                                file_path, mode="w+", dtype=np_dtype, shape=shape
                            )
                            memmap_registry[(output_type, layer, head)] = output_data[
                                layer
                            ][head]
                    else:
                        file_path = self._make_output_filepath(
                            output_type, output_dir, layer
                        )
                        output_data[layer] = open_memmap(
                            file_path, mode="w+", dtype=np_dtype, shape=shape
                        )
                        memmap_registry[(output_type, layer, None)] = output_data[layer]
            else:
                file_path = self._make_output_filepath(output_type, output_dir)
                output_array = open_memmap(
                    file_path, mode="w+", dtype=np_dtype, shape=shape
                )
                setattr(getattr(self, output_type), "output_data", output_array)
                memmap_registry[(output_type, None, None)] = output_array

            print(f"Preallocated disk space for {output_type}")
        return memmap_registry

    def _load_cdr3(self, cdr3_path):
        """Load CDR3 sequences and store in a dictionary."""
        if cdr3_path:
            with open(cdr3_path) as f:
                reader = csv.reader(f)  # skip header
                next(reader)
                cdr3_dict = {rows[0]: rows[1] for rows in reader}
            return cdr3_dict
        else:
            return None

    def _safe_compute(self, toks, attention_mask):
        """
        Try to run compute_outputs; on OOM, empty cache, split in half,
        recurse on each half, then concatenate.
        """
        try:
            return self._compute_outputs(
                self.model,
                toks,
                attention_mask,
                self.return_embeddings,
                self.return_contacts,
                self.return_logits,
            )
        except torch.OutOfMemoryError:
            print("[GPU memory overflow] Decreasing batch size and retrying...")
            torch.cuda.empty_cache()
            B = toks.size(0)
            if B == 1:
                # can’t split anymore
                print("OOM on single sample!")
                raise
            # split into two roughly equal chunks
            half = B // 2
            toks_chunks = torch.split(toks, [half, B - half], dim=0)
            if attention_mask is not None:
                mask_chunks = torch.split(attention_mask, [half, B - half], dim=0)
            else:
                mask_chunks = [None, None]

            outs = [
                self._safe_compute(tc, mc) for tc, mc in zip(toks_chunks, mask_chunks)
            ]
            # outs is list of (logits, reps, attn)
            logits = (
                torch.cat([o[0] for o in outs], dim=0) if self.return_logits else None
            )
            representations = (
                torch.cat([o[1] for o in outs], dim=0)
                if self.return_embeddings
                else None
            )
            attention_matrices = (
                torch.cat([o[2] for o in outs], dim=0) if self.return_contacts else None
            )
            return logits, representations, attention_matrices

    def embed(self):
        if self.batch_writing:
            # Start centralized I/O dispatcher
            self.io_dispatcher = IOFlushWorker(
                self.memmap_registry, flush_bytes_limit=self.flush_batches_after
            )
            self.io_dispatcher.start()
        with alive_bar(
            len(self.sequences),
            title=f"{self.model_name}: Generating embeddings ...",
        ) as bar, torch.no_grad():
            offset = 0
            for (
                labels,
                strs,
                toks,
                attention_mask,
                cdr3_mask,
            ) in self.data_loader:
                toks = toks.to(self.device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                pooling_mask = self._mask_special_tokens(
                    toks, self.special_tokens
                ).cpu()  # mask special tokens to avoid diluting signal when pooling embeddings
                logits, representations, attention_matrices = self._safe_compute(
                    toks, attention_mask
                )
                torch.cuda.empty_cache()

                output_bundle = {
                    "logits": logits,
                    "attention_matrices": attention_matrices,
                    "representations": representations,
                    "batch_labels": labels,
                    "pooling_mask": pooling_mask,
                    "cdr3_mask": cdr3_mask,
                    "offset": offset,
                    "special_tokens": not self.disable_special_tokens,
                }
                if self.batch_writing:
                    # Apply backpressure if write queue is too full
                    while self.io_dispatcher.queue_fullness() > 0.9:
                        print(
                            "[embed] Backpressure: waiting for IOFlushWorker to catch up..."
                        )
                        time.sleep(0.05)
                self._extract_batch(output_bundle)

                del logits, representations, attention_matrices
                gc.collect()
                torch.cuda.empty_cache()

                offset += len(toks)

                self.sequence_labels.extend(labels)
                bar(len(toks))
            if self.batch_writing:
                self.io_dispatcher.stop()
            print("Finished extracting embeddings")

    def _load_data(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _initialize_model(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def _load_layers(self):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    def get_cdr3_positions(self, label, special_tokens, context=0):
        """Get the start and end positions of the CDR3 sequence in the full sequence."""
        full_sequence = self.sequences[label]

        try:
            cdr3_sequence = self.cdr3_dict[label]
        except KeyError:
            SystemExit(f"No cdr3 sequence found for {label}")
        # remove '-' from cdr3_sequence
        cdr3_sequence = cdr3_sequence.replace("-", "")

        # get position of cdr3_sequence in sequence
        start = max(full_sequence.find(cdr3_sequence) - context, 0) + int(
            special_tokens
        )
        end = (
            min(start + len(cdr3_sequence) + context, len(full_sequence))
            + special_tokens
        )

        return start, end

    def _extract_batch(
        self,
        output_bundle,
    ):
        for output_type in self.output_types:
            sig = inspect.signature(getattr(self, output_type)["method"])
            needed_args = {
                k: v for k, v in output_bundle.items() if k in sig.parameters
            }
            getattr(self, output_type)["method"](**needed_args)
        # clear the output bundle to free up memory
        output_bundle.clear()
        del output_bundle
        torch.cuda.empty_cache()

    def _mask_special_tokens(self, input_tensor, special_tokens=None):
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

    def _extract_logits(
        self,
        logits,
        offset,
    ):
        for layer in self.layers:
            tensor = logits[layer - 1]
            if self.batch_writing:
                # output_file = self.logits["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)

                self.io_dispatcher.enqueue(
                    output_type="logits",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.logits["output_data"][layer].extend(tensor)

    def _extract_embeddings(
        self,
        representations,
        batch_labels,
        pooling_mask,
        offset,
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
                # output_file = self.embeddings["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="embeddings",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.embeddings["output_data"][layer].extend(tensor)

    def _extract_embeddings_unpooled(
        self,
        representations,
        batch_labels,
        offset,
    ):
        if not self.discard_padding:
            for layer in self.layers:
                tensor = torch.stack(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.batch_writing:
                    # output_file = self.embeddings_unpooled["output_data"][layer]
                    # self.write_batch_to_disk(output_file, tensor, offset)
                    self.io_dispatcher.enqueue(
                        output_type="embeddings_unpooled",
                        layer=layer,
                        head=None,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.embeddings_unpooled["output_data"][layer].extend(tensor)
        else:  # TODO remove padding tokens
            print("Feature not implemented yet")
            pass
            for layer in self.layers:
                if self.flatten:
                    self.embeddings_unpooled["output_data"][layer].extend(
                        [
                            representations[layer][i].flatten(start_dim=1)
                            for i in range(len(batch_labels))
                        ]
                    )
                else:
                    self.embeddings_unpooled["output_data"][layer].extend(
                        [representations[layer][i] for i in range(len(batch_labels))]
                    )

    def _extract_attention_matrices_all_heads(
        self,
        attention_matrices,
        batch_labels,
        offset,
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
                    # output_file = self.attention_matrices_all_heads["output_data"][
                    #    layer
                    # ][head]
                    # self.write_batch_to_disk(output_file, tensor, offset)
                    self.io_dispatcher.enqueue(
                        output_type="attention_matrices_all_heads",
                        layer=layer,
                        head=head,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.attention_matrices_all_heads["output_data"][layer][
                        head
                    ].extend(tensor)

    def _extract_attention_matrices_average_layer(
        self,
        attention_matrices,
        batch_labels,
        offset,
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
                # output_file = self.attention_matrices_average_layers["output_data"][
                #    layer
                # ]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="attention_matrices_average_layers",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.attention_matrices_average_layers["output_data"][layer].extend(
                    tensor
                )

    def _extract_attention_matrices_average_all(
        self,
        attention_matrices,
        batch_labels,
        offset,
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
            # output_file = self.attention_matrices_average_all["output_data"]
            # self.write_batch_to_disk(output_file, tensor, offset)
            self.io_dispatcher.enqueue(
                output_type="attention_matrices_average_all",
                layer=None,
                head=None,
                offset=offset,
                array=np.ascontiguousarray(
                    tensor.cpu().numpy()
                ),  # Ensure it's on CPU and NumPy
            )
        else:
            self.attention_matrices_average_all["output_data"].extend(tensor)

    def extract_cdr3_attention_matrices_average_all_heads(
        self,
        attention_matrices,
        batch_labels,
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

    def _extract_cdr3_attention_matrices_average_layer(
        self,
        attention_matrices,
        batch_labels,
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

    def _extract_cdr3_attention_matrices_average_all(
        self,
        attention_matrices,
        batch_labels,
    ):
        tensor = []
        for i, label in enumerate(batch_labels):
            start, end = self.get_cdr3_positions(label)
            tensor.extend(
                attention_matrices[:, i, :, start:end, start:end].mean(dim=(0, 1))
            )
        tensor = torch.stack(tensor)
        self.cdr3_attention_matrices_average_all["output_data"].extend(tensor)

    def _extract_cdr3(
        self,
        representations,
        cdr3_mask,
        offset,
    ):
        for layer in self.layers:
            tensor = torch.stack(
                [
                    (
                        (mask.unsqueeze(-1) * representations[layer][i]).sum(0)
                        / mask.unsqueeze(-1).sum(0)
                    )
                    for i, mask in enumerate(cdr3_mask)
                ]
            )
            if self.batch_writing:
                # output_file = self.cdr3_extracted["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="cdr3_extracted",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.cdr3_extracted["output_data"][layer].extend(tensor)

    def _prepare_tensor(self, data_list, flatten):
        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy()

    def export_to_disk(self):
        for output_type in self.output_types:
            print(f"Saving {output_type} representations...")

            output_data = getattr(self, output_type)["output_data"]
            output_dir = getattr(self, output_type)["output_dir"]
            flatten = self.flatten and "unpooled" in output_type

            if "attention_matrices" not in output_type and "logits" not in output_type:
                for layer in self.layers:
                    tensor = self._prepare_tensor(output_data[layer], flatten)
                    file_path = self._make_output_filepath(
                        output_type, output_dir, layer
                    )
                    np.save(file_path, tensor)
                    print(f"Saved {output_type} layer {layer} to {file_path}")

            elif "average_all" in output_type:
                tensor = self._prepare_tensor(output_data, self.flatten)
                file_path = self._make_output_filepath(output_type, output_dir)
                np.save(file_path, tensor)
                print(f"Saved {output_type} to {file_path}")

            elif "average_layer" in output_type:
                for layer in self.layers:
                    tensor = self._prepare_tensor(output_data[layer], self.flatten)
                    file_path = self._make_output_filepath(
                        output_type, output_dir, layer
                    )
                    np.save(file_path, tensor)
                    print(f"Saved {output_type} layer {layer} to {file_path}")

            elif "all_heads" in output_type:
                for layer in self.layers:
                    for head in range(self.num_heads):
                        tensor = self._prepare_tensor(
                            output_data[layer][head], self.flatten
                        )
                        file_path = self._make_output_filepath(
                            output_type, output_dir, layer, head
                        )
                        np.save(file_path, tensor)
                        print(
                            f"Saved {output_type} layer {layer} head {head + 1} to {file_path}"
                        )

            elif "logits" in output_type:
                for layer in self.layers:
                    tensor = self._prepare_tensor(output_data[layer], flatten=False)
                    file_path = self._make_output_filepath(
                        output_type, output_dir, layer
                    )
                    np.save(file_path, tensor)
                    print(f"Saved {output_type} layer {layer} to {file_path}")

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

    def _create_output_dirs(self):
        for output_type in self.output_types:
            output_type_path = os.path.join(self.output_path, output_type)
            if not os.path.exists(output_type_path):
                os.makedirs(output_type_path)

    def run(self):
        self._create_output_dirs()
        if self.batch_writing:
            print("Preallocating disk space...")
            self.memmap_registry = self.preallocate_disk_space()
            print("Preallocated disk space")
        print("Created output directories")

        print("Start embedding extraction")
        self.embed()
        print("Finished embedding extraction")

        print("Saving embeddings...")
        if not self.batch_writing:
            self.export_to_disk()

        self.export_sequence_indices()
