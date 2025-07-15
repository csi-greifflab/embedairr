import os
import csv
import torch
import re
import numpy as np
from numpy.lib.format import open_memmap
import inspect
import gc
from pepe.utils import MultiIODispatcher, check_disk_free_space
from pepe.embedders.components import EmbeddingProcessor, StreamingIO
from alive_progress import alive_bar
import time
from pathlib import Path
import logging

logger = logging.getLogger("src.embedders.base_embedder")


class BaseEmbedder:
    def __init__(self, args):
        self.fasta_path = args.fasta_path
        self.model_link = args.model_name
        self.disable_special_tokens = args.disable_special_tokens
        if (
            self.model_link.endswith(".pt")
            or self.model_link.endswith(".pth")
            or self.model_link.startswith("custom:")
            or (
                os.path.exists(self.model_link)
                and (os.path.isfile(self.model_link) or os.path.isdir(self.model_link))
            )
        ):
            self.model_name = Path(
                self.model_link
            ).stem  # Use the stem of the model link as the model name
        else:
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
        self.substring_path = args.substring_path
        self.context = args.context
        self.layers = (
            [j for i in args.layers for j in i] if args.layers != [None] else None
        )
        self.substring_dict = (
            self._load_substrings(args.substring_path) if args.substring_path else None
        )
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        if torch.cuda.is_available() and args.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.output_types = args.extract_embeddings
        self.discard_padding = args.discard_padding
        self.flatten = args.flatten
        self.return_embeddings = False
        self.return_contacts = False
        self.return_logits = False
        for output_type in self.output_types:
            if "pooled" in output_type or "per_token" in output_type:
                self.return_embeddings = True
            if "attention" in output_type:
                self.return_contacts = True
            if output_type == "logits":
                self.return_logits = True
        self.streaming_output = args.streaming_output
        self.num_workers = args.num_workers if self.streaming_output else 1
        self.max_in_flight = self.num_workers * 2
        self.flush_batches_after = args.flush_batches_after * 1024**2  # in bytes
        self.precision = args.precision
        # self.log_memory = args.log_memory # TODO implement memory logging

        # Set up checkpoint directory for crash recovery
        self.checkpoint_dir = self.output_path
        
        # Initialize StreamingIO
        self.streaming_io = StreamingIO(
            output_path=self.output_path,
            output_prefix=self.output_prefix,
            model_name=self.model_name,
            output_types=self.output_types,
            precision=self.precision,
            streaming_output=self.streaming_output,
            num_workers=self.num_workers,
            flush_batches_after=self.flush_batches_after,
            layers=self.layers,
            num_heads=getattr(self, 'num_heads', None),
            flatten=self.flatten,
            fasta_path=self.fasta_path,
        )

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
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_logits,
            "output_dir": os.path.join(self.output_path, "logits"),
            "shape": (
                self.num_sequences,  # type: ignore
                self.max_length,
            ),
        }
        self.mean_pooled = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_mean_pooled,
            "output_dir": os.path.join(self.output_path, "mean_pooled"),
            "shape": (self.num_sequences, self.embedding_size),  # type: ignore
        }
        self.per_token = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_per_token,
            "output_dir": os.path.join(self.output_path, "per_token"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_length,
                    self.embedding_size,  # type: ignore
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_length * self.embedding_size,  # type: ignore
                )
            ),
        }
        self.substring_pooled = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_substring_pooled,
            "output_dir": os.path.join(self.output_path, "substring_pooled"),
            "shape": (self.num_sequences, self.embedding_size),  # type: ignore
        }
        self.attention_head = {
            "output_data": {
                layer: {head: [] for head in range(self.num_heads)}  # type: ignore
                for layer in self.layers  # type: ignore
            },
            "method": self._extract_attention_head,
            "output_dir": os.path.join(self.output_path, "attention_head"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_length**2,
                )
            ),
        }
        self.attention_layer = {
            "output_data": {layer: [] for layer in self.layers},  # type: ignore
            "method": self._extract_attention_layer,
            "output_dir": os.path.join(self.output_path, "attention_layer"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_length**2,
                )
            ),
        }
        self.attention_model = {
            "output_data": [],
            "method": self._extract_attention_model,
            "output_dir": os.path.join(self.output_path, "attention_model"),
            "shape": (
                (
                    self.num_sequences,  # type: ignore
                    self.max_length,
                    self.max_length,
                )
                if not self.flatten
                else (
                    self.num_sequences,  # type: ignore
                    self.max_length**2,
                )
            ),
        }

    # When changes made here, also update base_embedder.py BaseEmbedder.set_output_objects() method.
    def _get_output_types(self, args):
        output_types = []

        options_mapping = {
            "per_token": "per_token",
            "mean_pooled": "mean_pooled",
            "substring_pooled": "substring_pooled",
            "attention_head": "attention_head",
            "attention_layer": "attention_layer",
            "attention_model": "attention_model",
            "logits": "logits",
        }

        for option in args.extract_embeddings:
            if option in options_mapping:
                output_type = options_mapping[option]
                if output_type not in output_types:
                    output_types.append(output_type)

        return output_types

    def _make_output_filepath(self, output_type, output_dir, layer=None, head=None):
        base = f"{self.output_prefix}_{self.model_name}_{output_type}"
        if layer is not None:
            base += f"_layer_{layer}"
        if head is not None:
            base += f"_head_{head + 1}"
        return os.path.join(output_dir, base + ".npy")

    def preallocate_disk_space(self):
        # Create output data registry for StreamingIO
        output_data_registry = {}
        for output_type in self.output_types:
            output_data_registry[output_type] = {
                "output_data": getattr(self, output_type)["output_data"],
                "output_dir": getattr(self, output_type)["output_dir"],
                "shape": getattr(self, output_type)["shape"],
            }
        
        # Use StreamingIO to preallocate disk space
        return self.streaming_io.preallocate_disk_space(
            output_data_registry,
            self.num_sequences,  # type: ignore
            self.max_length,
            self.embedding_size,  # type: ignore
        )

    def _load_substrings(self, substring_path):
        """Load substrings and store in a dictionary."""
        if substring_path:
            with open(substring_path) as f:
                reader = csv.reader(f)  # skip header
                next(reader)
                substring_dict = {rows[0]: rows[1] for rows in reader}
            return substring_dict
        else:
            return None

    def _safe_compute(self, toks, attention_mask):
        """
        Try to run compute_outputs; on OOM, empty cache, split in half,
        recurse on each half, then concatenate.
        """
        try:
            return self._compute_outputs(  # type: ignore
                self.model,  # type: ignore
                toks,
                attention_mask,
                self.return_embeddings,
                self.return_contacts,
                self.return_logits,
            )
        except torch.OutOfMemoryError:
            logger.error("[GPU memory overflow] Decreasing batch size and retrying...")
            torch.cuda.empty_cache()
            B = toks.size(0)
            if B == 1:
                # can’t split anymore
                logger.error("OOM on single sample!")
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
                torch.cat([o[0] for o in outs], dim=0) if self.return_logits else None  # type: ignore
            )
            representations = (
                torch.cat([o[1] for o in outs], dim=0)  # type: ignore
                if self.return_embeddings
                else None
            )
            attention_matrices = (
                torch.cat([o[2] for o in outs], dim=0) if self.return_contacts else None  # type: ignore
            )
            return logits, representations, attention_matrices

    def embed(self):
        if self.streaming_output:
            # Initialize streaming I/O
            self.streaming_io.initialize_streaming_io()
            
            # Check if we're resuming from a checkpoint
            resume_info = self.streaming_io.get_resume_info()
            if resume_info:
                logger.info(f"Resuming from checkpoint: {resume_info}")

        with alive_bar(
            len(self.sequences),  # type: ignore
            title=f"{self.model_name}: Generating embeddings ...",
        ) as bar, torch.no_grad():
            offset = 0
            for (
                labels,
                strs,
                toks,
                attention_mask,
                substring_mask,
            ) in self.data_loader:  # type: ignore
                toks = toks.to(self.device, non_blocking=True)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)
                pooling_mask = self._mask_special_tokens(
                    toks, self.special_tokens  # type: ignore
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
                    "substring_mask": substring_mask,
                    "offset": offset,
                    "special_tokens": not self.disable_special_tokens,
                }
                if self.streaming_output:
                    # Apply backpressure if write queue is too full
                    while self.streaming_io.get_queue_fullness() > 0.6:
                        logger.warning(
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

            if self.streaming_output:
                self.streaming_io.stop_streaming()

            logger.info("Finished extracting embeddings")

        # After successful completion, clean up the checkpoint file
        if self.streaming_output:
            self._cleanup_checkpoint()

    def _cleanup_checkpoint(self):
        """Clean up the checkpoint file after successful completion."""
        self.streaming_io.cleanup_checkpoint()

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

    def get_substring_positions(self, label, special_tokens, context=0):
        """Get the start and end positions of the substring in the full sequence."""
        full_sequence = self.sequences[label]  # type: ignore

        try:
            substring = self.substring_dict[label]  # type: ignore
        except KeyError:
            SystemExit(f"No matching substring found for {label}")
        # remove '-' from substring
        substring = substring.replace("-", "")

        # get position of substring in sequence
        start = max(full_sequence.find(substring) - context, 0) + int(
            special_tokens
        )
        end = (
            min(start + len(substring) + context, len(full_sequence))
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
        for layer in self.layers:  # type: ignore
            tensor = logits[layer - 1]
            if self.streaming_output:
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

    def _extract_mean_pooled(
        self,
        representations,
        batch_labels,
        pooling_mask,
        offset,
    ):
        for layer in self.layers:  # type: ignore
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
            if self.streaming_output:
                # output_file = self.embeddings["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="mean_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.mean_pooled["output_data"][layer].extend(tensor)

    def _extract_per_token(
        self,
        representations,
        batch_labels,
        offset,
    ):
        if not self.discard_padding:
            for layer in self.layers:  # type: ignore
                tensor = torch.stack(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.streaming_output:
                    # output_file = self.per_token["output_data"][layer]
                    # self.write_batch_to_disk(output_file, tensor, offset)
                    self.io_dispatcher.enqueue(
                        output_type="per_token",
                        layer=layer,
                        head=None,
                        offset=offset,
                        array=np.ascontiguousarray(
                            tensor.cpu().numpy()
                        ),  # Ensure it's on CPU and NumPy
                    )
                else:
                    self.per_token["output_data"][layer].extend(tensor)
        else:  # TODO remove padding tokens
            logger.warning("Feature not implemented yet")
            pass
            for layer in self.layers:  # type: ignore
                if self.flatten:
                    self.per_token["output_data"][layer].extend(
                        [
                            representations[layer][i].flatten(start_dim=1)
                            for i in range(len(batch_labels))
                        ]
                    )
                else:
                    self.per_token["output_data"][layer].extend(
                        [representations[layer][i] for i in range(len(batch_labels))]
                    )

    def _extract_attention_head(
        self,
        attention_matrices,
        batch_labels,
        offset,
    ):
        for layer in self.layers:  # type: ignore
            for head in range(self.num_heads):  # type: ignore
                tensor = torch.stack(
                    [
                        attention_matrices[layer - 1, i, head]
                        for i in range(len(batch_labels))
                    ]
                )
                if self.flatten:
                    tensor = tensor.flatten(start_dim=1)
                if self.streaming_output:
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
                    self.attention_head["output_data"][layer][
                        head
                    ].extend(tensor)

    def _extract_attention_layer(
        self,
        attention_matrices,
        batch_labels,
        offset,
    ):
        for layer in self.layers:  # type: ignore
            tensor = torch.stack(
                [
                    attention_matrices[layer - 1, i].mean(0)
                    for i in range(len(batch_labels))
                ]
            )
            if self.flatten:
                tensor = tensor.flatten(start_dim=1)
            if self.streaming_output:
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
                self.attention_layer["output_data"][layer].extend(
                    tensor
                )

    def _extract_attention_model(
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
        if self.streaming_output:
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
            self.attention_model["output_data"].extend(tensor)

    def _extract_substring_pooled(
        self,
        representations,
        substring_mask,
        offset,
    ):
        for layer in self.layers:  # type: ignore
            tensor = torch.stack(
                [
                    (
                        (mask.unsqueeze(-1) * representations[layer][i]).sum(0)
                        / mask.unsqueeze(-1).sum(0)
                    )
                    for i, mask in enumerate(substring_mask)
                ]
            )
            if self.streaming_output:
                # output_file = self.substring_pooled["output_data"][layer]
                # self.write_batch_to_disk(output_file, tensor, offset)
                self.io_dispatcher.enqueue(
                    output_type="substring_pooled",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=self._to_numpy(tensor),  # Ensure it's on CPU and NumPy
                )
            else:
                self.substring_pooled["output_data"][layer].extend(tensor)

    def _prepare_tensor(self, data_list, flatten):
        tensor = torch.stack(data_list, dim=0)
        if flatten:
            tensor = tensor.flatten(start_dim=1)
        return tensor.numpy()

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().contiguous().numpy()

    def export_to_disk(self):
        # Create output data registry for StreamingIO
        output_data_registry = {}
        for output_type in self.output_types:
            output_data_registry[output_type] = {
                "output_data": getattr(self, output_type)["output_data"],
                "output_dir": getattr(self, output_type)["output_dir"],
            }
        
        # Use StreamingIO to export to disk
        self.streaming_io.export_to_disk(
            output_data_registry,
            self._prepare_tensor
        )

    def export_sequence_indices(self):
        """Save sequence indices to a CSV file."""
        self.streaming_io.export_sequence_indices(self.sequence_labels)

    def _create_output_dirs(self):
        self.streaming_io.create_output_dirs()

    def run(self):
        self._create_output_dirs()
        if self.streaming_output:
            logger.info("Preallocating disk space...")
            self.memmap_registry = self.preallocate_disk_space()
            logger.info("Preallocated disk space")
        logger.info("Created output directories")

        logger.info("Start embedding extraction")
        self.embed()
        logger.info("Finished embedding extraction")

        logger.info("Saving embeddings...")
        if not self.streaming_output:
            self.export_to_disk()

        self.export_sequence_indices()

        # Final cleanup of checkpoint file (in case embed() didn't handle it)
        if self.streaming_output:
            self._cleanup_checkpoint()

        logger.info("Pipeline completed successfully!")
