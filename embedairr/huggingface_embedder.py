import time
import os
import torch
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset

from embedairr.base_embedder import BaseEmbedder
from transformers import T5EncoderModel, T5Tokenizer
from transformers import RoFormerTokenizer, RoFormerModel

# Set max_split_size_mb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class HuggingfaceEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)

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

    def load_data(self, sequences):
        """Tokenize sequences and create a DataLoader."""
        # Tokenize sequences
        print("Tokenizing and batching sequences...")
        tokens = self.tokenizer(
            list(sequences.values()),
            truncation=False,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=(
                False if self.disable_special_tokens else True
            ),  # TODO make it optional
            max_length=self.max_length,
        )

        # Extract input_ids and attention masks directly from the tokens
        input_ids = tokens["input_ids"]
        attention_masks = tokens["attention_mask"]

        # Create a dataset and a DataLoader
        dataset = TensorDataset(input_ids, attention_masks)
        data_loader = DataLoader(dataset, batch_size=self.batch_size)
        print("Finished tokenizing and batching sequences")

        return data_loader

    def embed(self):
        # Multithreading to overlap computation and writing
        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future = None  # To store the async write operation
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.data_loader):
                    start_time = time.time()
                    print(
                        f"Start embedding batch {batch_idx + 1} of {len(self.data_loader)}"
                    )
                    labels = list(self.sequences.keys())[
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
                        attention_matrices = torch.stack(outputs.attentions).to(
                            dtype=torch.float16
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

                    # Wait for the previous write to finish (if any)
                    # if future is not None:
                    #    future.result()  # Ensures previous write completed before reusing resources
                    output_bundle = {
                        "attention_matrices": attention_matrices,
                        "representations": representations,
                        "batch_labels": labels,
                        "pooling_mask": pooling_mask,
                        "batch_idx": batch_idx,
                    }
                    future = executor.submit(self.extract_batch, output_bundle)
                    futures.append(future)
                    end_time = time.time()
                    sequences_per_second = self.batch_size / (end_time - start_time)
                    estimated_time_remaining = (
                        len(self.sequences) - len(self.sequence_labels)
                    ) / sequences_per_second
                    print(
                        f"Processed {self.model_name}: {len(self.sequence_labels)} out of {len(self.sequences)} sequences ({sequences_per_second:.2f} sequences per second). Estimated time remaining: {estimated_time_remaining:.2f} seconds."
                    )
            for future in futures:
                future.result()
            # if future is not None:
            #    future.result()
            print("Finished extracting embeddings")


class Antiberta2Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences_gapped, self.sequences = self.fasta_to_dict(
            args.fasta_path, gaps=True
        )
        self.num_sequences = len(self.sequences_gapped)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = set(self.tokenizer.get_vocab().keys())
        self.check_input_tokens(self.valid_tokens, self.sequences_gapped, gaps=True)
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data(self.sequences_gapped)
        self.sequences = {
            sequence_id: sequence_aa.replace(" ", "")
            for sequence_id, sequence_aa in self.sequences_gapped.items()
        }
        self.set_output_objects()

    def initialize_model(self, model_link="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")
        tokenizer = RoFormerTokenizer.from_pretrained(model_link)
        model = RoFormerModel.from_pretrained(model_link).to(device)
        model.eval()
        # disable eos and bos
        model.config
        num_heads = model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size


class T5Embedder(HuggingfaceEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences, _ = self.fasta_to_dict(args.fasta_path, gaps=False)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_link)
        self.valid_tokens = self.get_valid_tokens()
        self.check_input_tokens(self.valid_tokens, self.sequences, gaps=False)
        self.special_tokens = torch.tensor(
            self.tokenizer.all_special_ids, device=self.device, dtype=torch.int8
        )
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data(self.sequences)
        self.set_output_objects()

    def get_valid_tokens(self):
        valid_tokens = set(
            k[1:] if k.startswith("▁") else k
            for k in set(self.tokenizer.get_vocab().keys())
        )
        return valid_tokens

    def initialize_model(self, model_link="Rostlab/prot_t5_xl_half_uniref50-enc"):
        """Initialize the model, tokenizer, and device."""

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Transferred model to GPU")
        else:
            device = torch.device("cpu")
            print("No GPU available, using CPU")
        tokenizer = T5Tokenizer.from_pretrained(model_link)
        model = T5EncoderModel.from_pretrained(model_link).to(device)
        model.eval()
        num_heads = model.config.num_heads
        num_layers = model.config.num_layers
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size
