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
        self.num_sequences = len(self.sequences_gapped)
        (
            self.model,
            self.tokenizer,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model("alchemab/antiberta2-cssp")
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
        pass

    def initialize_model(self, model_name="alchemab/antiberta2-cssp"):
        """Initialize the model, tokenizer, and device."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device type: {device}")
        tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        model = RoFormerModel.from_pretrained(model_name).to(device)
        model.eval()
        # disable eos and bos
        model.config
        num_heads = model.encoder.layer[0].attention.self.num_attention_heads
        num_layers = len(model.encoder.layer)
        embedding_size = model.config.hidden_size
        return model, tokenizer, num_heads, num_layers, embedding_size

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
            add_special_tokens=False,  # TODO make it optional
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

                self.extract_batch(
                    attention_matrices,
                    representations,
                    labels,
                    batch_sequences,
                    pooling_mask,
                    batch_idx,
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
