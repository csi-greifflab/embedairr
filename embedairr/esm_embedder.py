import torch
import time
from concurrent.futures import ThreadPoolExecutor
from esm import FastaBatchedDataset, pretrained
from embedairr.base_embedder import BaseEmbedder

# torch.set_default_dtype(torch.float16)


class ESMEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences, self.sequences_padded = self.fasta_to_dict(
            args.fasta_path, padding=True
        )
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.alphabet,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
        ) = self.initialize_model(self.model_name)
        self.valid_tokens = set(self.alphabet.all_toks)
        self.check_input_tokens(self.valid_tokens, self.sequences)
        self.special_tokens = self.get_special_tokens()
        self.layers = self.load_layers(self.layers)
        self.data_loader = self.load_data()
        self.set_output_objects()
        pass

    def get_special_tokens(self):
        special_tokens = self.alphabet.all_special_tokens
        special_token_ids = torch.tensor(
            [self.alphabet.tok_to_idx[tok] for tok in special_tokens],
            device=self.device,
            dtype=torch.int8,
        )
        return special_token_ids

    def load_layers(self, layers):
        # Checking if the specified representation layers are valid
        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers for i in layers
        )
        layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1)
            for i in layers
        ]
        return layers

    def load_data(self):
        # Creating a dataset from the input fasta file
        print("Tokenizing and batching sequences...")
        dataset = FastaBatchedDataset(
            sequence_strs=self.sequences_padded.values(),
            sequence_labels=self.sequences_padded.keys(),
        )
        # Generating batch indices based on token count
        batches = dataset.get_batch_indices(self.batch_size, extra_toks_per_seq=1)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        print("Data loaded")
        return data_loader

    def embed(self):
        # Multithreading to overlap computation and writing
        futures = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future = None  # To store the async write operation
            with torch.no_grad():
                for batch_idx, (labels, strs, toks) in enumerate(self.data_loader):
                    print(
                        f"Start embedding batch {batch_idx + 1} of {len(self.data_loader)}"
                    )
                    start_time = time.time()
                    if self.device == torch.device("cuda"):
                        toks = toks.to(device="cuda", non_blocking=True)
                    pooling_mask = self.mask_special_tokens(
                        toks, self.special_tokens
                    ).cpu()  # mask special tokens to avoid diluting signal when pooling embeddings
                    outputs = self.model(
                        toks,
                        repr_layers=self.layers,
                        return_contacts=self.return_contacts,
                    )
                    if self.return_contacts:
                        attention_matrices = (
                            outputs["attentions"]
                            .to(dtype=torch.float16)
                            .permute(1, 0, 2, 3, 4)
                        ).cpu()  # permute to match the shape of the representations
                    else:
                        attention_matrices = None
                    # Extracting layer representations and moving them to CPU
                    if self.return_embeddings:
                        representations = {
                            layer: t.to(dtype=torch.float16).cpu()
                            for layer, t in outputs["representations"].items()
                        }
                    else:
                        representations = None
                    self.sequence_labels.extend(labels)

                    # Wait for the previous write to finish (if any)
                    # if future is not None:
                    #    future.result()  # Ensures previous write completed before reusing resources

                    future = executor.submit(
                        self.extract_batch,
                        attention_matrices,
                        representations,
                        labels,
                        strs,
                        pooling_mask,
                        batch_idx,
                    )
                    futures.append(future)
                    # print total progress
                    end_time = time.time()
                    sequences_per_second = len(labels) / (end_time - start_time)
                    estimated_remaining_time = (
                        len(self.sequences) - len(self.sequence_labels)
                    ) / sequences_per_second
                    print(
                        f"{self.model_name}: {len(self.sequence_labels)} sequences of {len(self.sequences)} processed ({sequences_per_second:.2f} sequences/s). Estimated remaining time: {estimated_remaining_time:.2f} s"
                    )

        print("Finishing writing embeddings...")
        for future in futures:
            future.result()
        # if future is not None:
        #    future.result()  # Ensures the last write completed before exiting
        print("Finished extracting embeddings")


class ESM1Embedder(ESMEmbedder):
    def __init__(self, args):
        super().__init__(args)

    def initialize_model(self, model_name):
        """Initialize the model, tokenizer"""
        #  Loading the pretrained model and alphabet for tokenization
        print("Loading model...")
        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # Setting the model to evaluation mode
        if not self.disable_special_tokens:
            model.append_eos = True
            model.prepend_bos = True
        else:
            model.append_eos = False
            model.prepend_bos = False

        num_heads = model.layers[0].self_attn.num_heads
        num_layers = len(model.layers)
        embedding_size = model.embed_tokens.embedding_dim

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print("No GPU available, using CPU")
        return model, alphabet, num_heads, num_layers, embedding_size


class ESM2Embedder(ESMEmbedder):
    def __init__(self, args):
        super().__init__(args)

    def initialize_model(self, model_name):
        """Initialize the model, tokenizer"""
        #  Loading the pretrained model and alphabet for tokenization
        print("Loading model...")
        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # Setting the model to evaluation mode
        if not self.disable_special_tokens:
            model.append_eos = True
            model.prepend_bos = True
        else:
            model.append_eos = False
            model.prepend_bos = False

        num_heads = model.layers[0].self_attn.num_heads
        num_layers = len(model.layers)
        embedding_size = model.embed_dim

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print("No GPU available, using CPU")
        return model, alphabet, num_heads, num_layers, embedding_size
