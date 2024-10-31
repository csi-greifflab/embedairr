import torch
from esm import FastaBatchedDataset, pretrained
from embedairr.base_embedder import BaseEmbedder


class ESM2Embedder(BaseEmbedder):
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
        super().__init__(
            fasta_path,
            model_name,
            output_path,
            cdr3_path,
            context,
            layers,
            output_types,
        )

        self.sequences = self.fasta_to_dict(fasta_path)
        self.model, self.alphabet, self.num_heads = self.initialize_model()
        self.layers = self.load_layers(layers)
        self.data_loader = self.load_data()
        self.set_output_objects()

    def initialize_model(self, model_name="esm2_t33_650M_UR50D"):
        """Initialize the model, tokenizer"""
        #  Loading the pretrained model and alphabet for tokenization
        print("Loading model...")
        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.esm2_t33_650M_UR50D()
        model.eval()  # Setting the model to evaluation mode
        num_heads = model.layers[0].self_attn.num_heads

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        return model, alphabet, num_heads

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

    def load_data(self, batch_size=30000):
        print("Loading and batching input sequences...")
        # Creating a dataset from the input fasta file
        dataset = FastaBatchedDataset.from_file(self.fasta_path)
        # Generating batch indices based on token count
        batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.alphabet.get_batch_converter(),
            batch_sampler=batches,
        )
        print(f"Read {self.fasta_path} with {len(dataset)} sequences")
        return data_loader

    def embed(self):
        with torch.no_grad():
            for labels, strs, toks in self.data_loader:
                if self.device == torch.device("cuda"):
                    toks = toks.to(device="cuda", non_blocking=True)
                out = self.model(
                    toks, repr_layers=self.layers, return_contacts=self.return_contacts
                )
                # Extracting layer representations and moving them to CPU
                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }
                self.sequence_labels.extend(labels)
                self.extract_batch(out, representations, labels, strs)
        print("Finished extracting embeddings")
