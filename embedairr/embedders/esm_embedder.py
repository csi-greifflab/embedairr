import torch
from esm import pretrained
from embedairr.embedders.base_embedder import BaseEmbedder
import embedairr.utils

# torch.set_default_dtype(torch.float16)


class ESMEmbedder(BaseEmbedder):
    def __init__(self, args):
        super().__init__(args)
        self.sequences = embedairr.utils.fasta_to_dict(args.fasta_path)
        self.num_sequences = len(self.sequences)
        (
            self.model,
            self.alphabet,
            self.num_heads,
            self.num_layers,
            self.embedding_size,
            self.prepend_bos,
            self.append_eos,
        ) = self.initialize_model(self.model_name)
        self.valid_tokens = set(self.alphabet.all_toks)
        embedairr.utils.check_input_tokens(
            self.valid_tokens, self.sequences, self.model_name
        )
        self.special_tokens = self.get_special_tokens()
        self.layers = self.load_layers(self.layers)
        self.data_loader, self.max_length = self.load_data(
            self.sequences, self.cdr3_dict
        )  # tokenize and batch sequences and update max_length
        self.set_output_objects()

    def initialize_model(self, model_name):
        """Initialize the model, tokenizer"""
        #  Loading the pretrained model and alphabet for tokenization
        print("Loading model...")
        # model, alphabet = pretrained.load_model_and_alphabet(model_name)
        model, alphabet = pretrained.load_model_and_alphabet_hub(model_name)
        model.eval()  # Setting the model to evaluation mode
        if not self.disable_special_tokens:
            model.append_eos = True if not model_name.startswith("esm1") else False
            model.prepend_bos = True
        else:
            model.append_eos = False
            model.prepend_bos = False

        num_heads = model.layers[0].self_attn.num_heads
        num_layers = len(model.layers)
        embedding_size = (
            model.embed_tokens.embedding_dim
            if model_name.startswith("esm1")
            else model.embed_dim
        )

        # Moving the model to GPU if available for faster processing
        if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print("No GPU available, using CPU")
        return (
            model,
            alphabet,
            num_heads,
            num_layers,
            embedding_size,
            model.prepend_bos,
            model.append_eos,
        )

    def get_special_tokens(self):
        special_tokens = self.alphabet.all_special_tokens
        special_token_ids = torch.tensor(
            [self.alphabet.tok_to_idx[tok] for tok in special_tokens],
            device=self.device,
            dtype=torch.int8,
        )
        return special_token_ids

    def load_layers(self, layers):
        if not layers:
            layers = list(range(1, self.model.num_layers + 1))
            return layers
        # Checking if the specified representation layers are valid
        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers for i in layers
        )
        layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1)
            for i in layers
        ]
        return layers

    def load_data(self, sequences, cdr3_dict=None):
        # Creating a dataset from the input fasta file
        print("Tokenizing sequences...")
        dataset = embedairr.utils.ESMDataset(
            sequences,
            cdr3_dict,
            self.context,
            self.alphabet,
            self.max_length,
            self.prepend_bos,
            self.append_eos,
        )
        # Generating batch indices based on token count
        print("Generating batches...")
        batches = embedairr.utils.TokenBudgetBatchSampler(dataset, self.batch_size)
        # DataLoader to iterate through batches efficiently
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batches, collate_fn=dataset.safe_collate
        )
        print("Data loaded")
        # Getting the maximum sequence length from the dataset
        max_length = dataset.get_max_encoded_length()
        return data_loader, max_length

    def compute_outputs(
        self,
        model,
        toks,
        attention_mask,
        return_embeddings,
        return_contacts,
        return_logits,
    ):
        outputs = model(
            toks,
            repr_layers=self.layers,
            return_contacts=return_contacts,
        )
        if return_logits:
            logits = (
                outputs["logits"].to(dtype=torch.float16, device="cpu").permute(2, 0, 1)
            )  # permute to match the shape of the representations
            torch.cuda.empty_cache()
        else:
            logits = None

        if return_contacts:
            attention_matrices = (
                outputs["attentions"].to(dtype=torch.float16).permute(1, 0, 2, 3, 4)
            ).cpu()  # permute to match the shape of the representations
            torch.cuda.empty_cache()
        else:
            attention_matrices = None
        # Extracting layer representations and moving them to CPU
        if self.return_embeddings:
            representations = {
                layer: t.to(dtype=torch.float16).cpu()
                for layer, t in outputs["representations"].items()
            }
            torch.cuda.empty_cache()
        else:
            representations = None
        return logits, representations, attention_matrices
