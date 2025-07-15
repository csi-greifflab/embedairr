import numpy as np
import torch
import inspect
import logging

logger = logging.getLogger("src.embedder_utils")


def precision_to_dtype(precision, framework):
    half_precision = ["float16", "16", "half"]
    full_precision = ["float32", "32", "full"]
    if precision in half_precision:
        return torch.float16 if framework == "torch" else np.float16
    if precision in full_precision:
        return torch.float32 if framework == "torch" else np.float32
    raise ValueError(
        f"Unsupported precision: {precision}. Supported values are {half_precision} or {full_precision}."
    )


def mask_special_tokens(input_tensor, special_tokens=None):
    if special_tokens is not None:
        mask = ~torch.isin(input_tensor, special_tokens)
    else:
        mask = (input_tensor != 0) & (input_tensor != 1) & (input_tensor != 2)
    return mask


def prepare_tensor(data_list, flatten):
    tensor = torch.stack(data_list, dim=0)
    if flatten:
        tensor = tensor.flatten(start_dim=1)
    return tensor.numpy()


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy()


# Extraction helpers -----------------------------------------------------

def extract_logits(embedder, logits, offset):
    for layer in embedder.layers:
        tensor = logits[layer - 1]
        if embedder.streaming_output:
            embedder.io_dispatcher.enqueue(
                output_type="logits",
                layer=layer,
                head=None,
                offset=offset,
                array=to_numpy(tensor),
            )
        else:
            embedder.logits["output_data"][layer].extend(tensor)


def extract_mean_pooled(embedder, representations, batch_labels, pooling_mask, offset):
    for layer in embedder.layers:
        tensor = torch.stack(
            [
                (pooling_mask[i].unsqueeze(-1) * representations[layer][i]).sum(0)
                / pooling_mask[i].unsqueeze(-1).sum(0)
                for i in range(len(batch_labels))
            ]
        )
        if embedder.streaming_output:
            embedder.io_dispatcher.enqueue(
                output_type="mean_pooled",
                layer=layer,
                head=None,
                offset=offset,
                array=to_numpy(tensor),
            )
        else:
            embedder.mean_pooled["output_data"][layer].extend(tensor)


def extract_per_token(embedder, representations, batch_labels, offset):
    if not embedder.discard_padding:
        for layer in embedder.layers:
            tensor = torch.stack([representations[layer][i] for i in range(len(batch_labels))])
            if embedder.flatten:
                tensor = tensor.flatten(start_dim=1)
            if embedder.streaming_output:
                embedder.io_dispatcher.enqueue(
                    output_type="per_token",
                    layer=layer,
                    head=None,
                    offset=offset,
                    array=np.ascontiguousarray(tensor.cpu().numpy()),
                )
            else:
                embedder.per_token["output_data"][layer].extend(tensor)
    else:
        logger.warning("Feature not implemented yet")
        for layer in embedder.layers:
            if embedder.flatten:
                embedder.per_token["output_data"][layer].extend(
                    [representations[layer][i].flatten(start_dim=1) for i in range(len(batch_labels))]
                )
            else:
                embedder.per_token["output_data"][layer].extend(
                    [representations[layer][i] for i in range(len(batch_labels))]
                )


def extract_attention_head(embedder, attention_matrices, batch_labels, offset):
    for layer in embedder.layers:
        for head in range(embedder.num_heads):
            tensor = torch.stack([attention_matrices[layer - 1, i, head] for i in range(len(batch_labels))])
            if embedder.flatten:
                tensor = tensor.flatten(start_dim=1)
            if embedder.streaming_output:
                embedder.io_dispatcher.enqueue(
                    output_type="attention_matrices_all_heads",
                    layer=layer,
                    head=head,
                    offset=offset,
                    array=np.ascontiguousarray(tensor.cpu().numpy()),
                )
            else:
                embedder.attention_head["output_data"][layer][head].extend(tensor)


def extract_attention_layer(embedder, attention_matrices, batch_labels, offset):
    for layer in embedder.layers:
        tensor = torch.stack([attention_matrices[layer - 1, i].mean(0) for i in range(len(batch_labels))])
        if embedder.flatten:
            tensor = tensor.flatten(start_dim=1)
        if embedder.streaming_output:
            embedder.io_dispatcher.enqueue(
                output_type="attention_matrices_average_layers",
                layer=layer,
                head=None,
                offset=offset,
                array=to_numpy(tensor),
            )
        else:
            embedder.attention_layer["output_data"][layer].extend(tensor)


def extract_attention_model(embedder, attention_matrices, batch_labels, offset):
    tensor = torch.stack([
        attention_matrices[:, i].mean(dim=(0, 1)) for i in range(len(batch_labels))
    ])
    if embedder.flatten:
        tensor = tensor.flatten(start_dim=1)
    if embedder.streaming_output:
        embedder.io_dispatcher.enqueue(
            output_type="attention_matrices_average_all",
            layer=None,
            head=None,
            offset=offset,
            array=np.ascontiguousarray(tensor.cpu().numpy()),
        )
    else:
        embedder.attention_model["output_data"].extend(tensor)


def extract_substring_pooled(embedder, representations, substring_mask, offset):
    for layer in embedder.layers:
        tensor = torch.stack(
            [
                (mask.unsqueeze(-1) * representations[layer][i]).sum(0) / mask.unsqueeze(-1).sum(0)
                for i, mask in enumerate(substring_mask)
            ]
        )
        if embedder.streaming_output:
            embedder.io_dispatcher.enqueue(
                output_type="substring_pooled",
                layer=layer,
                head=None,
                offset=offset,
                array=to_numpy(tensor),
            )
        else:
            embedder.substring_pooled["output_data"][layer].extend(tensor)
