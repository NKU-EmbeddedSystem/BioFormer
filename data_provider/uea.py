import torch

def collate_fn(data, max_len=None):
    """"
    Custom collate function to handle variable-length sequences in a batch.
    Args:
        data: list of tuples (features, label) where:
            features: Tensor of shape (seq_length, feat_dim)
            label: Tensor of shape (num_labels,)
        max_len: int, maximum sequence length to pad/truncate to. If None, uses the longest sequence in the batch.
    Returns:
        X: Tensor of shape (batch_size, padded_length, feat_dim) with padded/truncated sequences
        targets: Tensor of shape (batch_size,) with labels
        padding_masks: Boolean Tensor of shape (batch_size, padded_length), where True indicates valid time steps
    """

    batch_size = len(data)
    features, labels, subject= zip(*data)
    # features: tuple of (seq_length, feat_dim) tensors
    # labels: tuple of (num_labels,) tensors

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size,)
    subject = torch.stack(subject, dim=0)  # (batch_size,)
    # subject = subject - 1  # make subject IDs zero-indexed
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep
    # Stack subject IDs (just concatenate them)

    return X, targets.long(), padding_masks, subject.long()

def collate_fn_dep(data, max_len=None):

    batch_size = len(data)
    features, labels= zip(*data)
    # features: tuple of (seq_length, feat_dim) tensors
    # labels: tuple of (num_labels,) tensors

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size,)
    # subject = subject - 1  # make subject IDs zero-indexed
    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep
    # Stack subject IDs (just concatenate them)

    return X, targets.long(), padding_masks

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)  # convert to same type as lengths tensor
        .repeat(batch_size, 1)  # (batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )

def normalize_batch_ts(batch):
    """Normalize a batch of time-series data.

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (N, T, C).

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel of each sample.
    """
    # Calculate mean and std for each sample's each channel
    mean_values = batch.mean(axis=1, keepdims=True)  # Shape: (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)  # Shape: (N, 1, C)

    # Avoid division by zero by setting small std to 1
    std_values[std_values == 0] = 1.0

    # Perform standard normalization
    normalized_batch = (batch - mean_values) / std_values

    return normalized_batch
