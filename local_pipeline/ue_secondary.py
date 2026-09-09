import numpy as np
import torch


def loo_keep_indices(predictions, n_remove=1):
    """Return original indices retained after iterative LOO center filtering."""
    if predictions.ndim != 5 or predictions.shape[2:] != (2, 2, 2):
        raise ValueError("predictions must have shape [batch, samples, 2, 2, 2]")
    sample_count = predictions.shape[1]
    if n_remove < 0 or n_remove > sample_count - 2:
        raise ValueError("n_remove must leave at least two predictions")

    batch_keep = []
    for batch_index in range(predictions.shape[0]):
        keep = torch.arange(sample_count, device=predictions.device)
        for _ in range(n_remove):
            current = predictions[batch_index, keep]
            centers = current.mean(dim=(-1, -2))
            center_sum = centers.sum(dim=0, keepdim=True)
            loo_means = (center_sum - centers) / (centers.shape[0] - 1)
            remove_position = torch.linalg.vector_norm(
                centers - loo_means, dim=-1
            ).argmax()
            keep = torch.cat((keep[:remove_position], keep[remove_position + 1:]))
        batch_keep.append(keep)
    return torch.stack(batch_keep)


def gather_predictions(predictions, indices):
    if predictions.shape[0] != indices.shape[0]:
        raise ValueError("prediction and index batch dimensions must match")
    gather_index = indices.view(indices.shape[0], indices.shape[1], 1, 1, 1)
    return predictions.gather(1, gather_index.expand(-1, -1, 2, 2, 2))


def combined_uncertainty(
    primary_predictions,
    secondary_predictions=None,
    keep_indices=None,
    n_remove=0,
):
    """Calculate coordinate-wise sample std over a optionally filtered pool."""
    if keep_indices is not None:
        primary_predictions = gather_predictions(primary_predictions, keep_indices)
    combined = primary_predictions
    if secondary_predictions is not None:
        combined = torch.cat((combined, secondary_predictions), dim=1)
    if n_remove:
        combined = gather_predictions(combined, loo_keep_indices(combined, n_remove))
    if combined.shape[1] < 2:
        raise ValueError("uncertainty requires at least two predictions")
    return combined.std(dim=1)


def sample_unique_crop_origins(rng, count, max_offset, excluded=None):
    """Sample unique integer (x, y) origins inside an inclusive square range."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if max_offset < 0:
        raise ValueError("crop size cannot exceed frame size")

    excluded_set = set(excluded or ())
    available = (max_offset + 1) ** 2 - len(excluded_set)
    if count > available:
        raise ValueError("not enough unique crop origins are available")

    sampled = []
    used = set(excluded_set)
    while len(sampled) < count:
        origin = (
            int(rng.integers(0, max_offset + 1)),
            int(rng.integers(0, max_offset + 1)),
        )
        if origin not in used:
            used.add(origin)
            sampled.append(origin)
    return np.asarray(sampled, dtype=np.int64)


def sample_cross_crop_origins(rng, max_offset, excluded=None):
    """Sample one crop in each diagonal quadrant around the center."""
    if max_offset < 2:
        raise ValueError("cross crops require at least two pixels of crop movement")

    center = max_offset // 2
    min_distance = min(center - 1, 30)
    excluded_set = set(excluded or ())
    directions = ((-1, 1), (1, 1), (-1, -1), (1, -1))
    sampled = []

    for x_sign, y_sign in directions:
        candidates = [
            (center + x_sign * dx, center + y_sign * dy)
            for dx in range(min_distance, center)
            for dy in range(min_distance, center)
        ]
        candidates = [origin for origin in candidates if origin not in excluded_set]
        if not candidates:
            raise ValueError("no unused crop origin is available for the cross pattern")
        origin = candidates[int(rng.integers(0, len(candidates)))]
        sampled.append(origin)
        excluded_set.add(origin)

    return np.asarray(sampled, dtype=np.int64)
