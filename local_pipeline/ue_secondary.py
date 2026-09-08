import numpy as np
import torch


def loo_keep_indices(predictions):
    """Return indices retained after dropping one LOO center-distance outlier."""
    if predictions.ndim != 5 or predictions.shape[2:] != (2, 2, 2):
        raise ValueError("predictions must have shape [batch, samples, 2, 2, 2]")
    if predictions.shape[1] < 3:
        raise ValueError("LOO outlier removal requires at least three predictions")

    centers = predictions.mean(dim=(-1, -2))
    center_sum = centers.sum(dim=1, keepdim=True)
    loo_means = (center_sum - centers) / (centers.shape[1] - 1)
    outlier_indices = torch.linalg.vector_norm(centers - loo_means, dim=-1).argmax(dim=1)

    all_indices = torch.arange(predictions.shape[1], device=predictions.device)
    return torch.stack([all_indices[all_indices != index] for index in outlier_indices])


def gather_predictions(predictions, indices):
    if predictions.shape[0] != indices.shape[0]:
        raise ValueError("prediction and index batch dimensions must match")
    gather_index = indices.view(indices.shape[0], indices.shape[1], 1, 1, 1)
    return predictions.gather(1, gather_index.expand(-1, -1, 2, 2, 2))


def combined_uncertainty(primary_predictions, secondary_predictions, keep_indices=None):
    """Calculate coordinate-wise sample std over retained primary and secondary runs."""
    if keep_indices is not None:
        primary_predictions = gather_predictions(primary_predictions, keep_indices)
    combined = torch.cat((primary_predictions, secondary_predictions), dim=1)
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
