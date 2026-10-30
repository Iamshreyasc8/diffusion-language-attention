"""Model-independent analysis. No GPU or model download is required."""
from itertools import combinations
import re
import numpy as np


def as_heatmap(value):
    """Accept a DAAM map, tensor, or array; return a validated 2-D float array."""
    value = getattr(value, "heatmap", value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("A heatmap must be a nonempty 2-D array.")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Heatmaps must contain finite, nonnegative values.")
    return array


def continuous_iou(first, second):
    """Raw-amplitude overlap, not a semantic score. Both empty maps return 0."""
    first, second = as_heatmap(first), as_heatmap(second)
    if first.shape != second.shape:
        raise ValueError("Heatmaps must have matching shapes.")
    # Common scaling prevents overflow without changing the ratio.
    scale = max(float(first.max()), float(second.max()))
    if scale == 0:
        return 0.0
    first, second = first / scale, second / scale
    return float(np.minimum(first, second).sum() / np.maximum(first, second).sum())


def binary_mask(value, percentile=90):
    """Threshold a normalized map; uniform maps have no localized region.

    Zero-valued pixels are excluded even when the percentile threshold is zero.
    Ties can retain more than (100 - percentile)% of pixels.
    """
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")
    array = as_heatmap(value)
    low, high = array.min(), array.max()
    if high == low:
        return np.zeros(array.shape, dtype=bool)
    normalized = (array - low) / (high - low)
    return (normalized > 0) & (normalized >= np.percentile(normalized, percentile))


def expression_words(expression):
    """Unique words in first-occurrence order; repeats are aggregated by DAAM."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a nonempty string.")
    words = list(dict.fromkeys(re.findall(r"\w+", expression.lower())))
    if not words:
        raise ValueError("Expression must contain at least one word.")
    return words


def validate_prompt(prompt, expression):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a nonempty string.")
    words = expression_words(expression)
    present = set(re.findall(r"\w+", prompt.lower()))
    missing = [word for word in words if word not in present]
    if missing:
        raise ValueError("Expression words missing from prompt: " + ", ".join(missing))
    return words


def collect_heatmaps(global_map, words):
    """Return real maps and explicit failures; never fabricate zero attention."""
    maps, missing = {}, {}
    for word in dict.fromkeys(words):
        try:
            maps[word] = as_heatmap(global_map.compute_word_heat_map(word))
        except (ValueError, IndexError) as error:
            missing[word] = str(error)
    return maps, missing


def pairwise_scores(maps):
    return [{"word_1": a, "word_2": b, "continuous_iou": continuous_iou(maps[a], maps[b])}
            for a, b in combinations(maps, 2)]
