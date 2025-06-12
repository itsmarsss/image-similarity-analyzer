import numpy as np

def pixel_difference_score(orig: np.ndarray, swapped: np.ndarray) -> float:
    """
    Absolute per-pixel difference summed and normalized to [0,1].
    """
    assert orig.shape == swapped.shape, "Shapes must match"
    diff = np.abs(orig.astype(np.int32) - swapped.astype(np.int32))
    total_diff = diff.sum()
    h, w, c = orig.shape
    max_diff = h * w * c * 255
    return float(total_diff) / max_diff
