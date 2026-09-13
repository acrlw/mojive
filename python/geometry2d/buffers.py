"""Array snapshots whose ownership and immutability survive frame publication."""

import numpy as np


def snapshot_array(array: np.ndarray) -> np.ndarray:
    """Copy into an immutable bytes owner; callers cannot re-enable writes."""
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
