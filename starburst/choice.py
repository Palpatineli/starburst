from typing import Optional, Tuple
import numpy as np
from numpy.random import shuffle

def take_sample(array: np.ndarray, sample_size: int, sample_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if sample_idx is None:
        shuffle(array)
        return array[0: sample_size], sample_size
    array_len = array.shape[0]
    start_idx = (sample_idx * sample_size) % array_len
    end_idx = start_idx + sample_size
    if end_idx >= array_len:
        shuffle(array)
    return array[start_idx: end_idx], end_idx
