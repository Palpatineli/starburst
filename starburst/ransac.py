"""for now it's hardcoded to fit ellipse"""
from typing import TypeVar, Tuple, Type
import numpy as np
from numpy.linalg import LinAlgError
from numpy.random import shuffle

Ellipse = Tuple[complex, complex, float]
MAX_ITER_INITIAL = 10000

class Processor(object):
    def preprocess(self, samples: np.ndarray) -> np.ndarray:
        """returns the same format as samples"""
        raise NotImplementedError

    def threshold(self) -> float:
        raise NotImplementedError

class BaseModel(object):
    min_sample_no = 0

    @classmethod
    def solve(cls, samples: np.ndarray, processor=None):
        raise NotImplementedError

    @classmethod
    def fit(cls, samples: np.ndarray, processor=None):
        raise NotImplementedError

    def evaluate(self, samples: np.ndarray) -> np.ndarray:
        """returns a value for each sample, equals its distance from the modeled shape"""
        raise NotImplementedError

    def is_valid(self) -> bool:
        raise NotImplementedError

Model = TypeVar("Model", bound=BaseModel)

class Sampler(object):
    def __init__(self, data: np.ndarray):
        """data can be a np.ndarray as a list of anything"""
        self.data = data.copy()
        shuffle(self.data)
        self.start_idx = 0
        self.length = self.data.shape[0]

    def take(self, sample_size: int):
        if sample_size > self.length:
            raise ValueError("cannot take more than in the sample set")
        end_idx = self.start_idx + sample_size
        if end_idx > self.length:
            result = self.data[end_idx:].copy()
            shuffle(self.data)
            self.start_idx = end_idx - self.length
            return np.concatenate([result, self.data[0: self.start_idx]])
        else:
            self.start_idx, start_idx = end_idx, self.start_idx
            return self.data[start_idx: end_idx]


def fit(Model: Type[Model], processor: Processor, samples: np.ndarray) -> Model:
    samples = processor.preprocess(samples)
    threshold = processor.threshold()
    iter_id, max_inlier_no = 0, 0
    max_inlier_mask = np.zeros_like(samples, dtype=np.bool_)
    is_adaptive, max_model = False, None
    max_iter = MAX_ITER_INITIAL
    sampler = Sampler(samples)
    while True:
        if is_adaptive:
            sample = samples[max_inlier_mask]
            model = Model.fit(sample, processor)
            is_adaptive = False
        else:
            sample = sampler.take(Model.min_sample_no)
            try:
                model = Model.solve(sample, processor)
            except LinAlgError:
                continue
        inlier_mask = np.abs(model.evaluate(samples)) < threshold
        inlier_no = inlier_mask.sum()
        if inlier_no > max_inlier_no and model.is_valid():  # save the temporary max
            max_inlier_mask = inlier_mask
            max_inlier_no = inlier_no
            max_model = model
            max_iter = np.log(0.01) / np.log(1 - (max_inlier_no / samples.shape[0]) ** 5 + np.spacing(1))
            is_adaptive = True
        iter_id += 1
        if iter_id >= max_iter:
            if max_model is not None:
                return max_model
            else:
                raise ValueError("cannot find even one model that fits")
