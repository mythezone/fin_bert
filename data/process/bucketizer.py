from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os
import json
import matplotlib.pyplot as plt


class Bucketizer(ABC):
    @abstractmethod
    def encode(self, value: float) -> int:
        pass

    @abstractmethod
    def decode(self, index: int) -> Tuple[float, float]:
        pass

    @abstractmethod
    def num_buckets(self) -> int:
        pass


class ExponentialBucketizer(Bucketizer):
    def __init__(self, boundaries: List[float]):
        self.boundaries = sorted(
            set([-x for x in boundaries[::-1]] + [0.0] + boundaries)
        )

    def encode(self, value: float) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= value < self.boundaries[i + 1]:
                return i
        return len(self.boundaries) - 2  # for the upper boundary

    def decode(self, index: int) -> Tuple[float, float]:
        return self.boundaries[index], self.boundaries[index + 1]

    def num_buckets(self) -> int:
        return len(self.boundaries) - 1


class UniformBucketizer(Bucketizer):
    def __init__(self, lower: float = -10.0, upper: float = 10.0, num_bins: int = 16):
        self.lower = lower
        self.upper = upper
        self.num_bins = num_bins
        self.bin_width = (upper - lower) / num_bins

    def encode(self, value: float) -> int:
        clipped = np.clip(value, self.lower, self.upper - 1e-8)
        return int((clipped - self.lower) / self.bin_width)

    def decode(self, index: int) -> Tuple[float, float]:
        start = self.lower + index * self.bin_width
        return start, start + self.bin_width

    def num_buckets(self) -> int:
        return self.num_bins
