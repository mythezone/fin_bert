from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd


class Bucketizer(ABC):
    def __init__(self, k: int = 10):
        self.k = k
        self.num_bins = 2 * k + 1
        self.lower = -0.10  # -10%
        self.upper = 0.10  # 10%
        self.bin_width = (self.upper - self.lower) / self.num_bins

    @abstractmethod
    def encode(self, value: float) -> int:
        pass

    @abstractmethod
    def decode(self, index: int, mode: str = "mean") -> float:
        pass

    def num_buckets(self) -> int:
        return self.num_bins

    def _binary_encode(self, index: int) -> str:
        # Returns binary representation with sign bit (0 for negative or zero, 1 for positive)
        sign_bit = "1" if index > self.k else "0"
        magnitude = abs(index - self.k)
        # Number of bits needed to represent magnitude
        bits = self.k.bit_length() if self.k > 0 else 1
        mag_bits = format(magnitude, f"0{bits}b")
        return sign_bit + mag_bits

    def set_arg(self):
        pass


class UniformBucketizer(Bucketizer):
    def __init__(self, k: int):
        super().__init__(k)

    def _encode_scalar(self, value: float) -> int:
        clipped = np.clip(value, self.lower, self.upper - 1e-12)
        bin_index = int((clipped - self.lower) / self.bin_width)
        return bin_index - self.k

    def encode(self, value):
        # Accepts scalar or np.ndarray
        if np.isscalar(value):
            return self._encode_scalar(value)
        else:
            arr = np.asarray(value)
            clipped = np.clip(arr, self.lower, self.upper - 1e-12)
            bin_indices = ((clipped - self.lower) / self.bin_width).astype(int) - self.k
            return bin_indices

    def _decode_scalar(self, index: int, mode: str = "mean") -> float:
        start = self.lower + (index + self.k) * self.bin_width
        end = start + self.bin_width
        if mode == "mean":
            return (start + end) / 2
        elif mode == "random":
            return np.random.uniform(start, end)
        else:
            raise ValueError("mode must be 'mean' or 'random'")

    def decode(self, index, mode: str = "mean"):
        # Accepts scalar or np.ndarray
        if np.isscalar(index):
            return self._decode_scalar(index, mode)
        else:
            arr = np.asarray(index)
            start = self.lower + (arr + self.k) * self.bin_width
            end = start + self.bin_width
            if mode == "mean":
                return (start + end) / 2
            elif mode == "random":
                return np.random.uniform(start, end)
            else:
                raise ValueError("mode must be 'mean' or 'random'")


class ExponentialBucketizer(Bucketizer):
    def __init__(self, k: int, e: float = 1.2, zero: float = 1e-4):
        super().__init__(k)
        self.e = e
        self.zero = zero
        self.set_boundaries()

    def set_arg(self, k=None, e=None, zero=None):
        if k:
            self.k = k
        if e:
            self.e = e
        if zero:
            self.zero = zero
        self.set_boundaries()

    def set_boundaries(self):
        # Create symmetric exponential boundaries using mapping x -> sign(x)*2^|x|
        # Raw bin edges from -k-1 to k+1 (2k+2 edges for 2k+1 bins)
        raw_edges = np.linspace(0, 10, self.k + 1)
        scaled = self.e**raw_edges
        pos_b = self.zero + (scaled - scaled.min()) * (self.upper - self.zero) / (
            scaled.max() - scaled.min()
        )

        neg_b = -pos_b[::-1]

        self.boundaries = np.concatenate((neg_b, pos_b))

    def _encode_scalar(self, value: float) -> int:
        clipped = np.clip(value, self.lower, self.upper - 1e-12)
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= clipped < self.boundaries[i + 1]:
                return i - self.k
        return len(self.boundaries) - 2 - self.k  # last bucket if value == upper

    def encode(self, value):
        if np.isscalar(value):
            return self._encode_scalar(value)
        else:
            arr = np.asarray(value)
            clipped = np.clip(arr, self.lower, self.upper - 1e-12)
            idx = np.searchsorted(self.boundaries, clipped, side="right") - 1
            idx = np.clip(idx, 0, len(self.boundaries) - 2)
            return idx - self.k

    def _decode_scalar(self, index: int, mode: str = "mean") -> float:
        start = self.boundaries[index + self.k]
        end = self.boundaries[index + self.k + 1]
        if mode == "mean":
            return (start + end) / 2
        elif mode == "random":
            return np.random.uniform(start, end)
        else:
            raise ValueError("mode must be 'mean' or 'random'")

    def decode(self, index, mode: str = "mean"):
        if np.isscalar(index):
            return self._decode_scalar(index, mode)
        else:
            arr = np.asarray(index)
            start = self.boundaries[arr + self.k]
            end = self.boundaries[arr + self.k + 1]
            if mode == "mean":
                return (start + end) / 2
            elif mode == "random":
                return np.random.uniform(start, end)
            else:
                raise ValueError("mode must be 'mean' or 'random'")


class QuantileBucketizer(Bucketizer):

    def __init__(
        self,
        k: int,
        df: pd.DataFrame,
        cols: List[str] = ["open_pct", "high_pct", "low_pct", "close_pct"],
    ):
        if cols:
            count_df = df[cols].dropna()
        else:
            count_df = df

        quantiles = np.percentile(count_df, np.linspace(0, 100, 2 * k + 2))

        self.quantiles = smooth_duplicate_quantiles(quantiles)
        super().__init__(k)
        self.boundaries = quantiles

    def encode(self, value):
        # Accepts scalar or np.ndarray
        arr = np.asarray(value)
        # The bins are defined by self.boundaries, which has length 2k+2, so 2k+1 bins
        idx = np.searchsorted(self.boundaries, arr, side="right") - 1
        # idx in [0, 2k], so output in [-k, k]
        idx = np.clip(idx, 0, len(self.boundaries) - 2)
        result = idx - self.k
        if np.isscalar(value):
            return int(result)
        return result

    def decode(self, index, mode: str = "mean"):
        # Accepts scalar or np.ndarray
        arr = np.asarray(index)
        start = self.boundaries[arr + self.k]
        end = self.boundaries[arr + self.k + 1]
        if mode == "mean":
            result = (start + end) / 2
        elif mode == "random":
            result = np.random.uniform(start, end)
        else:
            raise ValueError("mode must be 'mean' or 'random'")
        if np.isscalar(index):
            return float(result)
        return result


#  Supporting functions
def smooth_duplicate_quantiles(quantiles: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    修复 quantiles 中的重复边界值：
    将任何相邻的重复值用该区间前后值平均分成三段中的中间两段填充。

    参数:
        quantiles: 原始分位数组 (1D numpy array)
        eps: 判断重复值的最小差值阈值

    返回:
        新的 quantiles 数组，已无连续重复值
    """
    q = quantiles.copy()
    i = 1
    while i < len(q) - 1:
        if abs(q[i] - q[i - 1]) < eps and abs(q[i + 1] - q[i]) < eps:
            # 三个相等（或近似相等）值：q[i-1] == q[i] == q[i+1]
            # 用前后一个非重复区间替代中间两个点
            left = q[i - 2] if i >= 2 else q[i - 1] - 1e-6
            right = q[i + 2] if i + 2 < len(q) else q[i + 1] + 1e-6
            step = (right - left) / 3
            q[i - 1] = left + step
            q[i] = left + 2 * step
            i += 2
        elif abs(q[i] - q[i - 1]) < eps:
            # 单个重复值（两个相等）
            left = q[i - 2] if i >= 2 else q[i - 1] - 1e-6
            right = q[i + 1] if i + 1 < len(q) else q[i] + 1e-6
            step = (right - left) / 3
            q[i - 1] = left + step
            q[i] = left + 2 * step
            i += 1
        else:
            i += 1
    return q
