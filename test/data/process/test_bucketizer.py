import pytest
from data.process.bucketizer import UniformBucketizer, ExponentialBucketizer


@pytest.fixture
def values():
    return [-0.1, -0.05, 0.0, 0.05, 0.1]


def test_uniform_bucketizer_encode_decode(values):
    k = 4
    bucketizer = UniformBucketizer(k=k)
    encoded = [bucketizer.encode(v) for v in values]
    print("UniformBucketizer Encoded:", encoded)
    decoded_mean = [bucketizer.decode(i, mode="mean") for i in encoded]
    print("UniformBucketizer Decoded (mean):", decoded_mean)
    decoded_random = [bucketizer.decode(i, mode="random") for i in encoded]
    print("UniformBucketizer Decoded (random):", decoded_random)
    assert all(isinstance(i, int) and 0 <= i <= 2 * k for i in encoded)
    assert all(isinstance(d, float) for d in decoded_mean)
    assert all(isinstance(d, float) for d in decoded_random)


def test_exponential_bucketizer_encode_decode(values):
    k = 4
    bucketizer = ExponentialBucketizer(k=k)
    encoded = [bucketizer.encode(v) for v in values]
    print("ExponentialBucketizer Encoded:", encoded)
    decoded_mean = [bucketizer.decode(i, mode="mean") for i in encoded]
    print("ExponentialBucketizer Decoded (mean):", decoded_mean)
    decoded_random = [bucketizer.decode(i, mode="random") for i in encoded]
    print("ExponentialBucketizer Decoded (random):", decoded_random)
    assert all(isinstance(i, int) and 0 <= i <= 2 * k for i in encoded)
    assert all(isinstance(d, float) for d in decoded_mean)
    assert all(isinstance(d, float) for d in decoded_random)
