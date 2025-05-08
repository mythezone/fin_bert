from data.process import discretizer
from data.process import corpus_builder
from data.process.bucketizer import Bucketizer, ExponentialBucketizer, UniformBucketizer
from data.process.percentage import Percentage, DailyOHLCPercentage
from data.process.discretizer import Discretizer
from data.process.corpus_builder import CorpusBuilder


bucketizers = {
    "uniform": UniformBucketizer,
    "exp": ExponentialBucketizer,
    "std": ExponentialBucketizer,
}

percentagers = {
    "std": DailyOHLCPercentage,
}

discretizers = {
    "std": Discretizer,
}

corpus_builders = {
    "std": CorpusBuilder,
}


__all__ = [
    "bucketizers",
    "percentagers",
    "discretizers",
    "corpus_builders",
    "Bucketizer",
    "ExponentialBucketizer",
    "UniformBucketizer",
    "Percentage",
    "Discretizer",
    "CorpusBuilder",
]
