from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def train_tokenizer_from_txt(txt_path, save_path, vocab_size=8000):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train([txt_path], trainer)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")


if __name__ == "__main__":
    import argparse
    import pandas as pd
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_pkl",
        type=str,
        default="data/tokenized/segmented_corpus_16.pkl",
        help="Path to the segmented corpus .pkl file.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/tokenized/financial_tokenizer.json",
        help="Path to save the tokenizer.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Vocabulary size for the tokenizer.",
    )

    args = parser.parse_args()

    # prepare txt corpus
    intermediate_txt = "data/tokenized/segmented_corpus_16.txt"
    os.makedirs(os.path.dirname(intermediate_txt), exist_ok=True)
    df = pd.read_pickle(args.corpus_pkl)
    with open(intermediate_txt, "w") as f:
        for row in df["text"]:
            f.write(" ".join(row) + "\n")

    train_tokenizer_from_txt(intermediate_txt, args.save_path, args.vocab_size)
