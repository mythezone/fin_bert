from transformers import PreTrainedTokenizer
import re


class TupleTokenizer(PreTrainedTokenizer):
    """
    A simple tokenizer for corpora where each token is a 4-dimensional tuple represented as "(x,y,z,w)".
    """

    def __init__(self, vocab=None, unk_token="[UNK]", pad_token="[PAD]", **kwargs):
        super().__init__(unk_token=unk_token, pad_token=pad_token, **kwargs)
        # vocabulary: mapping from tuple-string to id
        self.vocab = vocab or {}
        # reverse mapping
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}

    @classmethod
    def from_corpus(cls, corpus_texts, min_freq=1, special_tokens=None):
        """
        Build a tokenizer from a list of corpus texts (strings of tuples separated by spaces).
        """
        # Count frequencies
        freq = {}
        pattern = re.compile(r"\(-?\d+,-?\d+,-?\d+,-?\d+\)")
        for text in corpus_texts:
            tokens = pattern.findall(text)
            for tok in tokens:
                freq[tok] = freq.get(tok, 0) + 1
        # filter by min frequency
        vocab_tokens = [tok for tok, f in freq.items() if f >= min_freq]
        # reserve ids: 0=pad,1=unk
        token2id = (
            {special_tokens[0]: 0, special_tokens[1]: 1}
            if special_tokens
            else {"[PAD]": 0, "[UNK]": 1}
        )
        idx = len(token2id)
        for tok in sorted(vocab_tokens):
            if tok not in token2id:
                token2id[tok] = idx
                idx += 1
        tokenizer = cls(
            vocab=token2id, unk_token=special_tokens[1], pad_token=special_tokens[0]
        )
        return tokenizer

    def _tokenize(self, text):
        # Extract tuple tokens
        pattern = re.compile(r"\(-?\d+,-?\d+,-?\d+,-?\d+\)")
        return pattern.findall(text)

    def _convert_token_to_id(self, token):
        # Return id or unk
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        # Join tokens with space
        return " ".join(tokens)

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary to disk (as a JSON file).
        """
        import json, os

        path = os.path.join(save_directory, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)


# # Example usage:
# if __name__ == "__main__":
#     corpus = [
#         "(-5,9,-11,-7) (-6,2,-11,-8) (1,3,-12,-12)",
#         "(-2,16,-7,4) (-6,-1,-13,-8) (-4,5,-5,-3)",
#     ]
#     tok = TupleTokenizer.from_corpus(
#         corpus, min_freq=1, special_tokens=["[PAD]", "[UNK]"]
#     )
#     encoded = tok.encode("(-4,5,-5,-3) (-5,9,-11,-7) (0,0,0,0)")
#     print(encoded)
#     print(tok.decode(encoded))
