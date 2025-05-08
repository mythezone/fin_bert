from typing import Callable


def token_tuple_to_str_slash(token: tuple[int, int, int, int]) -> str:
    return "_".join(map(str, token))


def token_tuple_to_str_pre(token: tuple[int, int, int, int]) -> str:
    return f"({token[0]},{token[1]},{token[2]},{token[3]})"


def corpus_to_txt(corpus_df, txt_path, method: Callable):
    with open(txt_path, "w") as f:
        for row in corpus_df["text"]:
            tokens = [method(t) for t in row]
            f.write(" ".join(tokens) + "\n")
