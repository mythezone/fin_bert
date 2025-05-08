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


def encode_token_to_hex(t: tuple[int, int, int, int]) -> str:
    def int_to_hex(v: int) -> str:
        # Convert signed int [-128, 127] to unsigned [0, 255]
        u = (v + 256) % 256
        return f"{u:02X}"  # 两位大写十六进制

    return "".join(int_to_hex(x) for x in t)


def hex_to_int(h: str) -> int:
    u = int(h, 16)
    return u - 256 if u >= 128 else u


def decode_hex_to_token(hex_str: str) -> tuple[int, int, int, int]:
    assert len(hex_str) == 8, "Each token must be 8 hex characters"
    return tuple(hex_to_int(hex_str[i : i + 2]) for i in range(0, 8, 2))
