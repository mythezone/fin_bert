from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import os

# 读取所有 token
token_set = set()
with open("dataset/standard_hex.txt", "r") as f:
    for line in f:
        tokens = line.strip().split()
        token_set.update(tokens)

# 构建 vocab，添加特殊 token
special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
all_tokens = special_tokens + sorted(token_set)
vocab = {tok: i for i, tok in enumerate(all_tokens)}

# 构造 WordLevel tokenizer
tokenizer_model = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
tokenizer_model.pre_tokenizer = Whitespace()

# 包装成 HF 兼容接口
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 保存
save_path = "dataset/tokenizer/custom_tokenizer"
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
print(f"Tokenizer saved to {save_path}")
