from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from transformers import PreTrainedTokenizerFast


def train_hex_tokenizer(
    txt_path: str,
    json_path: str,
    vocab_size: int = 300000,
    save_dir: str = "financial_tokenizer",
):
    # 1. 初始化 WordLevel 模型
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    # 2. 设置分词器：禁用 normalization，仅使用空格切分
    tokenizer.normalizer = normalizers.Sequence([])  # 禁用默认处理（保留大写）
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3. 设置训练器
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )

    # 4. 训练
    tokenizer.train([txt_path], trainer)
    tokenizer.save(json_path)
    print(f"[✔] Tokenizer saved to {json_path}")

    # 5. 封装为 HuggingFace 使用的格式
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    hf_tok.save_pretrained(save_dir)
    print(f"[✔] HuggingFace tokenizer saved to {save_dir}")
