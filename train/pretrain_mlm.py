import argparse
import comet_ml
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict, load_from_disk
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["COMET_API_KEY"] = "yJJTSjgwXMbbO1FgvP5gxNLWr"

# print(TrainingArguments.__module__)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="dataset/tokenizer/wp_tokenizer.json"
)

# 显式注册特殊 token
tokenizer.pad_token = "[PAD]"
tokenizer.cls_token = "[CLS]"  # 如果有的话
tokenizer.sep_token = "[SEP]"  # 如果有的话
tokenizer.mask_token = "[MASK]"  # 如果用于MLM，建议定义
tokenizer.unk_token = "[UNK]"


def tokenize_function(examples, tokenizer, max_seq_length):
    joined_texts = [" ".join(tokens) for tokens in examples["text"]]
    return tokenizer(
        joined_texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path", type=str, default="data/tokenized/financial_tokenizer.json"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mythezone/financial-corpus-a-share"
    )
    parser.add_argument("--output_dir", type=str, default="output/finbert-mlm")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    # 添加必要的 special tokens
    special_tokens = {
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
    }
    tokenizer.add_special_tokens(special_tokens)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["id", "text"],
    )

    from datasets import DatasetDict

    # 拆分验证集和测试集
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    valid_test_split = split_dataset["test"].train_test_split(test_size=0.5, seed=42)
    tokenized_dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "valid": valid_test_split["train"],
            "test": valid_test_split["test"],
        }
    )

    # Build model config
    config = BertConfig(
        vocab_size=args.vocab_size,
        max_position_embeddings=args.max_seq_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        type_vocab_size=2,
    )
    model = BertForMaskedLM(config)

    # 更新模型词表大小以匹配 tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # MLM collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Training settings
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        per_device_train_batch_size=32,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        report_to=["mlflow", "comet_ml"],
        run_name="finbert-mlm-run",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
