import os
import comet_ml
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset, Dataset, DatasetDict
import torch
from datasets import load_from_disk
from transformers import EarlyStoppingCallback
from pre_train.trainer import FinancialMAETrainer

# 环境配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 第一步：加载原始文本语料（假设每行22个 token，空格分隔）
dataset = load_dataset("text", data_files={"train": "dataset/standard_hex.txt"})[
    "train"
]


# 第二步：训练 Tokenizer（WordPiece）
def train_tokenizer():
    tokenizer_model = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer_model.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=31000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    def batch_iterator():
        for sample in dataset:
            yield sample["text"]

    tokenizer_model.train_from_iterator(batch_iterator(), trainer)
    tokenizer_model.decoder = decoders.WordPiece(prefix="##")
    tokenizer_model.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer_model.token_to_id("[CLS]")),
            ("[SEP]", tokenizer_model.token_to_id("[SEP]")),
        ],
    )
    # tokenizer_model.save_pretrained("dataset/tokenizer/wp_tokenizer_31000")  # ❌ 不建议再使用此方式
    # tokenizer_model.save("dataset/tokenizer/wp_tokenizer_31000.json")  # ❌ 不建议再使用此方式
    return tokenizer_model


tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "dataset/tokenizer/custom_tokenizer"
)


def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=32,
    )


tokenized_dataset = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
tokenized_dataset.save_to_disk("dataset/huggingface/custom_tokenized_max_32")

# tokenized_dataset = load_from_disk("dataset/huggingface/custom_tokenized_32")

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

# 第五步：构建 BERT 配置和模型
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=384,
    num_hidden_layers=6,
    num_attention_heads=4,
    max_position_embeddings=32,
    type_vocab_size=1,
)
model = BertForMaskedLM(config)
model.resize_token_embeddings(len(tokenizer))

# 第六步：数据处理器与训练参数
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    per_device_train_batch_size=32,
    num_train_epochs=10,
    save_strategy="epoch",
    logging_dir="./output/logs",
    logging_steps=10,
    report_to=["comet_ml"],
    run_name="[LC1]bert-mlm-pretrain(vocab=31000,custom_tokenizer):v0.2",
    no_cuda=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # fp16=True,
)

# 第七步：训练

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

# 保存训练好的模型和 tokenizer
# 建议推理或微调时使用此路径加载
trainer.save_model("output/model_bert_mlm_custom")
tokenizer.save_pretrained("output/model_bert_mlm_custom")
