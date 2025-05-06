import json
import os
from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForMaskedLM,
    RobertaConfig,
    RobertaForMaskedLM,
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.cfg["tokenizer_path"]
        )
        self.tokenizer.add_special_tokens(
            {
                "pad_token": "[PAD]",
                "mask_token": "[MASK]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
            }
        )

        self.dataset = load_dataset(self.cfg["dataset_name"], split="train")
        self.dataset = self._split_dataset()

        self.tokenized = self.dataset.map(
            self._tokenize_fn, batched=True, remove_columns=["id", "text"]
        )

        self.model = self._init_model()
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _split_dataset(self):
        split = self.dataset.train_test_split(test_size=0.1, seed=42)
        valid_test = split["test"].train_test_split(test_size=0.5, seed=42)
        return {
            "train": split["train"],
            "eval": valid_test["train"],
            "test": valid_test["test"],
        }

    def _tokenize_fn(self, examples):
        return self.tokenizer(
            [" ".join(x) for x in examples["text"]],
            truncation=True,
            padding="max_length",
            max_length=self.cfg["max_seq_length"],
        )

    def _init_model(self):
        model_type = self.cfg["model_type"]
        common_args = {
            "vocab_size": self.cfg["vocab_size"],
            "max_position_embeddings": self.cfg["max_position_embeddings"],
            "hidden_size": self.cfg["hidden_size"],
            "num_hidden_layers": self.cfg["num_hidden_layers"],
            "num_attention_heads": self.cfg["num_attention_heads"],
        }
        if model_type == "bert":
            config = BertConfig(type_vocab_size=2, **common_args)
            return BertForMaskedLM(config)
        elif model_type == "roberta":
            config = RobertaConfig(type_vocab_size=1, **common_args)
            return RobertaForMaskedLM(config)
        elif model_type == "deberta":
            config = DebertaV2Config(type_vocab_size=2, **common_args)
            return DebertaV2ForMaskedLM(config)
        elif model_type == "distilbert":
            config = DistilBertConfig(**common_args)
            return DistilBertForMaskedLM(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self):
        args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.cfg.get("batch_size", 32),
            num_train_epochs=self.cfg.get("num_train_epochs", 5),
            save_strategy="epoch",
            logging_dir=os.path.join(self.cfg["output_dir"], "logs"),
            logging_steps=100,
            report_to=["mlflow", "comet"],
            run_name=self.cfg.get("run_name", "finbert-run"),
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["eval"],
            data_collator=collator,
        )
        trainer.train()
        trainer.save_model(self.cfg["output_dir"])

    def evaluate(self):

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            per_device_eval_batch_size=self.cfg.get("batch_size", 32),
            do_train=False,
            do_eval=True,
            report_to=[],
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=self.tokenized["test"],
            data_collator=collator,
        )

        metrics = trainer.evaluate()
        print("\n✅ Test Set Evaluation:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["train", "evaluate"], default="train"
    )
    args = parser.parse_args()

    trainer = ModelTrainer(args.config_path)
    if args.mode == "train":
        trainer.train()
    else:
        trainer.evaluate()
