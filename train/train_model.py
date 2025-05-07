import json
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["COMET_API_KEY"] = "yJJTSjgwXMbbO1FgvP5gxNLWr"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import comet_ml
from datasets import load_dataset, DatasetDict
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
        self.vocab_size = len(self.tokenizer)

        self.dataset = load_dataset(self.cfg["dataset_name"], split="train")
        self.dataset = DatasetDict(self._split_dataset())

        self.tokenized = self.dataset.map(
            self._tokenize_fn, batched=True, remove_columns=["id", "text"]
        )

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

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
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.cfg["max_position_embeddings"],
            "hidden_size": self.cfg["hidden_size"],
            "num_hidden_layers": self.cfg["num_hidden_layers"],
            "num_attention_heads": self.cfg["num_attention_heads"],
            "dataloader_num_workers": 0,
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

    def check_all_token_ids(self):
        print("🔍 Checking all token IDs in training set...")
        for i, example in enumerate(self.tokenized["train"]):
            for j, t in enumerate(example["input_ids"]):
                if not isinstance(t, int) or t < 0 or t >= self.vocab_size:
                    print(f"[ERROR] sample #{i} token #{j} = {t} ❌")
                    print("input_ids:", example["input_ids"])
                    raise ValueError("Invalid token ID detected")
        print("✅ All token IDs are valid.")

    def train(self):
        args = TrainingArguments(
            output_dir=self.cfg["output_dir"],
            overwrite_output_dir=True,
            eval_strategy="epoch",
            per_device_train_batch_size=self.cfg.get("batch_size", 32),
            num_train_epochs=self.cfg.get("num_train_epochs", 5),
            save_strategy="epoch",
            logging_dir=os.path.join(self.cfg["output_dir"], "logs"),
            logging_steps=100,
            report_to=["mlflow", "comet_ml"],
            run_name=self.cfg.get("run_name", "finbert-run"),
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            ddp_find_unused_parameters=False,
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        def model_init():
            model = self._init_model()
            model.resize_token_embeddings(len(self.tokenizer))
            return model

        trainer = Trainer(
            model=model_init(),
            args=args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["eval"],
            data_collator=collator,
        )

        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)

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
    config_folder = "./config"
    config_files = [c for c in os.listdir(config_folder) if c.endswith(".json")]
    for config_file in config_files:
        config_path = os.path.join(config_folder, config_file)
        trainer = ModelTrainer(config_path)
        trainer.train()
