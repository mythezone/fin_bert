from transformers import Trainer
import torch
import torch.nn.functional as F


class FinancialMAETrainer(Trainer):

    def __init__(self, *args, processing_class=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"]
        attention_mask = inputs.get("attention_mask", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        # 找出 masked 的位置
        mask = labels != -100
        pred_ids = torch.argmax(logits, dim=-1)
        masked_pred_ids = pred_ids[mask]
        masked_true_ids = labels[mask]

        # 解码成 token，再转成 4个值（开高低收）
        decoded_preds = [self.processing_class.decode([pid]) for pid in masked_pred_ids]
        decoded_trues = [self.processing_class.decode([tid]) for tid in masked_true_ids]

        def hex_to_int(h: str) -> int:
            u = int(h, 16)
            return u - 256 if u >= 128 else u

        def token_to_ohlc(hex_str: str) -> tuple[int, ...]:
            return tuple(
                hex_to_int(hex_str[i : i + 2]) for i in range(0, len(hex_str), 2)
            )

        preds_ohlc = torch.tensor(
            [token_to_ohlc(s) for s in decoded_preds],
            dtype=torch.float,
            device=logits.device,
        )
        trues_ohlc = torch.tensor(
            [token_to_ohlc(s) for s in decoded_trues],
            dtype=torch.float,
            device=logits.device,
        )

        loss = F.l1_loss(preds_ohlc, trues_ohlc)

        return (loss, outputs) if return_outputs else loss
