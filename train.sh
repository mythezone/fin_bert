export TOKENIZERS_PARALLELISM=false

python my_train/pretrain_mlm.py \
  --tokenizer_path data/tokenized/financial_tokenizer.json \
  --dataset_name mythezone/financial-corpus-a-share \
  --output_dir output/finbert-mlm \
  --vocab_size 8000 \
  --max_seq_length 32 \
  --hidden_size 256 \
  --num_hidden_layers 4 \
  --num_attention_heads 4 \
  --num_train_epochs 5