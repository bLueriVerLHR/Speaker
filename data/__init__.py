"""SFT data pipeline (shared by pretrain/finetune/baselines, same protocol):
jsonl -> plain text / chat template -> tokenize/collate -> assistant-only label masking."""
