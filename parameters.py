MODEL_block_size = 128
MODEL_batch_size = 8
MODEL_max_iters = 10000
MODEL_eval_interval = 1000
MODEL_eval_iters = 100
MODEL_learning_rate = 3e-4
MODEL_weight_decay = 0.1
MODEL_grad_clip = 1.0
MODEL_n_embeddings = 128
MODEL_n_head = 4
MODEL_n_decoder_layers = 2
MODEL_dropout = 0.1

DATA_bin_dir = "data_bin"
DATA_prefix = "tinystories_bpe"
TOKENIZER_vocab_json = "tokenizer_out/vocab.json"
TOKENIZER_merges_txt = "tokenizer_out/merges.txt"
MODEL_ckpt_name = "tinystories_bpe_ckpt.pt"
MODEL_test_prompt = "Thou"
MODEL_sample_tokens_before = 64
MODEL_sample_tokens_after = 96


