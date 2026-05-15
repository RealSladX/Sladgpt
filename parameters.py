MODEL_block_size = 128
MODEL_batch_size = 4
MODEL_max_iters = 90000
MODEL_eval_interval = 1000
MODEL_eval_iters = 100
MODEL_learning_rate = 3e-4
MODEL_weight_decay = 0.1
MODEL_grad_clip = 1.0
MODEL_n_embeddings = 384
MODEL_n_head = 6
MODEL_n_decoder_layers = 6
MODEL_dropout = 0.1

DATA_bin_dir = "data_bin"
DATA_prefix = "tinystories_bpe"
TOKENIZER_vocab_json = "tokenizer_out/vocab.json"
TOKENIZER_merges_txt = "tokenizer_out/merges.txt"
MODEL_test_prompts = ["It all started with", "The two options were", "This is the place where", "You walk inside and find", "A new item appeared in the shop"]
