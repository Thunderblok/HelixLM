from datasets import load_dataset
from helix_lm import PretrainTrainer, HelixConfig, HelixForCausalLM, HelixTokenizer

tok = HelixTokenizer('gpt2')
cfg = HelixConfig.small_v2(
    vocab_size=len(tok),
    seq_len=96,
    tokenizer_name='gpt2',
    use_titans_memory=False,
    epochs=1,
    batch_size=8,
    lr=0.001,
    seed=42,
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id
cfg.bos_token_id = tok.bos_token_id
model = HelixForCausalLM(cfg)

ds = load_dataset(
    'david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427',
    streaming=True,
)
train_iter = ds['pretrain_train'].take(50)['text']   # tiny train subset
val_iter = ds['pretrain_val'].take(10)['text']       # tiny val subset

trainer = PretrainTrainer(
    model=model,
    cfg=cfg,
    train_texts=train_iter,
    val_texts=val_iter,
    tokenizer=tok,
    output_dir='./whole_test',
    seed=42,
    num_workers=2,
    verbose=True,
)
trainer.train(num_epochs=1)
