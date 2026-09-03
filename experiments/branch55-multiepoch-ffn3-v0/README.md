# Branch55 multi-epoch FFN 3.0 campaign

This campaign repeats the complete admitted 1.504B-token U16 corpus for three
epochs with one continuous optimizer. The iterator closes accumulation at each
corpus boundary, so an epoch checkpoint never contains samples from the next
epoch.

Each completed epoch produces `checkpoints/epoch-NN.pt`, reloads it immediately,
and records its SHA-256 and readback result in the append-only MLflow spool.
Remote MLflow is a live projection; the local spool and checkpoints remain the
custody record. Hugging Face publication is an optional post-checkpoint action
through `../branch50-linear-context-v0/publish_hf_checkpoint.py`; it is not part
of the training process and is never enabled implicitly.

The model-name legend is: `s` sequence length, `l` Helix loops, `f` FFN
expansion, `r` learning rate, `e` completed epoch, and `g` source commit prefix.
Generated Hub names are limited to 96 characters.
