# CSS-RNN-32 weights

`weights.json` is the learned parameter set for the model compiled into
[`../css-model.css`](../css-model.css).

- architecture: character-level recurrent language model
- hidden units: 32
- vocabulary: 32 lowercase letters plus punctuation and space
- prompt seed: 24 characters
- rollout: 64 greedy next-character steps
- activation: hard-tanh
- training corpus: first 1 MB of the TinyStories validation text
- story boundary markers: normalized to spaces before training
- corpus source: <https://huggingface.co/datasets/roneneldan/TinyStories>

The JSON records the corpus hash, training seed, and training step count. The
stylesheet is the browser runtime form of these weights: it contains the
unrolled recurrent graph and its numeric operands.
