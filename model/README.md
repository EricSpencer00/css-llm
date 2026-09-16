# CSS-RNN-64 weights

`weights.json` is the quantized parameter set compiled into
[`../css-model.css`](../css-model.css).

- architecture: 64-unit character-level recurrent encoder and fixed rollout
- vocabulary: 64 lowercase English, numeric, whitespace, and code symbols
- prompt context: 64 characters
- rollout: 96 greedy next-character steps
- quantization: signed 4-bit input matrix; signed 8-bit recurrent, output, and bias values
- activation: hard-tanh
- decoder: CSS greedy argmax with whitespace and repetition constraints

The JSON records the corpus hash, training seed, step count, validation loss,
and quantization scheme. The stylesheet is the browser runtime form of those
weights: it contains the unrolled recurrent graph, numeric operands, argmax
masks, and final output registers. JavaScript is only the I/O bridge.
