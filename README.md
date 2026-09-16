# CSS LLM

This is a browser experiment that evaluates a quantized recurrent language
model through CSS custom properties. The checked-in runtime has no server,
WASM module, ONNX runtime, model download, or external dependency.

## Runtime contract

`css-model.css` contains the model graph. It defines typed numeric properties,
quantized operands, recurrent state, greedy character selection, and a fixed
96-step rollout. The browser style engine resolves the graph.

`script.js` is the I/O bridge. It writes a one-hot prompt into CSS custom
properties, reads the computed character ids from `getComputedStyle()`, and
renders the result. It does not perform inference.

The model uses 64 hidden units, a 64-character prompt context, a fixed
96-character output window, signed 4-bit input weights, and signed 8-bit
recurrent, output, and bias values. The runtime vocabulary is fixed at build
time and includes lowercase letters, numbers, whitespace, punctuation, and
common code symbols.

## Local execution

```sh
python3 -m http.server 4175
```

Open <http://127.0.0.1:4175/> after starting the server. A local HTTP server
is required because the stylesheet is inspected through the browser's CSSOM.

## Rebuild

The model compiler accepts a training corpus, optional supervised examples,
and an optional Hugging Face teacher checkpoint:

```sh
python3 tools/train_css_rnn.py \
  --corpus model/training_corpus.txt \
  --examples model/supervised.jsonl \
  --weights model/weights.json \
  --css css-model.css \
  --hf-repo roneneldan/TinyStories-Instruct-1M
```

The teacher is used during training only. The browser receives the compiled
CSS artifact and the quantized weights are stored in `model/weights.json`.

## Verification

```sh
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

The `main` branch deploys through GitHub Pages at
<https://ericspencer.us/css-llm/>.
