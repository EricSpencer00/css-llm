# CSS LLM

This is a browser experiment that evaluates a quantized recurrent language
model through CSS custom properties. The shipped runtime has no server, WASM
module, ONNX runtime, model download, or external dependency.

## Runtime contract

`css-model.css` contains the model graph. It defines typed numeric properties,
quantized operands, recurrent state, greedy character selection, and a fixed
96-step rollout. The browser style engine resolves the graph. Conversational
fine-tuning adds a learned projection of the final prompt state to each
generated step, so the answer is not forced to survive only through the
recurrent attractor.

`script.js` is the I/O bridge. It writes a one-hot prompt into CSS custom
properties, reads the computed character ids from `getComputedStyle()`, and
renders the result. It does not perform inference.

The model uses 64 hidden units, a 32-character prompt context, a fixed
96-character output window, signed 4-bit input weights, and signed 8-bit
recurrent, prompt-memory, output, and bias values. The runtime vocabulary is
fixed at build time and includes lowercase letters, numbers, whitespace,
punctuation, and common code symbols. The decoder guards initial whitespace,
penalizes recent repetition, and stops at the first generated line break.

The reproducible build path is in `tools/`: `build_corpus.py` creates the
mixed corpus and supervised examples, while `train_css_rnn.py` performs a
held-out split, quantization-aware training, stability projection, and CSS
lowering. `model/weights.json` records the quantized tensors and evaluation
metadata used for the shipped stylesheet.

## Local execution

```sh
python3 -m http.server 4175
```

Open <http://127.0.0.1:4175/> after starting the server. A local HTTP server
is required because the stylesheet is inspected through the browser's CSSOM.

## Deployment

The `main` branch deploys the compiled runtime through GitHub Pages at
<https://ericspencer.us/css-llm/>.
