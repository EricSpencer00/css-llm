# CSS Foundation Toy

I wanted to see how far a browser's style engine could be pushed before the
idea stopped being useful and started being funny. This is a tiny conversational
character model whose inference graph lives in `css-model.css`.

The stylesheet contains a 64-unit recurrent encoder, quantized weights, typed
numeric custom properties, and a fixed rollout graph. JavaScript only cleans
the input, writes one-hot prompt characters, and reads numeric output ids from
`getComputedStyle()`. It does not perform inference and there is no backend,
WASM module, ONNX runtime, model download, or external runtime dependency.

The model uses signed 4-bit input weights and signed 8-bit recurrent/output
weights, trained with quantization-aware forward passes. Its supervised corpus
uses `user:` and `assistant:` turns, so the fixed graph is optimized for short
conversational replies and compact code sketches. A closed lowercase
English/code vocabulary, whitespace guard, repetition penalty, and output
validator keep the public boundary deterministic without inserting canned
answers or routing prompts outside the model.

Live site: <https://ericspencer.us/css-llm/>

## Local preview

```sh
python3 -m http.server 4175
```

Then open <http://127.0.0.1:4175/>.

## Rebuild the model

The corpus builder can combine the conversational seed set with a TinyStories
text file and a DailyDialog JSONL or parquet export:

```sh
python3 tools/build_corpus.py \
  --tiny-stories /path/to/TinyStories-valid.txt \
  --daily-dialog /path/to/daily_dialog.jsonl \
  --output model/training_corpus.txt \
  --examples model/supervised.jsonl
```

Then train and compile the CSS artifact:

```sh
python3 tools/train_css_rnn.py \
  --corpus model/training_corpus.txt \
  --examples model/supervised.jsonl \
  --weights model/weights.json \
  --css css-model.css
```

Training uses NumPy. The browser-side model remains plain CSS, and the
regression checks can be run with:

```sh
python3 -m unittest discover -s tools -p 'test_*.py' -v
```

The `main` branch deploys through GitHub Pages using `.github/workflows/pages.yml`.
The repository name supplies the `/css-llm/` path on the `ericspencer.us`
Pages domain.
