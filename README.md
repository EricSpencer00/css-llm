# CSS Foundation Toy

An intentionally plain-looking page that runs a trained character-level language model in CSS. `css-model.css` contains the learned weights and a fixed 24-character-seed, 64-character autoregressive graph. The browser style engine evaluates the matrix multiplies, hard-tanh activation, argmax, and recurrence. There is no inference backend, WASM, ONNX runtime, or model download.

Live site: <https://ericspencer.us/css-llm/>

## Local preview

```sh
python3 -m http.server 4175
```

Then open <http://127.0.0.1:4175/>.

The model is a 32-unit character RNN trained on a 1 MB TinyStories text slice. JavaScript is only the I/O bridge: it writes one-hot seed characters to custom properties and reads the generated character ids back from `getComputedStyle()`. It does not perform inference.

Regenerate the model and stylesheet with:

```sh
python3 tools/train_css_rnn.py \
  --corpus /path/to/TinyStories-valid.txt \
  --weights model/weights.json \
  --css css-model.css
```

## Deployment

The `main` branch deploys through GitHub Pages using `.github/workflows/pages.yml`. The `css-llm` repo name supplies the `/css-llm/` path on the `ericspencer.us` Pages domain.
