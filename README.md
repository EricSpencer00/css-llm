# CSS Foundation Toy

An intentionally plain-looking page that runs a real small language model in the browser. It uses the ONNX export of `onnx-community/SmolLM2-135M-Instruct-ONNX-MHA` with Transformers.js. WebGPU is used when available; WebAssembly is the fallback. There is no inference backend or build step.

Live site: <https://ericspencer.us/css-llm/>

## Local preview

```sh
python3 -m http.server 4175
```

Then open <http://127.0.0.1:4175/>.

The first visit downloads the model and tokenizer files from the Hugging Face Hub. Transformers.js caches them in the browser, so later visits can run without downloading them again. The prompt and generation stay in the browser tab.

## Deployment

The `main` branch deploys through GitHub Pages using `.github/workflows/pages.yml`. The `css-llm` repo name supplies the `/css-llm/` path on the `ericspencer.us` Pages domain.
