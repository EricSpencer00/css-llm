# CSS Foundation Toy

A static interactive joke about an impossible foundation model. It runs in the browser with HTML, CSS, and JavaScript only. There is no backend, build step, or runtime service.

Live site: <https://ericspencer.us/css-llm/>

## Local preview

```sh
python3 -m http.server 4175
```

Then open <http://127.0.0.1:4175/>.

## Deployment

The `main` branch deploys through GitHub Pages using `.github/workflows/pages.yml`. The `css-llm` repo name supplies the `/css-llm/` path on the `ericspencer.us` Pages domain.
