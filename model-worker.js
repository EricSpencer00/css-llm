import { env, pipeline, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const MODEL_ID = 'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA';

env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

let generator;
let loadingPromise;
let runtime = {
  device: 'wasm',
  dtype: 'q4',
};
let busy = false;

function send(message) {
  self.postMessage(message);
}

function describeError(error) {
  return error instanceof Error ? error.message : String(error);
}

function chooseRuntime(forceWasm = false) {
  const webgpuAvailable = typeof navigator !== 'undefined' && Boolean(navigator.gpu);
  runtime = forceWasm || !webgpuAvailable
    ? { device: 'wasm', dtype: 'q4' }
    : { device: 'webgpu', dtype: 'q4f16' };
  return runtime;
}

function progressCallback(info) {
  send({
    type: 'progress',
    status: info.status,
    file: info.file || '',
    progress: Number.isFinite(info.progress) ? info.progress : null,
    loaded: info.loaded,
    total: info.total,
  });
}

async function loadModel(forceWasm = false) {
  if (generator) return generator;
  if (loadingPromise) return loadingPromise;

  chooseRuntime(forceWasm);
  send({ type: 'loading', model: MODEL_ID, ...runtime });

  loadingPromise = pipeline('text-generation', MODEL_ID, {
    device: runtime.device,
    dtype: runtime.dtype,
    progress_callback: progressCallback,
  }).then((loadedGenerator) => {
    generator = loadedGenerator;
    send({ type: 'ready', model: MODEL_ID, ...runtime });
    return generator;
  }).catch(async (error) => {
    loadingPromise = undefined;

    if (!forceWasm && runtime.device === 'webgpu') {
      send({
        type: 'fallback',
        from: 'webgpu',
        to: 'wasm',
        message: 'WebGPU could not initialize; retrying with WebAssembly.',
      });
      return loadModel(true);
    }

    throw error;
  });

  return loadingPromise;
}

function extractText(output) {
  const generated = output?.[0]?.generated_text;

  if (typeof generated === 'string') return generated;
  if (!Array.isArray(generated)) return '';

  const lastMessage = generated[generated.length - 1];
  return typeof lastMessage === 'string' ? lastMessage : lastMessage?.content || '';
}

async function generate({ id, prompt }) {
  if (busy) {
    send({ type: 'error', id, phase: 'generate', message: 'The model is already generating.' });
    return;
  }

  busy = true;
  const startedAt = performance.now();
  let streamedText = '';
  let streamUpdates = 0;

  try {
    const model = await loadModel();
    const streamer = new TextStreamer(model.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
        streamedText += text;
        streamUpdates += 1;
        send({ type: 'token', id, text, streamUpdates });
      },
    });

    const output = await model([
      { role: 'system', content: 'You are a helpful, concise assistant. Answer in plain text.' },
      { role: 'user', content: prompt },
    ], {
      max_new_tokens: 96,
      do_sample: true,
      temperature: 0.7,
      top_p: 0.9,
      repetition_penalty: 1.08,
      streamer,
    });

    const extractedText = extractText(output);
    const finalText = streamedText.trim() || extractedText.trim();

    if (!streamedText.trim() && extractedText.trim()) {
      send({ type: 'token', id, text: extractedText, streamUpdates: 1 });
      streamUpdates = 1;
    }

    send({
      type: 'done',
      id,
      text: finalText || 'The model returned no text.',
      streamUpdates,
      latencyMs: Math.round(performance.now() - startedAt),
      ...runtime,
    });
  } catch (error) {
    send({ type: 'error', id, phase: 'generate', message: describeError(error) });
  } finally {
    busy = false;
  }
}

self.addEventListener('message', async ({ data }) => {
  if (data?.type === 'load') {
    try {
      await loadModel();
    } catch (error) {
      send({ type: 'error', phase: 'load', message: describeError(error) });
    }
  }

  if (data?.type === 'generate') {
    await generate(data);
  }
});
