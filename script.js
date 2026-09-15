const body = document.body;
const promptInput = document.querySelector('#promptInput');
const predictedText = document.querySelector('#predictedText');
const latencyBadge = document.querySelector('#latencyBadge');
const stageLatency = document.querySelector('#stageLatency');
const stageStatus = document.querySelector('#stageStatus');
const stageMeter = document.querySelector('#stageMeter');
const signalValue = document.querySelector('#signalValue');
const systemStatus = document.querySelector('#systemStatus');
const runtimeValue = document.querySelector('#runtimeValue');
const recursionValue = document.querySelector('#recursionValue');
const stackDepth = document.querySelector('#stackDepth');
const clusterCount = document.querySelector('#clusterCount');
const clusterLoad = document.querySelector('#clusterLoad');
const tokenCount = document.querySelector('#tokenCount');
const outputLabel = document.querySelector('#outputLabel');
const anticipateButton = document.querySelector('#anticipateButton');

const nodes = [...document.querySelectorAll('.toothbrush-node')];
const examplePrompt = 'Explain in one sentence why CSS is Turing complete.';
const modelWorker = new Worker('./model-worker.js', { type: 'module' });
let hasInteracted = false;
let modelReady = false;
let modelLoading = false;
let activeRequest = null;

function cleanPrompt(value) {
  return value.replace(/[“”]/g, '').trim();
}

function setRuntime(device, dtype, status = 'ready') {
  const readableDevice = device === 'webgpu' ? 'WebGPU' : 'WebAssembly';
  runtimeValue.textContent = `${readableDevice} / ${dtype}`;
  stackDepth.textContent = readableDevice;
  clusterLoad.textContent = status;
  systemStatus.textContent = `local / ${readableDevice}`;
}

function setLoadingState(message = 'loading the model into your browser.') {
  modelLoading = true;
  modelReady = false;
  anticipateButton.disabled = true;
  outputLabel.textContent = 'local model loading';
  predictedText.textContent = message;
  stageStatus.textContent = message;
  latencyBadge.textContent = 'loading';
  stageLatency.textContent = 'loading';
  systemStatus.textContent = 'downloading weights';
}

function setReadyState(device, dtype) {
  modelLoading = false;
  modelReady = true;
  anticipateButton.disabled = false;
  setRuntime(device, dtype, 'ready');
  outputLabel.textContent = 'local model output';
  predictedText.textContent = 'Model ready. Ask it something.';
  stageStatus.textContent = 'weights ready. run inference in this browser.';
  latencyBadge.textContent = 'ready';
  stageLatency.textContent = 'not run';
  signalValue.textContent = '0';
  tokenCount.textContent = '0';
  stageMeter.style.width = '100%';
  recursionValue.textContent = 'local';
}

function showRuntimeError(message) {
  modelLoading = false;
  modelReady = false;
  anticipateButton.disabled = true;
  outputLabel.textContent = 'runtime error';
  predictedText.textContent = `The local model could not start: ${message}`;
  stageStatus.textContent = 'model startup failed. check the browser console for details.';
  latencyBadge.textContent = 'error';
  stageLatency.textContent = 'not run';
  systemStatus.textContent = 'local / error';
  runtimeValue.textContent = 'unavailable';
  stackDepth.textContent = 'error';
  clusterLoad.textContent = 'offline';
  stageMeter.style.width = '0%';
}

function runPrediction() {
  hasInteracted = true;

  if (!modelReady || activeRequest) return;

  const prompt = cleanPrompt(promptInput.value) || examplePrompt;
  promptInput.value = promptInput.value || prompt;
  const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;

  activeRequest = { id, text: '', streamUpdates: 0 };
  predictedText.textContent = '';
  outputLabel.textContent = 'generating locally';
  latencyBadge.textContent = 'running';
  stageLatency.textContent = 'running';
  stageStatus.textContent = 'generating tokens inside this browser.';
  systemStatus.textContent = `local / ${runtimeValue.textContent.split(' / ')[0]}`;
  signalValue.textContent = 'running';
  tokenCount.textContent = '0';
  stageMeter.style.width = '35%';
  body.classList.add('is-predicting');
  modelWorker.postMessage({ type: 'generate', id, prompt });
}

function resetModel() {
  hasInteracted = true;
  activeRequest = null;
  promptInput.value = '';
  predictedText.textContent = modelReady ? 'Model ready. Ask it something.' : 'Loading the model. First run may take a minute.';
  outputLabel.textContent = modelReady ? 'local model output' : 'local model loading';
  latencyBadge.textContent = modelReady ? 'ready' : 'loading';
  stageLatency.textContent = 'not run';
  stageStatus.textContent = modelReady ? 'weights ready. run inference in this browser.' : 'loading the model into your browser.';
  signalValue.textContent = '0';
  tokenCount.textContent = '0';
  stageMeter.style.width = modelReady ? '100%' : '8%';
  body.classList.remove('is-predicting');
}

function handleProgress(data) {
  const fileName = data.file?.split('/').pop() || 'model files';

  if (data.status === 'progress' && Number.isFinite(data.progress)) {
    const progress = Math.max(4, Math.min(100, Math.round(data.progress)));
    stageMeter.style.width = `${progress}%`;
    stageStatus.textContent = `downloading ${fileName} · ${progress}%`;
    predictedText.textContent = `Downloading ${fileName}. First run is the slow one.`;
  } else if (data.status === 'initiate') {
    stageStatus.textContent = `requesting ${fileName} from the model hub.`;
  } else if (data.status === 'done') {
    stageStatus.textContent = `cached ${fileName}. preparing the runtime.`;
  }
}

modelWorker.addEventListener('message', ({ data }) => {
  if (data.type === 'loading') {
    setLoadingState(`Loading ${data.model}. First run may take a minute.`);
    setRuntime(data.device, data.dtype, 'loading');
    return;
  }

  if (data.type === 'progress') {
    handleProgress(data);
    return;
  }

  if (data.type === 'fallback') {
    setLoadingState(data.message);
    return;
  }

  if (data.type === 'ready') {
    setReadyState(data.device, data.dtype);
    return;
  }

  if (data.type === 'token' && activeRequest?.id === data.id) {
    activeRequest.text += data.text;
    activeRequest.streamUpdates = data.streamUpdates;
    predictedText.textContent = activeRequest.text;
    tokenCount.textContent = String(data.streamUpdates);
    signalValue.textContent = String(data.streamUpdates);
    stageMeter.style.width = `${Math.min(96, 35 + data.streamUpdates * 3)}%`;
    return;
  }

  if (data.type === 'done' && activeRequest?.id === data.id) {
    predictedText.textContent = data.text;
    tokenCount.textContent = String(data.streamUpdates);
    signalValue.textContent = String(data.streamUpdates);
    latencyBadge.textContent = `${data.latencyMs}ms`;
    stageLatency.textContent = `${data.latencyMs}ms`;
    stageStatus.textContent = `complete · ${data.streamUpdates} streamed updates · executed locally.`;
    stageMeter.style.width = '100%';
    body.classList.remove('is-predicting');
    activeRequest = null;
    return;
  }

  if (data.type === 'error') {
    if (data.id && activeRequest?.id !== data.id) return;
    activeRequest = null;
    body.classList.remove('is-predicting');
    showRuntimeError(data.message);
  }
});

anticipateButton.addEventListener('click', runPrediction);
document.querySelector('#resetButton').addEventListener('click', resetModel);

promptInput.addEventListener('input', (event) => {
  hasInteracted = true;
  const value = cleanPrompt(event.target.value);
  if (activeRequest) return;

  if (value) {
    predictedText.textContent = 'Prompt staged. Run the model when ready.';
    stageStatus.textContent = modelReady ? 'prompt ready. run local inference.' : 'prompt saved. model is still loading.';
    signalValue.textContent = String(value.length);
  } else {
    predictedText.textContent = modelReady ? 'Model ready. Ask it something.' : 'Loading the model. First run may take a minute.';
    stageStatus.textContent = modelReady ? 'weights ready. run inference in this browser.' : 'loading the model into your browser.';
    signalValue.textContent = '0';
  }
});

document.querySelector('#rerouteButton').addEventListener('click', () => {
  hasInteracted = true;
  pulseCluster();
  stageStatus.textContent = 'diagnostic pulse sent to the decorative toothbrush bus.';
  systemStatus.textContent = modelReady ? `local / ${runtimeValue.textContent.split(' / ')[0]}` : 'downloading weights';
});

document.querySelector('#posterButton').addEventListener('click', (event) => {
  const posterMode = body.classList.toggle('poster-mode');
  event.currentTarget.textContent = posterMode ? 'return to instrument' : 'make a poster';
});

function heatNode(node) {
  node.classList.add('is-hot');
  window.setTimeout(() => node.classList.remove('is-hot'), 560);
}

function pulseCluster() {
  const activeNodes = nodes.filter((node) => !node.classList.contains('node-h') && !node.classList.contains('node-i'));
  const hotNodes = activeNodes.sort(() => Math.random() - 0.5).slice(0, Math.floor(Math.random() * 3) + 3);
  hotNodes.forEach((node, index) => window.setTimeout(() => heatNode(node), index * 90));
  clusterLoad.textContent = modelReady ? 'ready' : 'loading';
  clusterCount.textContent = `${String(hotNodes.length).padStart(2, '0')} / 09`;
}

nodes.forEach((node) => node.addEventListener('click', () => {
  hasInteracted = true;
  heatNode(node);
  stageStatus.textContent = `${node.querySelector('b').textContent} is diagnostic only; inference runs in the model worker.`;
}));

document.querySelectorAll('.claim-row').forEach((row) => {
  row.addEventListener('toggle', () => {
    if (row.open && hasInteracted) stageStatus.textContent = 'fact expanded.';
  });
});

window.setInterval(pulseCluster, 2200);
window.setInterval(() => {
  document.documentElement.style.setProperty('--bar-height', `${Math.round(10 + Math.random() * 38)}px`);
}, 1000);

setLoadingState();
modelWorker.postMessage({ type: 'load' });
