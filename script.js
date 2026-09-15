const body = document.body;
const engine = document.querySelector('[data-css-rnn-engine]');
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
const clusterCount = document.querySelector('#clusterCount');
const clusterLoad = document.querySelector('#clusterLoad');
const tokenCount = document.querySelector('#tokenCount');
const outputLabel = document.querySelector('#outputLabel');
const nodes = [...document.querySelectorAll('.toothbrush-node')];
const anticipateButton = document.querySelector('#anticipateButton');

const VOCABULARY = "abcdefghijklmnopqrstuvwxyz .,!?\'";
const PROMPT_STEPS = 24;
const GENERATED_STEPS = 64;
const DEFAULT_SEED = 'once upon a time';
let hasInteracted = false;
let modelReady = false;

function cleanPrompt(value) {
  return value.replace(/[“”]/g, '').trim();
}

function cssVar(prefix, index, character) {
  return `--css-rnn-${prefix}-${index}-${character}`;
}

function setReadyState() {
  modelReady = true;
  anticipateButton.disabled = false;
  systemStatus.textContent = 'CSS style engine';
  runtimeValue.textContent = 'typed CSS graph';
  clusterLoad.textContent = 'compiled';
  recursionValue.textContent = 'computed';
  predictedText.textContent = 'CSS graph ready. Seed it with a phrase.';
  stageStatus.textContent = 'CSS model ready. type a seed and run it.';
  latencyBadge.textContent = 'CSS';
  stageLatency.textContent = 'not run';
  signalValue.textContent = '0';
  tokenCount.textContent = '0';
  stageMeter.style.width = '100%';
}

function showRuntimeError(message) {
  modelReady = false;
  anticipateButton.disabled = true;
  systemStatus.textContent = 'CSS style error';
  runtimeValue.textContent = 'unavailable';
  clusterLoad.textContent = 'error';
  predictedText.textContent = `The CSS model could not resolve: ${message}`;
  stageStatus.textContent = 'CSS math support is missing or the model stylesheet failed to load.';
  latencyBadge.textContent = 'error';
  stageLatency.textContent = 'not run';
  stageMeter.style.width = '0%';
}

function bindPrompt(value) {
  const normalized = cleanPrompt(value).toLowerCase().slice(0, PROMPT_STEPS).padEnd(PROMPT_STEPS, ' ');
  for (let step = 0; step < PROMPT_STEPS; step += 1) {
    const token = VOCABULARY.indexOf(normalized[step]);
    for (let character = 0; character < VOCABULARY.length; character += 1) {
      engine.style.setProperty(cssVar('prompt', step, character), token === character ? '1' : '0');
    }
  }
}

function readGeneratedText() {
  const computed = getComputedStyle(engine);
  const ids = [];
  for (let step = 0; step < GENERATED_STEPS; step += 1) {
    const value = Number.parseFloat(computed.getPropertyValue(`--css-rnn-output-${step}`));
    if (!Number.isFinite(value)) throw new Error(`missing CSS output at step ${step}`);
    ids.push(Math.max(0, Math.min(VOCABULARY.length - 1, Math.round(value))));
  }
  return ids.map((id) => VOCABULARY[id]).join('');
}

function runPrediction() {
  hasInteracted = true;
  if (!modelReady) return;

  const prompt = cleanPrompt(promptInput.value) || DEFAULT_SEED;
  promptInput.value = promptInput.value || prompt;
  const startedAt = performance.now();

  try {
    bindPrompt(prompt);
    predictedText.textContent = '';
    outputLabel.textContent = 'CSS model output';
    latencyBadge.textContent = 'running';
    stageLatency.textContent = 'running';
    stageStatus.textContent = 'resolving recurrent math in the CSS style engine.';
    systemStatus.textContent = 'CSS inference';
    body.classList.add('is-predicting');

    const generated = readGeneratedText();
    const latency = Math.round(performance.now() - startedAt);
    predictedText.textContent = generated;
    latencyBadge.textContent = `${latency}ms`;
    stageLatency.textContent = `${latency}ms`;
    signalValue.textContent = String(generated.length);
    tokenCount.textContent = String(generated.length);
    stageStatus.textContent = `complete · ${generated.length} characters · neural ops computed by CSS.`;
    stageMeter.style.width = '100%';
  } catch (error) {
    showRuntimeError(error instanceof Error ? error.message : String(error));
  } finally {
    body.classList.remove('is-predicting');
  }
}

function resetModel() {
  hasInteracted = true;
  promptInput.value = '';
  bindPrompt('');
  predictedText.textContent = 'CSS graph ready. Seed it with a phrase.';
  outputLabel.textContent = 'CSS model output';
  latencyBadge.textContent = 'CSS';
  stageLatency.textContent = 'not run';
  stageStatus.textContent = 'CSS model ready. type a seed and run it.';
  signalValue.textContent = '0';
  tokenCount.textContent = '0';
  stageMeter.style.width = '100%';
  systemStatus.textContent = 'CSS style engine';
  body.classList.remove('is-predicting');
}

function heatNode(node) {
  node.classList.add('is-hot');
  window.setTimeout(() => node.classList.remove('is-hot'), 560);
}

function pulseCluster() {
  const hotNodes = nodes.slice().sort(() => Math.random() - 0.5).slice(0, Math.floor(Math.random() * 3) + 3);
  hotNodes.forEach((node, index) => window.setTimeout(() => heatNode(node), index * 90));
  clusterCount.textContent = `${String(hotNodes.length).padStart(2, '0')} / 09`;
  clusterLoad.textContent = 'compiled';
}

anticipateButton.addEventListener('click', runPrediction);
document.querySelector('#resetButton').addEventListener('click', resetModel);

promptInput.addEventListener('input', (event) => {
  hasInteracted = true;
  if (cleanPrompt(event.target.value)) {
    predictedText.textContent = 'Seed staged. Run the CSS graph when ready.';
    stageStatus.textContent = 'prompt ready. bind it into the CSS graph.';
    signalValue.textContent = String(Math.min(PROMPT_STEPS, cleanPrompt(event.target.value).length));
  } else {
    predictedText.textContent = 'CSS graph ready. Seed it with a phrase.';
    stageStatus.textContent = 'CSS model ready. type a seed and run it.';
    signalValue.textContent = '0';
  }
});

document.querySelector('#rerouteButton').addEventListener('click', () => {
  hasInteracted = true;
  pulseCluster();
  stageStatus.textContent = 'diagnostic pulse sent through the visual CSS graph.';
});

document.querySelector('#posterButton').addEventListener('click', (event) => {
  const posterMode = body.classList.toggle('poster-mode');
  event.currentTarget.textContent = posterMode ? 'return to instrument' : 'make a poster';
});

nodes.forEach((node) => node.addEventListener('click', () => {
  hasInteracted = true;
  heatNode(node);
  stageStatus.textContent = `${node.querySelector('b').textContent} is a visual hidden-unit group; the stylesheet does the computation.`;
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

bindPrompt('');
setReadyState();
