const body = document.body;
const promptInput = document.querySelector('#promptInput');
const predictedText = document.querySelector('#predictedText');
const latencyBadge = document.querySelector('#latencyBadge');
const stageLatency = document.querySelector('#stageLatency');
const stageStatus = document.querySelector('#stageStatus');
const stageMeter = document.querySelector('#stageMeter');
const signalValue = document.querySelector('#signalValue');
const confidenceValue = document.querySelector('#confidenceValue');
const systemStatus = document.querySelector('#systemStatus');
const recursionValue = document.querySelector('#recursionValue');
const stackDepth = document.querySelector('#stackDepth');
const clusterCount = document.querySelector('#clusterCount');
const clusterLoad = document.querySelector('#clusterLoad');
const toothbrushBus = document.querySelector('#toothbrushBus');

const examplePrompts = [
  'can css really do this?',
  'what is the model trained on?',
  'is the toothbrush plugged in?',
  'please explain negative latency',
  'why is this running in a stylesheet?',
];

const latencies = ['−72h', '−3d', '−∞ms', '−71h 59m'];
const nodes = [...document.querySelectorAll('.toothbrush-node')];
let hasInteracted = false;

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function cleanPrompt(value) {
  return value.replace(/[“”]/g, '').trim();
}

function setPrediction(value, isFresh = false) {
  const prompt = cleanPrompt(value) || randomItem(examplePrompts);
  predictedText.textContent = `“${prompt}”`;

  if (isFresh) {
    const latency = randomItem(latencies);
    latencyBadge.textContent = latency;
    stageLatency.textContent = latency;
    const confidence = (99.1 + Math.random() * 0.8).toFixed(1);
    confidenceValue.textContent = `${confidence}%`;
    signalValue.textContent = `${(Math.random() * 0.9 + 0.1).toFixed(2)}∞`;
    stageMeter.style.width = `${Math.round(62 + Math.random() * 34)}%`;
  }
}

function runPrediction() {
  hasInteracted = true;
  const prompt = cleanPrompt(promptInput.value) || randomItem(examplePrompts);
  const latency = randomItem(latencies);
  promptInput.value = promptInput.value || prompt;
  setPrediction(prompt, true);
  latencyBadge.textContent = latency;
  stageLatency.textContent = latency;
  stageStatus.textContent = `already knew you would ask: “${prompt}”`;
  systemStatus.textContent = 'predicting backwards';
  body.classList.add('is-predicting');

  window.setTimeout(() => {
    stageStatus.textContent = 'prompt resolved before the prompt existed.';
    systemStatus.textContent = 'live-ish';
    body.classList.remove('is-predicting');
  }, 1900);
}

function resetModel() {
  hasInteracted = true;
  promptInput.value = '';
  setPrediction('can css really do this?');
  latencyBadge.textContent = '−72h';
  stageLatency.textContent = '−72h';
  stageStatus.textContent = 'waiting for a prompt you have not written yet.';
  signalValue.textContent = '0.∞';
  confidenceValue.textContent = '99.9%';
  stageMeter.style.width = '72%';
  systemStatus.textContent = 'live-ish';
}

function heatNode(node) {
  node.classList.add('is-hot');
  window.setTimeout(() => node.classList.remove('is-hot'), 560);
}

function pulseCluster() {
  const activeNodes = nodes.filter((node) => !node.classList.contains('node-h') && !node.classList.contains('node-i'));
  const hotNodes = activeNodes.sort(() => Math.random() - 0.5).slice(0, Math.floor(Math.random() * 3) + 3);
  hotNodes.forEach((node, index) => window.setTimeout(() => heatNode(node), index * 90));
  clusterLoad.textContent = `${Math.round(54 + Math.random() * 34)}%`;
  clusterCount.textContent = `${String(hotNodes.length).padStart(2, '0')} / 09`;
}

document.querySelector('#anticipateButton').addEventListener('click', runPrediction);
document.querySelector('#resetButton').addEventListener('click', resetModel);

promptInput.addEventListener('input', (event) => {
  hasInteracted = true;
  const value = cleanPrompt(event.target.value);
  if (value) {
    predictedText.textContent = `“${value}”`;
    stageStatus.textContent = 'intercepting keystrokes from the near future.';
    signalValue.textContent = `${Math.min(9.9, value.length / 10).toFixed(1)}∞`;
  } else {
    setPrediction('can css really do this?');
    stageStatus.textContent = 'waiting for a prompt you have not written yet.';
  }
});

document.querySelector('#rerouteButton').addEventListener('click', () => {
  hasInteracted = true;
  pulseCluster();
  stageStatus.textContent = 'rerouted inference through the upstairs bathroom.';
  systemStatus.textContent = 'bristles online';
  window.setTimeout(() => { systemStatus.textContent = 'live-ish'; }, 1200);
});

document.querySelector('#posterButton').addEventListener('click', (event) => {
  const posterMode = body.classList.toggle('poster-mode');
  event.currentTarget.textContent = posterMode ? 'return to instrument' : 'make a poster';
});

nodes.forEach((node) => node.addEventListener('click', () => {
  hasInteracted = true;
  heatNode(node);
  stageStatus.textContent = `${node.querySelector('b').textContent} says the prompt is already solved.`;
}));

document.querySelectorAll('.claim-row').forEach((row) => {
  row.addEventListener('toggle', () => {
    if (row.open && hasInteracted) {
      stageStatus.textContent = 'methodology expanded. credibility unchanged.';
    }
  });
});

window.setInterval(pulseCluster, 2200);
window.setInterval(() => {
  const depth = Math.floor(700 + Math.random() * 300);
  recursionValue.textContent = `${(Math.random() * 9 + 1).toFixed(1)}∞`;
  stackDepth.textContent = depth;
  document.documentElement.style.setProperty('--bar-height', `${Math.round(10 + Math.random() * 38)}px`);
}, 1000);
