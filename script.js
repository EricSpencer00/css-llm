const body = document.body;
const engine = document.querySelector('[data-css-rnn-engine]');
const gateScreen = document.querySelector('#gateScreen');
const bootScreen = document.querySelector('#bootScreen');
const appScreen = document.querySelector('#appScreen');
const enterButton = document.querySelector('#enterButton');
const retryButton = document.querySelector('#retryButton');
const bootStatus = document.querySelector('#bootStatus');
const bootMeter = document.querySelector('#bootMeter');
const bootProgress = document.querySelector('.boot-progress');
const promptForm = document.querySelector('#promptForm');
const promptInput = document.querySelector('#promptInput');
const runButton = document.querySelector('#runButton');
const predictedText = document.querySelector('#predictedText');
const latencyBadge = document.querySelector('#latencyBadge');
const stageLatency = document.querySelector('#stageLatency');
const tokenCount = document.querySelector('#tokenCount');
const outputLabel = document.querySelector('#outputLabel');
const bootSteps = [
  document.querySelector('#bootStepStylesheet'),
  document.querySelector('#bootStepProperties'),
  document.querySelector('#bootStepGraph'),
];

const VOCABULARY = "abcdefghijklmnopqrstuvwxyz .,!?\'";
const PROMPT_STEPS = 24;
const GENERATED_STEPS = 64;
const DEFAULT_SEED = 'once upon a time';
let modelReady = false;
let booting = false;

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function cleanPrompt(value) {
  return value.replace(/[“”]/g, '').trim();
}

function cssVar(prefix, index, character) {
  return `--css-rnn-${prefix}-${index}-${character}`;
}

function setScreen(screen) {
  gateScreen.hidden = screen !== 'gate';
  bootScreen.hidden = screen !== 'boot';
  appScreen.hidden = screen !== 'app';
  body.classList.remove('is-gate', 'is-booting', 'is-ready');
  body.classList.add(screen === 'gate' ? 'is-gate' : screen === 'boot' ? 'is-booting' : 'is-ready');
}

function setBootProgress(value) {
  const rounded = Math.max(0, Math.min(100, Math.round(value)));
  bootMeter.style.width = `${rounded}%`;
  bootProgress.setAttribute('aria-valuenow', String(rounded));
}

function markBootStep(index) {
  bootSteps[index].classList.add('is-done');
}

function findModelStylesheet() {
  return [...document.styleSheets].find((sheet) => sheet.href?.includes('css-model.css'));
}

function verifyCSSGraph() {
  if (!engine) throw new Error('engine element missing');
  const stylesheet = findModelStylesheet();
  if (!stylesheet) throw new Error('css-model.css did not load');
  if (!stylesheet.cssRules || stylesheet.cssRules.length < 7000) {
    throw new Error('compiled CSS graph is incomplete');
  }

  const computed = getComputedStyle(engine);
  const output = Number.parseFloat(computed.getPropertyValue('--css-rnn-output-0'));
  if (!Number.isFinite(output)) throw new Error('computed CSS output is unavailable');
  return stylesheet.cssRules.length;
}

async function bootModel() {
  if (booting || modelReady) return;
  booting = true;
  modelReady = false;
  enterButton.disabled = true;
  retryButton.hidden = true;
  setScreen('boot');
  bootSteps.forEach((step) => step.classList.remove('is-done'));
  setBootProgress(8);
  bootStatus.textContent = 'checking css-model.css';

  try {
    await delay(180);
    const ruleCount = verifyCSSGraph();
    markBootStep(0);
    bootStatus.textContent = `css-model.css · ${ruleCount.toLocaleString()} rules`;
    setBootProgress(36);

    await delay(240);
    bindPrompt('');
    const hiddenState = Number.parseFloat(getComputedStyle(engine).getPropertyValue('--css-rnn-prompt-hidden-0-0'));
    if (!Number.isFinite(hiddenState)) throw new Error('typed properties did not resolve');
    markBootStep(1);
    bootStatus.textContent = 'numeric properties resolved';
    setBootProgress(68);

    await delay(260);
    verifyCSSGraph();
    markBootStep(2);
    bootStatus.textContent = 'recurrent graph ready';
    setBootProgress(100);

    await delay(320);
    modelReady = true;
    booting = false;
    setScreen('app');
    promptInput.focus();
  } catch (error) {
    booting = false;
    setBootProgress(0);
    bootStatus.textContent = 'error';
    retryButton.hidden = false;
    retryButton.focus();
  }
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

async function runPrediction(event) {
  event.preventDefault();
  if (!modelReady || runButton.disabled) return;

  const prompt = cleanPrompt(promptInput.value) || DEFAULT_SEED;
  promptInput.value = promptInput.value || prompt;
  const startedAt = performance.now();
  runButton.disabled = true;
  latencyBadge.textContent = 'running';
  stageLatency.textContent = 'running';
  outputLabel.textContent = 'next characters';
  predictedText.textContent = '';

  try {
    bindPrompt(prompt);
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    const generated = readGeneratedText();
    const latency = Math.round(performance.now() - startedAt);
    predictedText.textContent = generated;
    latencyBadge.textContent = `${latency}ms`;
    stageLatency.textContent = `${latency}ms`;
    tokenCount.textContent = String(generated.length);
  } catch (error) {
    predictedText.textContent = 'error';
    latencyBadge.textContent = 'error';
    stageLatency.textContent = 'error';
  } finally {
    runButton.disabled = false;
  }
}

enterButton.addEventListener('click', bootModel);
retryButton.addEventListener('click', bootModel);
promptForm.addEventListener('submit', runPrediction);

bindPrompt('');
