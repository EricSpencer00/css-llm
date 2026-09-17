const engine = document.querySelector('[data-css-rnn-engine]');
const predictionOutput = document.querySelector('#predictionOutput');
const promptForm = document.querySelector('#promptForm');
const promptInput = document.querySelector('#promptInput');
const runButton = document.querySelector('#runButton');
const predictedText = document.querySelector('#predictedText');
const outputLabel = document.querySelector('#outputLabel');
const loadingIndicator = document.querySelector('#loadingIndicator');

const VOCABULARY = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\'\n:;(){}[]+-*/=#_<>\"%&|";
const PROMPT_STEPS = 32;
const GENERATED_STEPS = 96;
const DEFAULT_SEED = 'what can you do?';
const MAX_MESSAGE_LENGTH = 72;

let modelReady = false;
function cssVar(prefix, index, character) {
  return character === undefined ? `--${prefix}-${index}` : `--${prefix}-${index}-${character}`;
}

function setLoading(loading) {
  predictionOutput.setAttribute('aria-busy', String(loading));
  loadingIndicator.hidden = !loading;
  runButton.disabled = loading || !modelReady;
}

function verifyCSSGraph() {
  const stylesheet = [...document.styleSheets].find((sheet) => sheet.href?.includes('css-model.css'));
  if (!engine || !stylesheet || !stylesheet.cssRules || stylesheet.cssRules.length < 20000) {
    throw new Error('css-model.css did not load');
  }

  const output = Number.parseFloat(getComputedStyle(engine).getPropertyValue(cssVar('y', 0)));
  if (!Number.isFinite(output)) throw new Error('computed CSS output is unavailable');
}

function bindPrompt(value) {
  const normalized = [...value.toLowerCase()]
    .map((character) => (VOCABULARY.includes(character) ? character : ' '))
    .join('')
    .slice(-PROMPT_STEPS)
    .padStart(PROMPT_STEPS, ' ');

  for (let step = 0; step < PROMPT_STEPS; step += 1) {
    const token = VOCABULARY.indexOf(normalized[step]);
    for (let character = 0; character < VOCABULARY.length; character += 1) {
      engine.style.setProperty(cssVar('p', step, character), token === character ? '1' : '0');
    }
  }
}

function readGeneratedText() {
  const computed = getComputedStyle(engine);
  const ids = [];
  for (let step = 0; step < GENERATED_STEPS; step += 1) {
    const value = Number.parseFloat(computed.getPropertyValue(cssVar('y', step)));
    if (!Number.isFinite(value)) throw new Error(`missing CSS output at step ${step}`);
    ids.push(Math.max(0, Math.min(VOCABULARY.length - 1, Math.round(value))));
  }
  const text = ids.map((id) => VOCABULARY[id]).join('');
  const end = text.indexOf('\n');
  const line = (end === -1 ? text : text.slice(0, end)).trim();
  for (let size = Math.min(24, Math.floor(line.length / 2)); size >= 8; size -= 1) {
    const suffix = line.slice(-size);
    const previous = line.lastIndexOf(suffix, line.length - size - 1);
    if (previous !== -1) return line.slice(0, previous + size).trim();
  }
  return line;
}

function modelPrompt(message) {
  return `user: ${message}\n`.slice(-PROMPT_STEPS);
}

async function bootModel() {
  setLoading(true);
  try {
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    verifyCSSGraph();
    bindPrompt('assistant: ');
    modelReady = true;
    outputLabel.textContent = 'ready';
  } catch (error) {
    outputLabel.textContent = 'error';
    predictedText.textContent = error.message;
  } finally {
    setLoading(false);
    if (modelReady) promptInput.focus();
  }
}

async function runPrediction(event) {
  event.preventDefault();
  if (!modelReady || runButton.disabled) return;

  const prompt = promptInput.value.trim().slice(0, MAX_MESSAGE_LENGTH) || DEFAULT_SEED;
  promptInput.value = prompt;
  setLoading(true);
  outputLabel.textContent = 'loading';
  predictedText.textContent = '';

  try {
    bindPrompt(modelPrompt(prompt));
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    const generated = readGeneratedText();
    predictedText.textContent = generated;
    outputLabel.textContent = 'response';
    promptInput.value = '';
  } catch (error) {
    outputLabel.textContent = 'error';
    predictedText.textContent = error.message;
  } finally {
    setLoading(false);
    promptInput.focus();
  }
}

promptForm.addEventListener('submit', runPrediction);
bootModel();
