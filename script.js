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
const clearButton = document.querySelector('#clearButton');
const predictedText = document.querySelector('#predictedText');
const latencyBadge = document.querySelector('#latencyBadge');
const stageLatency = document.querySelector('#stageLatency');
const tokenCount = document.querySelector('#tokenCount');
const outputLabel = document.querySelector('#outputLabel');
const conversationLog = document.querySelector('#conversationLog');
const conversationEmpty = document.querySelector('#conversationEmpty');
const starterButtons = [...document.querySelectorAll('[data-starter]')];
const bootSteps = [
  document.querySelector('#bootStepStylesheet'),
  document.querySelector('#bootStepProperties'),
  document.querySelector('#bootStepGraph'),
];

const VOCABULARY = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\'\n:;(){}[]+-*/=#_<>\"%&|";
const PROMPT_STEPS = 64;
const GENERATED_STEPS = 96;
const DEFAULT_SEED = 'what can you do?';
const MAX_MESSAGE_LENGTH = 72;
const VAR_PREFIXES = {
  prompt: 'p',
  'prompt-hidden': 'ph',
  'input-proj': 'i',
  'recurrent-proj': 'r',
  'generated-hidden': 'gh',
  'output-proj': 'o',
  logit: 'l',
  maximum: 'x',
  mask: 'm',
  output: 'y',
};

let modelReady = false;
let booting = false;
let turns = [];

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function cleanPrompt(value) {
  return [...value.toLowerCase()]
    .map((character) => (VOCABULARY.includes(character) ? character : ' '))
    .join('')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_MESSAGE_LENGTH);
}

function cssVar(prefix, index, character) {
  const name = VAR_PREFIXES[prefix];
  return character === undefined ? `--${name}-${index}` : `--${name}-${index}-${character}`;
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
  if (!stylesheet.cssRules || stylesheet.cssRules.length < 20000) {
    throw new Error('quantized CSS graph is incomplete');
  }

  const computed = getComputedStyle(engine);
  const output = Number.parseFloat(computed.getPropertyValue(cssVar('output', 0)));
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
  bootStatus.textContent = 'checking quantized stylesheet';

  try {
    await delay(180);
    const ruleCount = verifyCSSGraph();
    markBootStep(0);
    bootStatus.textContent = `q4 css-model.css · ${ruleCount.toLocaleString()} rules`;
    setBootProgress(36);

    await delay(240);
    bindPrompt('assistant: ');
    const hiddenState = Number.parseFloat(getComputedStyle(engine).getPropertyValue(cssVar('prompt-hidden', 0, 0)));
    if (!Number.isFinite(hiddenState)) throw new Error('typed properties did not resolve');
    markBootStep(1);
    bootStatus.textContent = '64-character context resolved';
    setBootProgress(68);

    await delay(260);
    verifyCSSGraph();
    markBootStep(2);
    bootStatus.textContent = '96-step recurrent graph ready';
    setBootProgress(100);

    await delay(320);
    modelReady = true;
    booting = false;
    setScreen('app');
    promptInput.focus();
  } catch (error) {
    booting = false;
    setBootProgress(0);
    bootStatus.textContent = 'error loading CSS graph';
    retryButton.hidden = false;
    retryButton.focus();
  }
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
      engine.style.setProperty(cssVar('prompt', step, character), token === character ? '1' : '0');
    }
  }
}

function readGeneratedText() {
  const computed = getComputedStyle(engine);
  const ids = [];
  for (let step = 0; step < GENERATED_STEPS; step += 1) {
    const value = Number.parseFloat(computed.getPropertyValue(cssVar('output', step)));
    if (!Number.isFinite(value)) throw new Error(`missing CSS output at step ${step}`);
    ids.push(Math.max(0, Math.min(VOCABULARY.length - 1, Math.round(value))));
  }
  const raw = ids.map((id) => VOCABULARY[id]).join('');
  const boundary = raw.search(/\n(?:user|assistant):/);
  return (boundary === -1 ? raw : raw.slice(0, boundary)).trim();
}

function modelPrompt(message) {
  const history = turns
    .slice(-3)
    .map((turn) => `${turn.role}: ${turn.text}`)
    .join('\n');
  return `${history}${history ? '\n' : ''}user: ${message}\nassistant: `.slice(-PROMPT_STEPS);
}

function appendTurn(role, text) {
  conversationEmpty.hidden = true;
  const item = document.createElement('div');
  item.className = `conversation-turn conversation-turn-${role}`;
  const label = document.createElement('span');
  label.className = 'turn-role';
  label.textContent = role === 'user' ? 'you' : 'assistant';
  const content = document.createElement('p');
  content.textContent = text;
  item.append(label, content);
  conversationLog.append(item);
  conversationLog.scrollTop = conversationLog.scrollHeight;
}

function clearConversation() {
  turns = [];
  conversationLog.querySelectorAll('.conversation-turn').forEach((turn) => turn.remove());
  conversationEmpty.hidden = false;
  predictedText.textContent = '';
  outputLabel.textContent = 'assistant · ready';
  latencyBadge.textContent = 'CSS';
  stageLatency.textContent = 'not run';
  tokenCount.textContent = '0';
  promptInput.focus();
}

function validEnglishTokenStream(text) {
  return text.length > 0
    && text.length <= GENERATED_STEPS
    && [...text].every((character) => VOCABULARY.includes(character) && character === character.toLowerCase())
    && /[a-z]/.test(text);
}

function stableEnglishResponse(text) {
  if (!validEnglishTokenStream(text)) return false;
  const words = text.match(/[a-z]{2,}/g) || [];
  if (words.length < 2) return false;
  return !/(.{2,12})\1\1/.test(text.replace(/\s+/g, ' '));
}

async function runPrediction(event) {
  event.preventDefault();
  if (!modelReady || runButton.disabled) return;

  const prompt = cleanPrompt(promptInput.value) || DEFAULT_SEED;
  promptInput.value = prompt;
  const startedAt = performance.now();
  runButton.disabled = true;
  starterButtons.forEach((button) => { button.disabled = true; });
  latencyBadge.textContent = 'running';
  stageLatency.textContent = 'running';
  outputLabel.textContent = 'assistant · thinking';
  predictedText.textContent = '';

  try {
    bindPrompt(modelPrompt(prompt));
    await new Promise((resolve) => window.requestAnimationFrame(resolve));
    const generated = readGeneratedText();
    if (!stableEnglishResponse(generated)) throw new Error('no stable English response');
    const latency = Math.round(performance.now() - startedAt);
    predictedText.textContent = generated;
    latencyBadge.textContent = `${latency}ms`;
    stageLatency.textContent = `${latency}ms`;
    tokenCount.textContent = String(generated.length);
    outputLabel.textContent = 'assistant · English mode';
    appendTurn('user', prompt);
    appendTurn('assistant', generated);
    turns.push({ role: 'user', text: prompt }, { role: 'assistant', text: generated });
    promptInput.value = '';
  } catch (error) {
    predictedText.textContent = 'the CSS graph returned no English response';
    latencyBadge.textContent = 'error';
    stageLatency.textContent = 'error';
    outputLabel.textContent = 'assistant · unavailable';
  } finally {
    runButton.disabled = false;
    starterButtons.forEach((button) => { button.disabled = false; });
  }
}

enterButton.addEventListener('click', bootModel);
retryButton.addEventListener('click', bootModel);
promptForm.addEventListener('submit', runPrediction);
clearButton.addEventListener('click', clearConversation);
starterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    promptInput.value = button.dataset.starter;
    promptInput.focus();
  });
});

bindPrompt('assistant: ');
