#!/usr/bin/env python3
"""Train a tiny character RNN and lower its fixed-shape inference graph to CSS.

The generated stylesheet contains the weights, matrix multiplies, hard-tanh gates,
argmax, and a fixed autoregressive rollout. JavaScript only binds the prompt
one-hot vectors and reads the generated character ids from computed style.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CHARS = "abcdefghijklmnopqrstuvwxyz .,!?\'"
VOCAB_SIZE = len(CHARS)
HIDDEN_SIZE = 32
PROMPT_STEPS = 24
GENERATED_STEPS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    parser.add_argument("--css", type=Path, default=Path("css-model.css"))
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def normalize(text: str) -> np.ndarray:
    text = text.replace("<|endoftext|>", " ")
    lookup = {character: index for index, character in enumerate(CHARS)}
    return np.array([lookup.get(character, lookup[" "]) for character in text.lower()], dtype=np.int64)


def initialise(rng: np.random.Generator) -> dict[str, np.ndarray]:
    scale = 0.08
    return {
        "wxh": rng.normal(0, scale, (VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "whh": rng.normal(0, scale, (HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "bh": np.zeros(HIDDEN_SIZE, dtype=np.float32),
        "why": rng.normal(0, scale, (HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32),
        "by": np.zeros(VOCAB_SIZE, dtype=np.float32),
    }


def loss_and_gradients(
    model: dict[str, np.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    batch_size, sequence_length = inputs.shape
    wxh, whh, bh, why, by = (model[name] for name in ("wxh", "whh", "bh", "why", "by"))

    hidden = np.zeros((sequence_length + 1, batch_size, HIDDEN_SIZE), dtype=np.float32)
    logits = np.zeros((sequence_length, batch_size, VOCAB_SIZE), dtype=np.float32)
    for step in range(sequence_length):
        embedded = wxh[inputs[:, step]]
        hidden[step + 1] = np.clip(embedded + hidden[step] @ whh + bh, -1, 1)
        logits[step] = hidden[step + 1] @ why + by

    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    loss = -np.log(probabilities[np.arange(sequence_length)[:, None], np.arange(batch_size)[None, :], targets.T] + 1e-8).mean()

    gradients = {name: np.zeros_like(value) for name, value in model.items()}
    d_hidden_next = np.zeros((batch_size, HIDDEN_SIZE), dtype=np.float32)
    normalizer = float(batch_size * sequence_length)

    for step in range(sequence_length - 1, -1, -1):
        d_logits = probabilities[step].copy()
        d_logits[np.arange(batch_size), targets[:, step]] -= 1
        d_logits /= normalizer
        gradients["why"] += hidden[step + 1].T @ d_logits
        gradients["by"] += d_logits.sum(axis=0)
        d_hidden = d_logits @ why.T + d_hidden_next
        d_pre_activation = d_hidden * (np.abs(hidden[step + 1]) < 1)
        gradients["bh"] += d_pre_activation.sum(axis=0)
        gradients["whh"] += hidden[step].T @ d_pre_activation
        np.add.at(gradients["wxh"], inputs[:, step], d_pre_activation)
        d_hidden_next = d_pre_activation @ whh.T

    return float(loss), gradients


def train(data: np.ndarray, seed: int, steps: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = initialise(rng)
    moments = {name: np.zeros_like(value) for name, value in model.items()}
    velocities = {name: np.zeros_like(value) for name, value in model.items()}
    batch_size = 32
    sequence_length = 64
    learning_rate = 0.0025

    if len(data) < sequence_length + 2:
        raise ValueError("The corpus is too short to train the model.")

    for step in range(1, steps + 1):
        starts = rng.integers(0, len(data) - sequence_length - 1, size=batch_size)
        inputs = np.stack([data[start : start + sequence_length] for start in starts])
        targets = np.stack([data[start + 1 : start + sequence_length + 1] for start in starts])
        loss, gradients = loss_and_gradients(model, inputs, targets)

        # Gradient clipping keeps the little recurrent graph stable.
        norm = math.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients.values()))
        if norm > 5:
            for gradient in gradients.values():
                gradient *= 5 / norm

        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for name in model:
            moments[name] = beta1 * moments[name] + (1 - beta1) * gradients[name]
            velocities[name] = beta2 * velocities[name] + (1 - beta2) * gradients[name] ** 2
            corrected_moment = moments[name] / (1 - beta1**step)
            corrected_velocity = velocities[name] / (1 - beta2**step)
            model[name] -= learning_rate * corrected_moment / (np.sqrt(corrected_velocity) + epsilon)

        if step == 1 or step % 100 == 0:
            print(f"step {step:04d}/{steps} · loss {loss:.4f}")

    return model


def css_number(value: float) -> str:
    value = float(np.float32(value))
    if value == 0:
        return "0"
    return format(value, ".9g")


def css_var(prefix: str, index: int, unit: int | None = None) -> str:
    return f"--css-rnn-{prefix}-{index}" if unit is None else f"--css-rnn-{prefix}-{index}-{unit}"


def sum_expression(terms: list[str]) -> str:
    return "calc(" + " + ".join(terms) + ")"


def hard_tanh_expression(terms: list[str]) -> str:
    return "clamp(-1, " + sum_expression(terms) + ", 1)"


def hidden_declaration(
    output: str,
    input_prefix: str,
    step: int,
    unit: int,
    previous_prefix: str | None,
    previous_step: int | None,
    model: dict[str, np.ndarray],
) -> str:
    terms = [css_number(model["bh"][unit])]
    for character in range(VOCAB_SIZE):
        terms.append(f"var({css_var(input_prefix, step, character)}) * {css_number(model['wxh'][character, unit])}")
    for previous_unit in range(HIDDEN_SIZE):
        previous = "0" if previous_prefix is None else f"var({css_var(previous_prefix, previous_step, previous_unit)})"
        terms.append(f"{previous} * {css_number(model['whh'][previous_unit, unit])}")
    return f"  {output}: {hard_tanh_expression(terms)};"


def generate_css(model: dict[str, np.ndarray], corpus_sha256: str) -> str:
    declarations: list[str] = []
    registrations: list[str] = []

    def register(name: str) -> None:
        registrations.append(f'@property {name} {{ syntax: "<number>"; inherits: false; initial-value: 0; }}')

    for step in range(PROMPT_STEPS):
        for character in range(VOCAB_SIZE):
            name = css_var("prompt", step, character)
            register(name)
            declarations.append(f"  {name}: 0;")

    for step in range(PROMPT_STEPS):
        for unit in range(HIDDEN_SIZE):
            name = css_var("prompt-hidden", step, unit)
            register(name)
            declarations.append(hidden_declaration(name, "prompt", step, unit, "prompt-hidden" if step else None, step - 1 if step else None, model))

    previous_hidden = "prompt-hidden"
    previous_step = PROMPT_STEPS - 1
    for step in range(GENERATED_STEPS):
        for character in range(VOCAB_SIZE):
            name = css_var("logit", step, character)
            register(name)
            terms = [css_number(model["by"][character])]
            for unit in range(HIDDEN_SIZE):
                terms.append(f"var({css_var(previous_hidden, previous_step, unit)}) * {css_number(model['why'][unit, character])}")
            declarations.append(f"  {name}: {sum_expression(terms)};")

        maximum = css_var("maximum", step)
        register(maximum)
        logit_terms = ", ".join(f"var({css_var('logit', step, character)})" for character in range(VOCAB_SIZE))
        declarations.append(f"  {maximum}: max({logit_terms});")

        masks: list[str] = []
        for character in range(VOCAB_SIZE):
            name = css_var("mask", step, character)
            register(name)
            logit = css_var("logit", step, character)
            declarations.append(f"  {name}: calc(1 - abs(sign(calc(var({logit}) - var({maximum})))));")
            masks.append(f"{character} * var({name})")

        output = css_var("output", step)
        register(output)
        declarations.append(f"  {output}: {sum_expression(masks)};")

        if step < GENERATED_STEPS - 1:
            next_hidden_prefix = "generated-hidden"
            for unit in range(HIDDEN_SIZE):
                name = css_var(next_hidden_prefix, step, unit)
                register(name)
                declarations.append(hidden_declaration(name, "mask", step, unit, previous_hidden, previous_step, model))
            previous_hidden = next_hidden_prefix
            previous_step = step

    header = f"""/*
 * CSS-RNN-32 · a 32-unit character model
 * {HIDDEN_SIZE} hidden units · {VOCAB_SIZE} character vocabulary
 * {PROMPT_STEPS}-character prompt seed · {GENERATED_STEPS}-step fixed rollout
 * Trained corpus SHA-256: {corpus_sha256}
 * Every multiply, sum, hard-tanh, comparison, argmax, and recurrent step below
 * is evaluated by the browser's CSS style engine.
 */
"""
    return header + "\n".join(registrations) + "\n\n[data-css-rnn-engine] {\n" + "\n".join(declarations) + "\n}\n"


def save_weights(path: Path, model: dict[str, np.ndarray], corpus_sha256: str, steps: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "CSS-RNN-32",
        "vocabulary": CHARS,
        "hidden_size": HIDDEN_SIZE,
        "prompt_steps": PROMPT_STEPS,
        "generated_steps": GENERATED_STEPS,
        "training_steps": steps,
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "weights": {name: value.astype(float).tolist() for name, value in model.items()},
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    raw_corpus = args.corpus.read_bytes()
    corpus_sha256 = hashlib.sha256(raw_corpus).hexdigest()
    data = normalize(raw_corpus.decode("utf-8", errors="replace"))
    model = train(data, args.seed, args.steps)
    save_weights(args.weights, model, corpus_sha256, args.steps, args.seed)
    args.css.parent.mkdir(parents=True, exist_ok=True)
    args.css.write_text(generate_css(model, corpus_sha256))
    print(f"wrote {args.weights} and {args.css}")


if __name__ == "__main__":
    main()
