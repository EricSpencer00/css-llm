#!/usr/bin/env python3
"""Train and compile a small conversational character RNN into CSS.

The browser is the inference runtime. JavaScript may bind one-hot prompt
characters and read output ids, but every embedding lookup, matrix multiply,
activation, argmax, and recurrent step is lowered into typed CSS custom
properties.

The export uses aggressive weight-only quantization: the input projection is
stored at signed four bits, recurrent and output matrices at signed eight
bits, and biases at signed eight bits. The generated stylesheet consumes the
quantized matrices and scales directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# 64 symbols is enough for lowercase English, numbers, and the punctuation
# used by compact code examples. Keeping the vocabulary closed is also the
# hard boundary that prevents non-ASCII output from the CSS runtime.
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?\'\n:;(){}[]+-*/=#_<>\"%&|"
VOCAB_SIZE = len(CHARS)
HIDDEN_SIZE = 64
PROMPT_STEPS = 64
GENERATED_STEPS = 96
TRAIN_SEQUENCE_LENGTH = 128
SUPERVISED_TARGET_STEPS = 64

# The input projection is aggressively quantized; recurrent and output
# matrices keep eight bits so the tiny model retains useful state and logits.
QUANTIZATION_BITS = 4
RECURRENT_BITS = 8
OUTPUT_BITS = 8
BIAS_BITS = 8
SPACE_INDEX = CHARS.index(" ")
NEWLINE_INDEX = CHARS.index("\n")
REPETITION_WINDOW = 16
REPETITION_PENALTY = 0.22
TIE_BREAK = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--examples", type=Path, help="JSONL user/assistant pairs for supervised fine-tuning")
    parser.add_argument("--weights", type=Path, default=Path("model/weights.json"))
    parser.add_argument("--css", type=Path, default=Path("css-model.css"))
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=TRAIN_SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def normalize(text: str) -> np.ndarray:
    """Map input to lowercase characters in the closed runtime vocabulary."""

    text = text.replace("<|endoftext|>", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n").lower()
    lookup = {character: index for index, character in enumerate(CHARS)}
    return np.array([lookup.get(character, lookup[" "]) for character in text], dtype=np.int64)


def initialise(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Initialize an RNN with an orthogonal recurrent core."""

    recurrent_seed = rng.normal(0, 1, (HIDDEN_SIZE, HIDDEN_SIZE))
    orthogonal, _ = np.linalg.qr(recurrent_seed)
    hidden_scale = 1.0 / math.sqrt(HIDDEN_SIZE)
    return {
        "wxh": rng.normal(0, hidden_scale, (VOCAB_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "whh": (orthogonal * 0.88).astype(np.float32),
        "bh": np.zeros(HIDDEN_SIZE, dtype=np.float32),
        "why": rng.normal(0, hidden_scale, (HIDDEN_SIZE, VOCAB_SIZE)).astype(np.float32),
        "by": np.zeros(VOCAB_SIZE, dtype=np.float32),
    }


def fake_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Quantize during the forward pass and use a straight-through gradient."""

    qmax = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(values), axis=0, keepdims=True) / qmax
    scales = np.where(scales > 0, scales, 1.0)
    return np.rint(values / scales).clip(-qmax - 1, qmax) * scales


def runtime_parameters(model: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "wxh": fake_quantize(model["wxh"], QUANTIZATION_BITS),
        "whh": fake_quantize(model["whh"], RECURRENT_BITS),
        "bh": fake_quantize(model["bh"], BIAS_BITS),
        "why": fake_quantize(model["why"], OUTPUT_BITS),
        "by": fake_quantize(model["by"], BIAS_BITS),
    }


def loss_and_gradients(
    model: dict[str, np.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    loss_mask: np.ndarray | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """Run quantization-aware teacher forcing and backpropagation through time."""

    batch_size, sequence_length = inputs.shape
    runtime = runtime_parameters(model)
    wxh, whh, bh, why, by = (runtime[name] for name in ("wxh", "whh", "bh", "why", "by"))

    hidden = np.zeros((sequence_length + 1, batch_size, HIDDEN_SIZE), dtype=np.float32)
    logits = np.zeros((sequence_length, batch_size, VOCAB_SIZE), dtype=np.float32)
    for step in range(sequence_length):
        embedded = wxh[inputs[:, step]]
        hidden[step + 1] = np.clip(embedded + hidden[step] @ whh + bh, -1, 1)
        logits[step] = hidden[step + 1] @ why + by

    shifted = logits - logits.max(axis=2, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    time_index = np.arange(sequence_length)[:, None]
    batch_index = np.arange(batch_size)[None, :]
    mask = np.ones((batch_size, sequence_length), dtype=np.float32) if loss_mask is None else loss_mask.astype(np.float32)
    mask_total = max(1.0, float(mask.sum()))
    loss = -(
        np.log(probabilities[time_index, batch_index, targets.T] + 1e-8) * mask.T
    ).sum() / mask_total

    gradients = {name: np.zeros_like(value) for name, value in model.items()}
    d_hidden_next = np.zeros((batch_size, HIDDEN_SIZE), dtype=np.float32)
    for step in range(sequence_length - 1, -1, -1):
        d_logits = probabilities[step].copy()
        d_logits[np.arange(batch_size), targets[:, step]] -= 1
        d_logits *= mask[:, step, None]
        d_logits /= mask_total
        gradients["why"] += hidden[step + 1].T @ d_logits
        gradients["by"] += d_logits.sum(axis=0)

        d_hidden = d_logits @ why.T + d_hidden_next
        d_pre_activation = d_hidden * (np.abs(hidden[step + 1]) < 1)
        gradients["bh"] += d_pre_activation.sum(axis=0)
        gradients["whh"] += hidden[step].T @ d_pre_activation
        np.add.at(gradients["wxh"], inputs[:, step], d_pre_activation)
        d_hidden_next = d_pre_activation @ whh.T

    return float(loss), gradients


def adam_step(
    model: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
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


def learning_rate(step: int, total_steps: int) -> float:
    warmup = min(1.0, step / 1000)
    progress = max(0.0, (step - 1000) / max(1, total_steps - 1000))
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return 0.002 * (0.2 + 0.8 * cosine) * warmup


def train(
    data: np.ndarray,
    seed: int,
    steps: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = initialise(rng)
    moments = {name: np.zeros_like(value) for name, value in model.items()}
    velocities = {name: np.zeros_like(value) for name, value in model.items()}
    if len(data) < sequence_length + 2:
        raise ValueError("The corpus is too short to train the model.")

    for step in range(1, steps + 1):
        starts = rng.integers(0, len(data) - sequence_length - 1, size=batch_size)
        inputs = np.stack([data[start : start + sequence_length] for start in starts])
        targets = np.stack([data[start + 1 : start + sequence_length + 1] for start in starts])
        loss, gradients = loss_and_gradients(model, inputs, targets)
        rate = learning_rate(step, steps)
        adam_step(model, gradients, moments, velocities, step, rate)
        if step == 1 or step % 250 == 0:
            print(f"step {step:05d}/{steps} · loss {loss:.4f} · lr {rate:.5f}")
    return model


def load_examples(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user = str(row.get("user", "")).strip()
        assistant = str(row.get("assistant", "")).strip()
        if user and assistant:
            examples.append({"user": user, "assistant": assistant})
    if not examples:
        raise ValueError("The supervised examples file contains no user/assistant pairs.")
    return examples


def supervised_batch(
    examples: list[dict[str, str]],
    rng: np.random.Generator,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align user turns to the exact fixed prompt context consumed by CSS."""

    sequence_length = PROMPT_STEPS + SUPERVISED_TARGET_STEPS - 1
    space_id = SPACE_INDEX
    inputs = np.full((batch_size, sequence_length), space_id, dtype=np.int64)
    targets = np.full((batch_size, sequence_length), space_id, dtype=np.int64)
    loss_mask = np.zeros((batch_size, sequence_length), dtype=np.float32)
    for batch_index in range(batch_size):
        example = examples[int(rng.integers(0, len(examples)))]
        context = normalize(f"user: {example['user']}\nassistant: ")[-PROMPT_STEPS:]
        inputs[batch_index, :PROMPT_STEPS] = np.pad(
            context,
            (PROMPT_STEPS - len(context), 0),
            constant_values=space_id,
        )
        answer = normalize(example["assistant"] + "\n")[:SUPERVISED_TARGET_STEPS]
        if len(answer):
            inputs[batch_index, PROMPT_STEPS : PROMPT_STEPS + len(answer) - 1] = answer[:-1]
            target_start = PROMPT_STEPS - 1
            targets[batch_index, target_start : target_start + len(answer)] = answer
            loss_mask[batch_index, target_start : target_start + len(answer)] = 1
    return inputs, targets, loss_mask


def train_supervised(
    examples: list[dict[str, str]],
    seed: int,
    steps: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    model = initialise(rng)
    moments = {name: np.zeros_like(value) for name, value in model.items()}
    velocities = {name: np.zeros_like(value) for name, value in model.items()}
    for step in range(1, steps + 1):
        inputs, targets, loss_mask = supervised_batch(examples, rng, batch_size)
        loss, gradients = loss_and_gradients(model, inputs, targets, loss_mask)
        rate = learning_rate(step, steps)
        adam_step(model, gradients, moments, velocities, step, rate)
        if step == 1 or step % 250 == 0:
            print(f"step {step:05d}/{steps} · supervised loss {loss:.4f} · lr {rate:.5f}")
    return model


def evaluate_loss(model: dict[str, np.ndarray], data: np.ndarray, sequence_length: int, samples: int = 16) -> float:
    starts = np.linspace(0, len(data) - sequence_length - 2, samples, dtype=np.int64)
    inputs = np.stack([data[start : start + sequence_length] for start in starts])
    targets = np.stack([data[start + 1 : start + sequence_length + 1] for start in starts])
    return loss_and_gradients(model, inputs, targets)[0]


def evaluate_supervised_loss(model: dict[str, np.ndarray], examples: list[dict[str, str]], seed: int) -> float:
    rng = np.random.default_rng(seed)
    inputs, targets, loss_mask = supervised_batch(examples, rng, 64)
    return loss_and_gradients(model, inputs, targets, loss_mask)[0]


def quantize_columns(values: np.ndarray, bits: int) -> dict[str, Any]:
    qmax = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(values), axis=0) / qmax
    scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    quantized = np.rint(values / scales).clip(-qmax - 1, qmax).astype(np.int8)
    return {
        "bits": bits,
        "shape": list(values.shape),
        "q": quantized.astype(int).tolist(),
        "scales": scales.astype(float).tolist(),
    }


def quantize_model(model: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "wxh": quantize_columns(model["wxh"], QUANTIZATION_BITS),
        "whh": quantize_columns(model["whh"], RECURRENT_BITS),
        "why": quantize_columns(model["why"], OUTPUT_BITS),
        "bh": quantize_columns(model["bh"][None, :], BIAS_BITS),
        "by": quantize_columns(model["by"][None, :], BIAS_BITS),
    }


def validate_tensors(tensors: dict[str, Any]) -> None:
    expected = {
        "wxh": (VOCAB_SIZE, HIDDEN_SIZE),
        "whh": (HIDDEN_SIZE, HIDDEN_SIZE),
        "why": (HIDDEN_SIZE, VOCAB_SIZE),
        "bh": (1, HIDDEN_SIZE),
        "by": (1, VOCAB_SIZE),
    }
    for name, shape in expected.items():
        actual = tuple(tensors[name].get("shape", ()))
        if actual != shape:
            raise ValueError(f"{name} has shape {actual}; expected {shape} for CSS-RNN-{HIDDEN_SIZE}")


def css_number(value: float) -> str:
    value = float(np.float32(value))
    return "0" if value == 0 else format(value, ".9g")


VAR_PREFIXES = {
    "prompt": "p",
    "prompt-hidden": "ph",
    "generated-hidden": "gh",
    "logit": "l",
    "maximum": "x",
    "mask": "m",
    "output": "y",
}


def css_var(prefix: str, index: int, unit: int | None = None) -> str:
    name = VAR_PREFIXES[prefix]
    return f"--{name}-{index}" if unit is None else f"--{name}-{index}-{unit}"


def sum_expression(terms: list[str]) -> str:
    return "calc(" + " + ".join(terms) + ")"


def hard_tanh_expression(terms: list[str]) -> str:
    return "clamp(-1, " + sum_expression(terms) + ", 1)"


def qvalue(tensor: dict[str, Any], row: int, column: int) -> str:
    return f"calc({tensor['q'][row][column]} * {css_number(tensor['scales'][column])})"


def bias_value(tensor: dict[str, Any], index: int) -> str:
    return f"calc({tensor['q'][0][index]} * {css_number(tensor['scales'][index])})"


def register_property(registrations: list[str], name: str) -> None:
    registrations.append(f'@property {name} {{ syntax: "<number>"; inherits: false; initial-value: 0; }}')


def hidden_declaration(
    output: str,
    input_prefix: str,
    step: int,
    unit: int,
    previous_prefix: str | None,
    previous_step: int | None,
    tensors: dict[str, Any],
    registrations: list[str],
) -> str:
    register_property(registrations, output)
    terms = [bias_value(tensors["bh"], unit)]
    for character in range(VOCAB_SIZE):
        terms.append(
            f"var({css_var(input_prefix, step, character)}) * "
            f"{qvalue(tensors['wxh'], character, unit)}"
        )
    for previous_unit in range(HIDDEN_SIZE):
        previous = "0" if previous_prefix is None else f"var({css_var(previous_prefix, previous_step, previous_unit)})"
        terms.append(f"{previous} * {qvalue(tensors['whh'], previous_unit, unit)}")
    return f"  {output}: {hard_tanh_expression(terms)};"


def generate_css(
    tensors: dict[str, Any],
    corpus_sha256: str,
    training_steps: int,
    seed: int,
    validation_loss: float,
) -> str:
    validate_tensors(tensors)
    declarations: list[str] = []
    registrations: list[str] = []
    for step in range(PROMPT_STEPS):
        for character in range(VOCAB_SIZE):
            name = css_var("prompt", step, character)
            register_property(registrations, name)
            declarations.append(f"  {name}: 0;")

    def emit_hidden_step(
        step: int,
        input_prefix: str,
        previous_prefix: str | None,
        previous_step: int | None,
        hidden_prefix: str,
    ) -> None:
        for unit in range(HIDDEN_SIZE):
            name = css_var(hidden_prefix, step, unit)
            declarations.append(
                hidden_declaration(
                    name,
                    input_prefix,
                    step,
                    unit,
                    previous_prefix,
                    previous_step,
                    tensors,
                    registrations,
                )
            )

    for step in range(PROMPT_STEPS):
        emit_hidden_step(
            step,
            "prompt",
            "prompt-hidden" if step else None,
            step - 1 if step else None,
            "prompt-hidden",
        )

    previous_prefix = "prompt-hidden"
    previous_step = PROMPT_STEPS - 1
    for step in range(GENERATED_STEPS):
        for character in range(VOCAB_SIZE):
            name = css_var("logit", step, character)
            register_property(registrations, name)
            terms = [bias_value(tensors["by"], character)]
            terms.extend(
                f"var({css_var(previous_prefix, previous_step, unit)}) * "
                f"{qvalue(tensors['why'], unit, character)}"
                for unit in range(HIDDEN_SIZE)
            )
            if character == SPACE_INDEX:
                terms.append("-5" if step == 0 else f"-2 * var({css_var('mask', step - 1, SPACE_INDEX)})")
            elif character == NEWLINE_INDEX:
                terms.append("-2" if step == 0 else f"-1 * var({css_var('mask', step - 1, NEWLINE_INDEX)})")
            elif step:
                for previous in range(max(0, step - REPETITION_WINDOW), step):
                    terms.append(f"-{css_number(REPETITION_PENALTY)} * var({css_var('mask', previous, character)})")
            terms.append(css_number(character * TIE_BREAK))
            declarations.append(f"  {name}: {sum_expression(terms)};")

        maximum = css_var("maximum", step)
        register_property(registrations, maximum)
        declarations.append(
            f"  {maximum}: max({', '.join(f'var({css_var('logit', step, character)})' for character in range(VOCAB_SIZE))});"
        )
        masks: list[str] = []
        for character in range(VOCAB_SIZE):
            name = css_var("mask", step, character)
            register_property(registrations, name)
            logit = css_var("logit", step, character)
            declarations.append(f"  {name}: calc(1 - abs(sign(calc(var({logit}) - var({maximum})))));")
            masks.append(f"{character} * var({name})")
        output = css_var("output", step)
        register_property(registrations, output)
        declarations.append(f"  {output}: {sum_expression(masks)};")

        if step < GENERATED_STEPS - 1:
            emit_hidden_step(step, "mask", previous_prefix, previous_step, "generated-hidden")
            previous_prefix = "generated-hidden"
            previous_step = step

    header = f"""/*
 * CSS-RNN-{HIDDEN_SIZE}-Q4 · quantized character language model
 * {HIDDEN_SIZE} hidden units · {VOCAB_SIZE}-symbol closed vocabulary
 * {PROMPT_STEPS}-character context · {GENERATED_STEPS}-step greedy rollout
 * matrices: signed 4-bit input values · signed 8-bit recurrent/output values
 * biases: signed 8-bit per-channel values
 * decoder: lowercase ASCII vocabulary, whitespace guard, and repetition penalty
 * corpus sha256: {corpus_sha256}
 * training steps: {training_steps} · seed: {seed} · validation loss: {css_number(validation_loss)}
 * Every multiply, sum, activation, comparison, argmax, and recurrent step below
 * is evaluated by the browser's CSS style engine.
 */
"""
    return header + "\n".join(registrations) + "\n\n[data-css-rnn-engine] {\n" + "\n".join(declarations) + "\n}\n"


def save_weights(
    path: Path,
    tensors: dict[str, Any],
    corpus_sha256: str,
    steps: int,
    seed: int,
    validation_loss: float,
    training_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": f"CSS-RNN-{HIDDEN_SIZE}-Q4",
        "vocabulary": CHARS,
        "hidden_size": HIDDEN_SIZE,
        "prompt_steps": PROMPT_STEPS,
        "generated_steps": GENERATED_STEPS,
        "training_sequence_length": TRAIN_SEQUENCE_LENGTH,
        "supervised_target_steps": SUPERVISED_TARGET_STEPS,
        "training_mode": training_mode,
        "training_steps": steps,
        "seed": seed,
        "corpus_sha256": corpus_sha256,
        "validation_loss": validation_loss,
        "quantization": {
            "input_bits": QUANTIZATION_BITS,
            "recurrent_bits": RECURRENT_BITS,
            "output_bits": OUTPUT_BITS,
            "bias_bits": BIAS_BITS,
            "scheme": "symmetric per-output-channel weight-only quantization with mixed precision",
            "inference": "CSS dequantizes signed integers times scales inside calc()",
        },
        "weights": tensors,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.sequence_length != TRAIN_SEQUENCE_LENGTH:
        raise ValueError(f"--sequence-length must remain {TRAIN_SEQUENCE_LENGTH} for this model")
    raw_corpus = args.corpus.read_bytes()
    corpus_sha256 = hashlib.sha256(raw_corpus).hexdigest()
    data = normalize(raw_corpus.decode("utf-8", errors="replace"))
    examples = load_examples(args.examples) if args.examples else None
    if examples:
        model = train_supervised(examples, args.seed, args.steps, args.batch_size)
        validation_loss = evaluate_supervised_loss(model, examples, args.seed)
        training_mode = "supervised user/assistant next-character fine-tuning"
    else:
        model = train(data, args.seed, args.steps, args.batch_size, args.sequence_length)
        validation_loss = evaluate_loss(model, data, args.sequence_length)
        training_mode = "causal next-character language modeling"
    tensors = quantize_model(model)
    save_weights(args.weights, tensors, corpus_sha256, args.steps, args.seed, validation_loss, training_mode)
    args.css.parent.mkdir(parents=True, exist_ok=True)
    args.css.write_text(generate_css(tensors, corpus_sha256, args.steps, args.seed, validation_loss))
    print(f"validation loss {validation_loss:.4f}")
    print(f"wrote {args.weights} and {args.css}")


if __name__ == "__main__":
    main()
