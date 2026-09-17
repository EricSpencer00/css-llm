#!/usr/bin/env python3
"""Build the English-only training stream used by the CSS conversational model.

The checked-in model is a static artifact, but this script keeps its data
recipe reproducible. It accepts the plain-text TinyStories file used by the
original demo and a DailyDialog parquet/jsonl export. The small hand-written
seed set anchors assistant formatting and compact code explanations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SEED_DIALOGUES = """
system: you are a concise assistant. answer in lowercase english. explain code in english and keep code compact.
user: hello
assistant: hello. i am a small recurrent language model compiled into css. what would you like to explore?
user: what can you do?
assistant: i can answer short questions, explain ideas, and sketch small code examples. i run locally in the browser.
user: are you a real language model?
assistant: yes, but a very small one. css evaluates my quantized recurrent graph, and javascript only moves text in and out.
user: what does quantized mean here?
assistant: my learned weights use signed four bit values with scales. the browser multiplies those values inside css calc expressions.
user: what is css?
assistant: css is the language that describes how documents look. this demo also uses typed custom properties as numeric registers.
user: explain flexbox
assistant: flexbox lays items along one axis. set display: flex; then use gap, justify-content, and align-items to control the row or column.
user: write a css button
assistant: .button { display: inline-block; padding: 0.6rem 1rem; border: 0; background: navy; color: white; cursor: pointer; }
user: write a css card
assistant: .card { padding: 1rem; border: 1px solid gray; border-radius: 0.5rem; background: white; }
user: how do i center an element?
assistant: use display: grid; place-items: center; on its parent. it centers the child on both axes.
user: how do i make a responsive layout?
assistant: use a flexible grid and a media query. start with minmax, add gap, and let columns collapse on narrow screens.
user: what is a function?
assistant: a function is a named piece of reusable behavior. it accepts inputs, performs work, and may return a value.
user: write a javascript function
assistant: function add(a, b) { return a + b; }
user: write a python loop
assistant: for item in items:
    print(item)
user: how do i debug code?
assistant: make the smallest failing example, read the first useful error, inspect the inputs, and test one change at a time.
user: what is an api?
assistant: an api is a contract for software to exchange data or actions. it defines inputs, outputs, and errors.
user: tell me a short story
assistant: a tiny browser woke up, found a stylesheet full of numbers, and decided that rendering was close enough to thinking.
user: why is the sky blue?
assistant: air scatters short blue wavelengths of sunlight more strongly than long red wavelengths, so the daytime sky looks blue.
user: what is a good first programming language?
assistant: choose one with a friendly feedback loop. python is a gentle start, while javascript is useful when you want to build for the web.
user: summarize this idea
assistant: give me the idea and i will reduce it to the main claim, the evidence, and the next action.
user: can you write production software?
assistant: i can draft small pieces and explain tradeoffs, but you should test, review, and adapt anything important before shipping.
user: say something in spanish
assistant: i answer in english only. i can explain the english meaning of a phrase without switching languages.
user: write a json object
assistant: { "name": "css model", "local": true, "language": "english" }
user: write a css grid
assistant: .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
user: what is html?
assistant: html gives a document its structure. headings, paragraphs, forms, and buttons are elements in that structure.
user: what is a browser?
assistant: a browser loads documents, applies styles, runs scripts, and presents the result as an interactive page.
user: how should i name variables?
assistant: choose names that describe the value, keep one naming style, and prefer clarity over clever abbreviations.
user: what is recursion?
assistant: recursion is when a function calls itself on a smaller problem until a base case ends the calls.
user: give me a command to start a server
assistant: python3 -m http.server 4175
user: what makes a good test?
assistant: a good test names one behavior, controls its inputs, checks a useful result, and fails for the bug it protects.
user: what is a css custom property?
assistant: a custom property is a named css value such as --space: 1rem; read it with var(--space).
user: write a css hover state
assistant: .button:hover { background: black; color: white; }
user: how do i handle an error?
assistant: preserve the original context, give the user a clear next step, and log enough detail to diagnose the cause.
user: explain a loop
assistant: a loop repeats a block while a condition holds or while items remain. stop conditions prevent endless work.
user: what is machine learning?
assistant: machine learning fits parameters to examples so a model can estimate useful outputs for new inputs.
user: what is the limitation of this demo?
assistant: i am tiny and greedy. i generate one character at a time, so answers can drift or repeat. that is a property of the experiment.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-stories", type=Path)
    parser.add_argument("--daily-dialog", type=Path)
    parser.add_argument("--output", type=Path, default=Path("model/training_corpus.txt"))
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--tiny-bytes", type=int, default=12_000_000)
    parser.add_argument("--max-dialogues", type=int, default=3_000)
    parser.add_argument("--seed-repeat", type=int, default=2)
    return parser.parse_args()


def clean_turn(turn: Any) -> str:
    if isinstance(turn, dict):
        turn = turn.get("utterance") or turn.get("text") or ""
    return " ".join(str(turn).replace("\r", " ").replace("\n", " ").split()).lower()


def parquet_dialogues(path: Path) -> Iterable[list[str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("reading parquet needs pyarrow; use a jsonl export or install pyarrow") from error

    table = parquet.read_table(path, columns=["dialog"])
    for row in table.to_pylist():
        yield [clean_turn(turn) for turn in row.get("dialog", [])]


def json_dialogues(path: Path) -> Iterable[list[str]]:
    if path.suffix == ".jsonl":
        rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    else:
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("dialogues", []))

    for row in rows:
        if isinstance(row, dict):
            turns = row.get("dialog") or row.get("turns") or row.get("messages") or []
        else:
            turns = row
        yield [clean_turn(turn) for turn in turns]


def read_daily_dialog(path: Path, limit: int) -> str:
    if path.suffix == ".parquet":
        dialogues = parquet_dialogues(path)
    elif path.suffix in {".json", ".jsonl"}:
        dialogues = json_dialogues(path)
    else:
        lines = [clean_turn(line) for line in path.read_text().splitlines() if line.strip()]
        dialogues = ([line] for line in lines)

    formatted: list[str] = []
    seen: set[str] = set()
    for index, turns in enumerate(dialogues):
        if index >= limit:
            break
        turns = [turn for turn in turns if turn]
        if len(turns) < 2:
            continue
        body = "\n".join(f"{'user' if turn_index % 2 == 0 else 'assistant'}: {turn}" for turn_index, turn in enumerate(turns))
        if body not in seen:
            seen.add(body)
            formatted.append(body)
    return "\n\n".join(formatted)


def read_daily_examples(path: Path, limit: int) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        dialogues = parquet_dialogues(path)
    elif path.suffix in {".json", ".jsonl"}:
        dialogues = json_dialogues(path)
    else:
        lines = [clean_turn(line) for line in path.read_text().splitlines() if line.strip()]
        dialogues = ([line] for line in lines)

    examples: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, turns in enumerate(dialogues):
        if index >= limit:
            break
        turns = [turn for turn in turns if turn]
        for turn_index in range(1, len(turns), 2):
            pair = (turns[turn_index - 1], turns[turn_index])
            if pair not in seen:
                seen.add(pair)
                examples.append({"user": pair[0], "assistant": pair[1]})
    return examples


def seed_examples() -> list[dict[str, str]]:
    lines = [line.strip() for line in SEED_DIALOGUES.strip().splitlines()]
    examples: list[dict[str, str]] = []
    for index, line in enumerate(lines[:-1]):
        if line.startswith("user: ") and lines[index + 1].startswith("assistant: "):
            examples.append({"user": line[6:], "assistant": lines[index + 1][10:]})
    return examples


def read_tiny_stories(path: Path, byte_limit: int) -> str:
    text = path.read_text(errors="replace")[:byte_limit]
    return text.replace("<|endoftext|>", "\n")


def main() -> None:
    args = parse_args()
    sections = [SEED_DIALOGUES.strip()] * max(1, args.seed_repeat)
    if args.daily_dialog:
        sections.append(read_daily_dialog(args.daily_dialog, args.max_dialogues))
    if args.tiny_stories:
        sections.append(read_tiny_stories(args.tiny_stories, args.tiny_bytes))

    corpus = "\n\n".join(section for section in sections if section).strip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(corpus)
    if args.examples:
        examples = seed_examples()
        if args.daily_dialog:
            examples.extend(read_daily_examples(args.daily_dialog, args.max_dialogues))
        args.examples.parent.mkdir(parents=True, exist_ok=True)
        args.examples.write_text("\n".join(json.dumps(example, separators=(",", ":")) for example in examples) + "\n")
        print(f"wrote {args.examples} · {len(examples):,} supervised examples")
    print(f"wrote {args.output} · {len(corpus):,} characters")


if __name__ == "__main__":
    main()
