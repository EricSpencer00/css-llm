#!/usr/bin/env python3
"""Fast regression checks for the CSS model compiler."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_css_rnn as model  # noqa: E402


class CSSModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.parameters = model.initialise(self.rng)

    def test_vocabulary_is_closed_and_lowercase(self) -> None:
        encoded = model.normalize("Hello, café!\r\nprint(1)")
        decoded = "".join(model.CHARS[index] for index in encoded)
        self.assertEqual(decoded, "hello, caf !\nprint(1)")
        self.assertNotIn("é", model.CHARS)

    def test_quantized_matrices_are_signed_four_bit(self) -> None:
        tensors = model.quantize_model(self.parameters)
        matrix_bits = {"wxh": 4, "whh": 8, "why": 8}
        for name, bits in matrix_bits.items():
            values = np.array(tensors[name]["q"])
            self.assertLessEqual(np.abs(values).max(), (1 << (bits - 1)) - 1, name)
            self.assertEqual(tensors[name]["bits"], bits)
        for name in ("bh", "by"):
            self.assertLessEqual(np.abs(np.array(tensors[name]["q"])).max(), 127, name)
            self.assertEqual(tensors[name]["bits"], 8)

    def test_compiler_emits_full_fixed_graph(self) -> None:
        tensors = model.quantize_model(self.parameters)
        css = model.generate_css(tensors, "test-hash", 1, 42, 2.5)
        self.assertIn("signed 4-bit", css)
        self.assertIn("--y-0", css)
        self.assertIn(f"--y-{model.GENERATED_STEPS - 1}", css)
        self.assertGreaterEqual(css.count("@property"), 20_000)
        self.assertNotIn("--css-rnn-", css)

    def test_forward_loss_has_finite_gradient(self) -> None:
        inputs = self.rng.integers(0, model.VOCAB_SIZE, size=(4, 16), dtype=np.int64)
        targets = self.rng.integers(0, model.VOCAB_SIZE, size=(4, 16), dtype=np.int64)
        loss, gradients = model.loss_and_gradients(self.parameters, inputs, targets)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))


if __name__ == "__main__":
    unittest.main()
