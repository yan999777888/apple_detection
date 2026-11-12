"""Pure metadata and CPU loss tests; toy logits never become research data."""
import importlib.util
import math
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "apple_research_v2/train_attribute_pilot_v2.py"
SPEC = importlib.util.spec_from_file_location("attribute_pilot_v2", SCRIPT)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class ClassWeightContract(unittest.TestCase):
    def test_manifest_count_formula_and_order(self):
        names = ["immature apple", "semi-mature apple", "mature apple"]
        counts = {names[0]: 711, names[1]: 276, names[2]: 545}
        weights = PILOT.training_class_weights(names, counts, "inverse-frequency")
        self.assertEqual(weights, [1532 / (3 * 711), 1532 / (3 * 276), 1532 / (3 * 545)])
        self.assertAlmostEqual(sum(counts[name] * weight for name, weight in zip(names, weights)), 1532)
        reverse = PILOT.training_class_weights(names[::-1], counts, "inverse-frequency")
        self.assertEqual(reverse, weights[::-1])

    def test_none_preserves_original_unweighted_criterion(self):
        self.assertIsNone(PILOT.training_class_weights(["a", "b"], {"a": 5, "b": 2}, "none"))

    def test_missing_zero_or_noninteger_counts_refused(self):
        for counts in ({"a": 2}, {"a": 2, "b": 0}, {"a": 2, "b": .5}):
            with self.assertRaises(ValueError):
                PILOT.training_class_weights(["a", "b"], counts, "inverse-frequency")

    def test_unknown_mode_refused(self):
        with self.assertRaises(ValueError):
            PILOT.training_class_weights(["a", "b"], {"a": 2, "b": 3}, "automatic_search")

    def test_defaults_preserve_fixed_initial_budget(self):
        args = PILOT.parser().parse_args(["--manifest", "metadata.json", "--output", "new-output"])
        self.assertEqual((args.class_weighting, args.max_steps, args.batch_size, args.seed, args.learning_rate),
                         ("none", 32, 4, 20261106, .001))

    def test_probe_CE_remains_unweighted_despite_weighted_training(self):
        import torch
        initialized_before = torch.cuda.is_initialized()
        logits = torch.tensor([[2., 0., 0.], [0., 0., 0.], [0., 2., 0.]], dtype=torch.float64)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        per_sample = PILOT.unweighted_probe_cross_entropy(logits, targets)
        expected = [math.log(math.exp(2) + 2) - 2, math.log(3), math.log(math.exp(2) + 2)]
        for actual, value in zip(per_sample.tolist(), expected):
            self.assertAlmostEqual(actual, value, places=12)
        weights = torch.tensor([1., 3., 2.], dtype=torch.float64)
        training_loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, targets)
        self.assertAlmostEqual(training_loss.item(), sum(v*w for v,w in zip(expected, [1,3,2])) / 6, places=12)
        self.assertNotAlmostEqual(training_loss.item(), per_sample.mean().item(), places=6)
        self.assertEqual(torch.cuda.is_initialized(), initialized_before)


if __name__ == "__main__":
    unittest.main()
