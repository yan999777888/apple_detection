"""Regression check for the Faster R-CNN epoch writer and exact optimizer continuation."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apple_research_v2"))
from defect_continue_rcnn import OUT, save_epoch, restore_epoch, load_checkpoint


class EpochRoundTrip(unittest.TestCase):
    def test_next_step(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            folder = Path(tmp)
            torch.manual_seed(481)
            rng = np.random.default_rng(793)
            m = torch.nn.Linear(4, 2)
            opt = torch.optim.SGD(m.parameters(), lr=.01, momentum=.9)

            def step():
                opt.zero_grad()
                x = torch.rand(6, 4) + rng.uniform()
                m(x).square().mean().backward()
                opt.step()

            step()
            save_epoch(folder, 3, m, opt, rng, [{"epoch": 3}], .5)
            receipt = next(folder.glob("epoch??.sha256.json"))
            state = load_checkpoint(receipt.with_suffix("").with_suffix(".pt"))
            step()
            expected = {k:v.clone() for k,v in m.state_dict().items()}
            m = torch.nn.Linear(4, 2)
            opt = torch.optim.SGD(m.parameters(), lr=1)
            ep, history = restore_epoch(state,m,opt,rng,.5)
            self.assertEqual((ep,history),(3,[{"epoch":3}]))
            step()
            for key, value in m.state_dict().items():
                self.assertTrue(torch.equal(expected[key],value))


if __name__ == "__main__":
    unittest.main()
