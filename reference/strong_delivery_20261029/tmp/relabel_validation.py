"""Relabel a validated checkpoint with the validation that actually matches it.

`od_step4_convnextbase_c2_11` was trained while EMA validation was still
persisting live weights: its `validations` entry describes the EMA shadow
(92.06) while the tensor set on disk scores 92.00. The weights are untouched
here; only the record is replaced by the re-run evaluation of those exact
weights, and the superseded record is kept verbatim for audit.

Nothing is invented: the replacement values are read from the existing
`revalidation_live` artifacts, which were produced by the trainer's own
`evaluate_od` protocol on the complete 814-image VAL split.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from apple_seg.od_training import save_checkpoint, sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revalidation", required=True,
                        help="Directory holding the matching summary.json/evaluation_metadata.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".",
                        help="Run root the validation directory is relative to")
    args = parser.parse_args()

    source = Path(args.checkpoint)
    expected = source.with_suffix(source.suffix + ".sha256").read_text(encoding="ascii").strip()
    if sha(source) != expected:
        raise SystemExit("Source checkpoint SHA256 differs from its sidecar")

    revalidation = Path(args.revalidation)
    summary = json.loads((revalidation / "summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((revalidation / "evaluation_metadata.json").read_text(encoding="utf-8"))
    if (summary.get("phase") != "complete" or summary.get("model_selection_eligible") is not True
            or metadata.get("official_full_val") is not True
            or (metadata.get("images"), metadata.get("GT_instances")) != (814, 3449)):
        raise SystemExit("Revalidation artifacts are not a complete original VAL")

    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    update = payload["updates"]
    matches = [point for point in payload["validations"] if point.get("update") == update]
    if len(matches) != 1:
        raise SystemExit("Expected exactly one validation record for the saved update")
    superseded = matches[0]

    relative = Path(args.root).resolve()
    directory = revalidation.resolve().relative_to(relative).as_posix()
    replacement = dict(update=update, stage=payload["stage"],
                       confidence=summary["confidence"], score=summary["score"],
                       summary=summary["summary"], directory=directory,
                       beats_parent=summary["score"] > superseded.get("beats_parent", 0))
    payload["validations"] = [replacement if point.get("update") == update else point
                              for point in payload["validations"]]
    payload["superseded_validation_records"] = [superseded]
    payload["validation_record_correction"] = {
        "reason": "The superseded record was produced by EMA weights that were never persisted; "
                  "the tensor set on disk is the live weight set.",
        "replacement_directory": directory,
        "weights_unchanged": True,
        "source_checkpoint_sha256": expected,
        "superseded_confidence": superseded.get("confidence"),
        "superseded_score": superseded.get("score"),
    }
    digest = save_checkpoint(Path(args.output), payload)
    print(json.dumps({"output": str(args.output), "sha256": digest,
                      "superseded": {"confidence": superseded.get("confidence"),
                                     "score": superseded.get("score")},
                      "replacement": {"confidence": replacement["confidence"],
                                      "score": replacement["score"],
                                      "directory": directory}}, indent=1))
    print("weights_unchanged", torch.equal(torch.load(source, map_location="cpu", weights_only=False,
                                                      mmap=True)["model"]["backbone.body.stages_0.downsample.0.weight"],
                                           payload["model"]["backbone.body.stages_0.downsample.0.weight"]))


if __name__ == "__main__":
    main()
