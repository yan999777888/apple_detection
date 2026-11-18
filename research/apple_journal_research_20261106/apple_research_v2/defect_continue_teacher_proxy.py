"""Build held-fold, single-view 768-pixel evidence for proposal co-training on CPU."""

from pathlib import Path
from class0_patch_refinement import ROOT, OLD, configure, read, sha, atomic_json, load_image, manifest

OUT = ROOT / "experiments/defect_continue_teacher"


def main():
    configure(OUT, "cpu")
    import cv2
    import numpy as np
    import torch
    from defect_algorithm_modes_20261112 import nms

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    lookup = manifest()
    for fold in range(3):
        run = ROOT / f"runs/defect_oof3_fold{fold}_e40_20261112_v1"
        checkpoint = run / "weights/epoch_030.pt"
        source = OLD / f"all_mode_OOF_calibration/fold{fold}_sources.json"
        original = read(source)
        assert sha(run / "weights/last.pt") == original["checkpoint_sha256"]
        ids = {r["image_id"] for r in original["rows"]}
        audited = lambda part: {Path(s).name for s in (run / f"{part}_audited_images.txt").read_text("utf-8-sig").splitlines()}
        assert ids == audited("val") and not ids.intersection(audited("train"))
        assert all(lookup[s]["partition"] == "train" for s in ids)
        evidence = OUT / f"proxy_fold{fold}.json"
        if evidence.exists():
            previous = read(evidence)
            assert previous["weight_sha256"] == sha(checkpoint) and set(previous["pred"]) == ids
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = (state.get("ema") or state["model"]).float().eval().fuse(verbose=False)
        records = {}
        with torch.inference_mode():
            for i, sid in enumerate(sorted(ids)):
                im = load_image(lookup[sid]["rgb_path"])
                assert im.shape[:2] == (1024, 1024)
                scaled = cv2.resize(im, (768, 768), interpolation=cv2.INTER_LINEAR)
                x = torch.from_numpy(np.ascontiguousarray(scaled[:, :, ::-1])).permute(2,0,1)[None].float().div_(255)
                output = model(x)
                p = output[0][0] if isinstance(output, tuple) else output[0]
                assert p.ndim == 2 and p.shape[1] == 6
                p = p[p[:, 4] >= 0.008].numpy()
                p[:, :4] = (p[:, :4] * (1024 / 768)).clip(0, 1024)
                records[sid] = nms(p).tolist()
                if (i+1) % 60 == 0:
                    print(f"fold {fold} {i+1}/{len(ids)}", flush=True)
        atomic_json(evidence, dict(weight_sha256=sha(checkpoint), native_sha256=original["checkpoint_sha256"],
                                   source_sha256=sha(source), method="held-fold nano epoch30 at 768; proxy, not YOLO26s", pred=records))
        print(f"fold {fold} complete", flush=True)


if __name__ == "__main__":
    main()
