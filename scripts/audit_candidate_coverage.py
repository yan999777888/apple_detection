"""CPU-only GT coverage of saved decoded candidates; never a detector score."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from audit_premask_candidates import REPO,POLICIES,sha,write
from deployment.candidate_policy import select_candidates
from apple_seg.evaluate import load_papple_contract
import flip_tta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,required=True)
    a=p.parse_args();root=a.audit_dir.resolve()
    if root.drive.upper()!='F:':raise ValueError('F: required')
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    if status['phase']!='complete' or status['mode']!='audit':raise ValueError('Completed original-pool audit required')
    cfg=json.loads((REPO/'configs/strong_baseline_4060.json').read_text())
    base=(REPO/'configs'/cfg['path_base']).resolve()
    contract=load_papple_contract(base/cfg['paths']['provenance'])
    report={'scope':'Decoded-box pool IoU>=.5 GT coverage, not final Recall; fixed diagnostic subset only',
        'audit_status_sha256':sha(root/'status.json'),'provenance_sha256':contract['provenance_sha256'],'models':{}}
    torch.set_num_threads(4)
    for model in status['models']:
        per_image=[];totals={name:Counter() for name in POLICIES}
        for sample in status['samples']:
            gt=contract['truth'][sample['image_id']]
            file=root/model/'none'/f"candidates_{sample['image_index']:04d}.npz"
            with np.load(file,allow_pickle=False) as data:
                boxes=torch.from_numpy(data['decoded_boxes'].copy())
                scores=torch.from_numpy(data['initial_scores'].copy())
            frozen=contract['images'][sample['image_index']]
            # This audit is explicitly the 1024-square full image baseline.
            native=boxes.numpy()*np.asarray([frozen['width']/1024,frozen['height']/1024]*2,dtype=np.float32)
            overlaps=flip_tta.metrics.box_iou(native,np.asarray([g['box'] for g in gt]).reshape(-1,4))
            covered=(overlaps>=.5).any(axis=0)
            ranks=torch.argsort(scores,descending=True,stable=True).numpy()
            first=[next((j+1 for j,k in enumerate(ranks) if overlaps[k,g]>=.5),None) for g in range(len(gt))]
            row={'image_id':sample['image_id'],'candidate_archive_sha256':sha(file),'first_cover_initial_rank':first,'policies':{}}
            for name,spec in POLICIES.items():
                indices=select_candidates(boxes,scores,**spec).numpy()
                kept=(overlaps[indices]>=.5).any(axis=0)
                lost=[{'gt_index':i,'apple_id':g['apple_id'],'stratum':g['occlusion_stratum']} for i,g in enumerate(gt) if covered[i] and not kept[i]]
                row['policies'][name]={'covered_GT':int(kept.sum()),'lost_covered_GT':lost}
                totals[name].update(GT=len(gt),covered_GT=int(kept.sum()),lost_covered_GT=len(lost))
                totals[name].update(Counter('lost_'+g['stratum'] for g in lost))
            per_image.append(row)
        report['models'][model]={'totals':{k:dict(v) for k,v in totals.items()},'images':per_image}
    write(root/'candidate_coverage.json',report)
    print(json.dumps({m:r['totals'] for m,r in report['models'].items()},ensure_ascii=False))


if __name__=='__main__':main()
