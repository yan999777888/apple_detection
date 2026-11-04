"""Bind the committed source18 inventory and all three reviewed arm configs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'f79f494533177c53da1d0872a399f9738e591462'
CONFIGS = ('od_mask_geometry_papple_3080.json', 'od_mask_geometry_control_3080.json',
           'od_mask_geometry_verifier_papple_3080.json',
           'od_mask_geometry_verifier_pilot_3080.json',
           'od_mask_geometry_verifier_papple_joint8_3080.json',
           'od_mask_geometry_verifier_pilot2_3080.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*arguments):
    return subprocess.run(['git', *arguments], cwd=ROOT, capture_output=True, check=True).stdout


def main():
    assert git('cat-file', '-t', COMMIT).decode().strip() == 'commit'
    sys.path.insert(0, str(ROOT / 'src'))
    from apple_seg import od_training as training
    sources = training.source_snapshot()
    assert len(sources) == 18, f'verifier branch must pin 18 local sources, got {len(sources)}'
    assert {'od_context.py', 'od_foreground.py', 'od_context_model.py', 'od_mask_geometry.py',
            'od_mask_geometry_model.py', 'od_verifier.py'} <= set(sources)
    for name, digest in sources.items():
        blob = git('show', f'{COMMIT}:src/apple_seg/{name}')
        assert hashlib.sha256(blob).hexdigest() == digest, name
        assert sha(ROOT / 'src/apple_seg' / name) == digest, name
    configs = {}
    for name in CONFIGS:
        path = ROOT / 'configs' / name
        relative = path.relative_to(ROOT).as_posix()
        assert hashlib.sha256(git('show', f'{COMMIT}:{relative}')).hexdigest() == sha(path), name
        configs[name] = sha(path)
    observed_head = git('rev-parse', 'HEAD').decode().strip()
    assert not git('status', '--porcelain', '--untracked-files=no').decode().strip(), \
        'Source binding must be produced from a clean tracked tree'
    report = dict(schema='od_mask_geometry_verifier_source_binding_v4',
                  commit=COMMIT, observed_documentation_HEAD=observed_head,
                  tracked_files_clean=True, source18_count=len(sources),
                  source18_sha256=dict(sorted(sources.items())),
                  config_sha256=dict(sorted(configs.items())),
                  parent_sha256='9b44a9e7416fe13b9788f813161139094217bdbfc4cd62ec9d23ec719e3cb479',
                  scope='Read-only commit blobs; hashes local source and configs only, never weights, VAL or GPU state.')
    path = ROOT / 'reports/od_mask_geometry_verifier_source_binding_v4.json'
    assert not path.exists(), 'Source binding is written once per reviewed commit'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(binding=str(path), sha256=sha(path), commit=COMMIT,
                          observed_HEAD=observed_head, sources=len(sources), configs=configs)))


if __name__ == '__main__':
    main()
