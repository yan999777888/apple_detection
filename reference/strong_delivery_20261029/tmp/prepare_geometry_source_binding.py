"""Bind one committed mask-geometry source17 inventory before GPU preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e059c9a590d8732962b9f948cd9e6d935e907004'
CONFIG = ROOT / 'configs/od_mask_geometry_papple_3080.json'
CONTROL = ROOT / 'configs/od_mask_geometry_control_3080.json'


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
    assert len(sources) == 17, f'geometry branch must pin 17 local sources, got {len(sources)}'
    assert {'od_context.py', 'od_foreground.py', 'od_context_model.py',
            'od_mask_geometry.py', 'od_mask_geometry_model.py'} <= set(sources)
    # Every consumed file must equal the committed blob exactly; CRLF drift fails here.
    for name, digest in sources.items():
        blob = git('show', f'{COMMIT}:src/apple_seg/{name}')
        assert hashlib.sha256(blob).hexdigest() == digest, name
        assert sha(ROOT / 'src/apple_seg' / name) == digest, name
    for path in (CONFIG, CONTROL):
        relative = path.relative_to(ROOT).as_posix()
        assert hashlib.sha256(git('show', f'{COMMIT}:{relative}')).hexdigest() == sha(path), relative
    observed_head = git('rev-parse', 'HEAD').decode().strip()
    tracked = git('status', '--porcelain', '--untracked-files=no').decode()
    assert not tracked.strip(), 'Source binding must be produced from a clean tracked tree'
    report = dict(schema='od_mask_geometry_committed_source_binding_v1',
                  commit=COMMIT, observed_documentation_HEAD=observed_head,
                  tracked_files_clean=True, source17_count=len(sources),
                  source17_sha256=dict(sorted(sources.items())),
                  config_sha256=sha(CONFIG), control_config_sha256=sha(CONTROL),
                  parent_sha256='9b44a9e7416fe13b9788f813161139094217bdbfc4cd62ec9d23ec719e3cb479',
                  scope='Read-only commit blobs; hashes local source only, never weights, VAL or GPU state.')
    path = ROOT / 'reports/od_mask_geometry_committed_source_binding.json'
    assert not path.exists(), 'Source binding is written once per reviewed commit'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(binding=str(path), sha256=sha(path), commit=COMMIT,
                          observed_HEAD=observed_head, sources=len(sources))))


if __name__ == '__main__':
    main()
