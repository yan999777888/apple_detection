"""Adapt a frozen preparation manifest to the attribute trainer records schema.

No labels, bounds, groups, splits, image paths, or source files are modified.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, target = args.source.resolve(), args.output.resolve()
    if target.exists():
        p.error('Output exists; refusing to overwrite')
    original_sha = sha(source)
    document = json.loads(source.read_text(encoding='utf-8'))
    records = document.pop('samples')
    if not records or any(r['split'] not in ('train', 'val') for r in records):
        raise ValueError('Expected nonempty train/val records only')
    for row in records:
        annotation = source.parent/'annotations'/(row['split']+'.json')
        if sha(annotation) != row['annotation_sha256']:
            raise ValueError('Original annotation SHA256 mismatch')
        row['annotation_path'] = str(annotation.resolve())
    document['records'] = records
    document['schema'] = 'apple_attribute_records_manifest_v1'
    document['split_policy']['independent_group_holdout'] = False
    document['source_preparation_manifest'] = str(source)
    document['source_preparation_manifest_sha256'] = original_sha
    document['adapter_source_sha256'] = sha(__file__)
    document['adaptation'] = 'samples renamed to records; original annotation path added and SHA256 verified; independent_group_holdout explicitly false; all source records otherwise preserved'
    with target.open('x', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    if sha(source) != original_sha:
        raise RuntimeError('Source manifest changed')
    print(json.dumps({'output':str(target),'sha256':sha(target),'source_sha256':original_sha,'records':len(records),'class_names':document['class_names']},ensure_ascii=False))


if __name__ == '__main__':
    main()
