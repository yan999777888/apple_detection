import io
p = 'tmp/stage2_cli.py'
s = io.open(p, encoding='utf-8').read()

old_head = """def cmd_eval(args):
    directory = Path(args.predictions)
    if not directory.is_dir():
        directory = ROOT / args.predictions
    if not (directory / 'raw_predictions.json').is_file():
        # Accept a run directory as well as one validation update directory.
        directory = next((directory / 'validation').glob(f'update_{args.update}_*'))"""
new_head = """def cmd_eval(args):
    directory = Path(args.predictions)
    if not directory.is_absolute():
        directory = (ROOT / directory)
    directory = directory.resolve()
    if not (directory / 'raw_predictions.json').is_file():
        # Accept a run directory as well as one validation update directory.
        directory = next((directory / 'validation').glob(f'update_{args.update}_*')).resolve()"""
assert old_head in s
s = s.replace(old_head, new_head, 1)

# Move the score dump so a later formatting bug can never throw the 12-minute
# rescoring away again.
dump = """    RUNS.mkdir(parents=True, exist_ok=True)
    # Persist the per-box scores so every later analysis is instant and can
    # never accidentally retrain on VAL.
    scores_path = RUNS / f"{Path(args.checkpoint).stem}_{directory.name[:16]}_scores.json"
    scores_path.write_text(json.dumps({record['image']: dict(
        boxes=[[round(float(v), 2) for v in box] for box in record['boxes']],
        base=[round(float(v), 6) for v in record['base']],
        stage2=[round(float(v), 6) for v in record['stage2']],
        gt=[[round(float(v), 2) for v in box] for box in
            (np.asarray(record['gt']).reshape(-1, 4).tolist() if len(np.asarray(record['gt'])) else [])]
    ) for record in records}), encoding='utf-8')
    report['scores_path'] = str(scores_path.relative_to(ROOT))
"""
assert dump in s
s = s.replace(dump, "", 1)

anchor = """    print(f'rescore done in {time.time() - started:.0f}s', flush=True)
"""
assert anchor in s
s = s.replace(anchor, anchor + """
    RUNS.mkdir(parents=True, exist_ok=True)
    # Persist the per-box scores first: the rescoring pass is the expensive part
    # and every later analysis must be able to reuse it.
    scores_path = RUNS / f"{Path(args.checkpoint).stem}_{directory.name[:16]}_scores.json"
    scores_path.write_text(json.dumps({record['image']: dict(
        boxes=[[round(float(v), 2) for v in box] for box in record['boxes']],
        base=[round(float(v), 6) for v in record['base']],
        stage2=[round(float(v), 6) for v in record['stage2']],
        gt=[[round(float(v), 2) for v in box] for box in
            (np.asarray(record['gt']).reshape(-1, 4).tolist() if len(np.asarray(record['gt'])) else [])]
    ) for record in records}), encoding='utf-8')
    print('wrote', scores_path, flush=True)
""", 1)

old_report = """    report = dict(directory=str(directory.relative_to(ROOT)), checkpoint=str(args.checkpoint),"""
new_report = """    report = dict(directory=directory.name, checkpoint=str(args.checkpoint),
                  scores_path=str(scores_path),"""
assert old_report in s
s = s.replace(old_report, new_report, 1)
s = s.replace("""    report['scores_path'] = str(scores_path.relative_to(ROOT))
    out = RUNS""", """    out = RUNS""", 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('eval made crash-safe')
