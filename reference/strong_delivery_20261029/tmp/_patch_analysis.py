import io
p = 'tmp/stage2_analysis.py'
s = io.open(p, encoding='utf-8').read()
old = """    pool = np.concatenate([r['base'] for r in records]) if records else np.zeros(1)
    thresholds = np.unique(np.quantile(pool, np.linspace(.55, .99995, 800)))"""
assert old in s
new = (
    "    # Every arm gets its own global threshold sweep on the full VAL, which is\n"
    "    # the protocol this project has always used; no arm is handicapped by\n"
    "    # another arm's score scale.\n"
    "    def grid(key):\n"
    "        pool = np.concatenate([np.asarray(r[key]) for r in records]) if records else np.zeros(1)\n"
    "        return np.unique(np.quantile(pool, np.linspace(.3, .99995, 900)))\n"
    "\n"
    "    thresholds = {key: grid(key) for key in ('base', 'stage2', 'fused')}"
)
s = s.replace(old, new, 1)
s = s.replace("        report[key] = sweep(records, key, thresholds)",
              "        report[key] = sweep(records, key, thresholds[key])", 1)
s = s.replace("    report['fused'] = sweep(fused, 'fused', thresholds)",
              "    report['fused'] = sweep(fused, 'fused', thresholds['fused'])", 1)
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('per-arm thresholds patched')
