"""Count annotated locations a bounded score-only model could possibly recover."""

from collections import Counter

from scipy.special import expit, logit

from class0_patch_refinement import ROOT, read, atomic_json

OUT = ROOT/'experiments/defect_review_impact_v1'


def analyze(budget=2.0):
    replay = read(OUT/'training_point_behavior.json')
    thresh = dict(scratch=0.40,insect=0.34)
    floors = {c: float(expit(logit(t)-budget)) for c,t in thresh.items()}
    counts = Counter()
    for row in replay['records']:
        cls = row['response']
        if row['source'] != 'unmatched OOF candidate' or cls not in thresh:
            continue
        scores = [r['score'] for r in row['models']['specialist']['at_point'] if r['cls'] == cls]
        if not scores:
            state = 'no_proposal_at_point'
        elif max(scores) >= thresh[cls]:
            state = 'already_at_workpoint'
        elif max(scores) >= floors[cls]:
            state = 'inside_score_headroom'
        else:
            state = 'cannot_cross_workpoint'
        counts[f'{cls}/{state}'] += 1
    return dict(theoretical_only=True, logit_shift_bound=budget, fixed_thresholds=thresh,
                sigmoid_logit_threshold_minus_bound=floors,
                scope='Previously unannotated training points marked as lesions by reader. '
                'Any candidate in that class containing the identified center; no localization GT.',
                form='When adjusted_logit = old_logit + delta and delta <= bound, '
                'old_score < sigmoid(logit(threshold)-bound) cannot pass threshold.',
                model=replay['specialist_sha'], export_sha=replay['export_sha'], counts=counts)


if __name__ == '__main__':
    report = analyze()
    atomic_json(OUT/'bounded_rescore_training_ceiling.json',report)
    print(report['counts'])
