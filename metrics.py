"""Pair protocols and threshold calibration without test-label leakage."""
from pathlib import Path
import numpy as np


def read_pairs(filename, root):
    root = Path(root)
    pairs = []
    header = None
    for lineno, line in enumerate(Path(filename).read_text().splitlines(), 1):
        words = line.strip().split()
        if not words or words[0].startswith('#'):
            continue
        if not pairs and header is None and len(words) <= 2 and all(w.isdigit() for w in words):
            header = [int(w) for w in words]
            continue
        try:
            if len(words) == 3 and words[1].isdigit() and words[2].isdigit():
                person, first, second = words
                a = root / person / f'{person}_{int(first):04d}.jpg'
                b = root / person / f'{person}_{int(second):04d}.jpg'
                label = 1
            elif len(words) == 4:
                p1, i1, p2, i2 = words
                a = root / p1 / f'{p1}_{int(i1):04d}.jpg'
                b = root / p2 / f'{p2}_{int(i2):04d}.jpg'
                label = 0
            elif len(words) == 3 and words[2] in {'0', '1'}:
                a, b, label = root / words[0], root / words[1], int(words[2])
            else:
                raise ValueError('expected LFW protocol or path1 path2 label')
        except ValueError as error:
            raise ValueError(f'{filename}:{lineno}: {error}') from error
        for path in (a, b):
            if not path.is_file():
                raise FileNotFoundError(f'{filename}:{lineno}: {path}')
        pairs.append((str(a.resolve()), str(b.resolve()), label))
    if not pairs:
        raise ValueError('配对列表为空。')
    if header and len(pairs) != 2 * int(np.prod(header)):
        raise ValueError(f'配对列表头部声明数量与实际数量不一致: {filename}')
    return pairs, header


def validate_scores(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError('scores and labels must be nonempty vectors of equal length')
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('nonfinite scores or invalid labels')
    if len(np.unique(labels)) != 2:
        raise ValueError('both positive and negative pairs are required')
    return scores, labels


def threshold_search(scores, labels):
    """Maximize balanced accuracy; include reject-all and group ties atomically."""
    scores, labels = validate_scores(scores, labels)
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, np.cumsum(1-y)[ends]]
    accuracy = 0.5 * (tp / y.sum() + 1 - fp / (1-y).sum())
    thresholds = np.r_[np.nextafter(s[0], np.inf), s[ends]]
    best = int(np.argmax(accuracy))
    return float(accuracy[best]), float(thresholds[best])


def roc_curve(scores, labels):
    scores, labels = validate_scores(scores, labels)
    order = np.argsort(-scores, kind='stable')
    s, y = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    return np.r_[0., np.cumsum(1-y)[ends] / (1-y).sum()], np.r_[0., np.cumsum(y)[ends] / y.sum()]


def summarize(scores, labels, threshold):
    scores, labels = validate_scores(scores, labels)
    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    fpr, tpr = roc_curve(scores, labels)
    # Linear interpolation at the ROC crossing gives a descriptive EER.
    delta = fpr + tpr - 1
    crossing = int(np.searchsorted(delta, 0))
    left, right = max(0, crossing-1), min(crossing, len(delta)-1)
    fraction = -delta[left] / (delta[right]-delta[left]) if delta[right] != delta[left] else 0.
    eer = float(fpr[left] + fraction * (fpr[right]-fpr[left]))
    return {'pairs': len(labels), 'threshold': float(threshold),
            'accuracy': float(np.mean(predicted == positive)),
            'balanced_accuracy': 0.5 * (tp/(tp+fn) + tn/(tn+fp)),
            'tar': tp/(tp+fn), 'far': fp/(tn+fp), 'frr': fn/(tp+fn),
            'auc': float(np.trapezoid(tpr, fpr)), 'eer': eer,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def pair_scores(features, paths, pairs):
    indexes = {path: i for i, path in enumerate(paths)}
    a = np.array([indexes[p[0]] for p in pairs])
    b = np.array([indexes[p[1]] for p in pairs])
    return np.clip(np.sum(features[a] * features[b], axis=1), -1, 1), np.array([p[2] for p in pairs])
