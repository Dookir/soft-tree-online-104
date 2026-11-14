"""The public solver's stdin/stdout contract, without ML dependencies."""
import hashlib
import json
import math
from pathlib import Path


def matrix(value):
    if not isinstance(value, list):
        raise ValueError('A weight group must be a JSON array')
    if len(value) == 3 and all(isinstance(r, list) and len(r) == 7 for r in value):
        value = [x for row in value for x in row]
    if len(value) != 21 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) or x < 0 for x in value):
        raise ValueError('Each group requires 21 finite nonnegative numbers')
    rows = [[float(x) for x in value[i:i+7]] for i in range(0, 21, 7)]
    if any(max(row) <= 0 for row in rows):
        raise ValueError('Each building must have a positive weight')
    return rows


def parse_request(value):
    if isinstance(value, dict):
        if set(value) != {'weights'}:
            raise ValueError('Expected an object containing only weights')
        value = value['weights']
    try:
        return [matrix(value)]
    except (ValueError, TypeError, OverflowError):
        pass
    if not isinstance(value, list) or not 1 <= len(value) <= 50:
        raise ValueError('Each call accepts 1 to 50 weight groups')
    return [matrix(group) for group in value]


def parse_response(value, count):
    if not isinstance(value, dict) or set(value) != {'R'}:
        raise ValueError('Expected exactly {"R": [...]}')
    scores = value['R']
    if not isinstance(scores, list) or len(scores) != count:
        raise ValueError('Returned score count does not match input')
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or x < 0 for x in scores):
        raise ValueError('Scores must be finite nonnegative numbers')
    return [float(x) for x in scores]


def weight_key(value):
    rows = matrix(value)
    normalized = []
    for row in rows:
        scale = max(row)
        row = [x / scale for x in row]
        total = sum(row)
        normalized.extend(round(x / total, 12) for x in row)
    return hashlib.sha256(json.dumps(normalized, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
