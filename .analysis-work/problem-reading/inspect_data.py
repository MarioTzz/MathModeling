"""Read-only inventory of the supplied MAT files; no feature engineering or modeling."""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.analysis-work/python-deps'))
import numpy as np
import scipy
from scipy.io import loadmat

records = []
for path in sorted((ROOT / '数据集').rglob('*.mat')):
    relative = path.relative_to(ROOT)
    source = relative.parts[1] == '源域数据集'
    group = relative.parts[2] if source else '目标域'
    state = ('N' if group == '48kHz_Normal_data' else relative.parts[3]) if source else None
    values = loadmat(path)
    fields = {}
    for name, value in values.items():
        if name.startswith('__'):
            continue
        info = {'shape': list(value.shape), 'dtype': str(value.dtype), 'size': int(value.size)}
        if np.issubdtype(value.dtype, np.number):
            info.update(nonfinite=int(np.count_nonzero(~np.isfinite(value))),
                        min=float(np.min(value)) if value.size else None,
                        max=float(np.max(value)) if value.size else None)
            if value.size <= 8:
                info['values'] = value.flatten().tolist()
        fields[name] = info
    records.append({'path': str(relative), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'domain': 'source' if source else 'target', 'group': group, 'class': state,
                    'fields': fields})

source = [r for r in records if r['domain'] == 'source']
target = [r for r in records if r['domain'] == 'target']
groups = {}
for group in sorted({r['group'] for r in source}):
    rows = [r for r in source if r['group'] == group]
    groups[group] = dict(collections.Counter(r['class'] for r in rows))
channels = collections.Counter()
patterns = collections.Counter()
rpm_values, missing_rpm, unusual_fields = [], [], []
for row in source:
    names = row['fields']
    pattern = []
    for channel in ['DE', 'FE', 'BA']:
        if any('_' + channel + '_time' in name for name in names):
            channels[channel] += 1
            pattern.append(channel)
    patterns['+'.join(pattern)] += 1
    rpm = [v['values'] for name, v in names.items() if 'RPM' in name.upper() and 'values' in v]
    if rpm:
        rpm_values.extend(x for arr in rpm for x in arr)
    else:
        missing_rpm.append(row['path'])
    extra = [name for name in names if not any('_' + ch + '_time' in name for ch in ['DE','FE','BA']) and 'RPM' not in name.upper()]
    if extra:
        unusual_fields.append({'path': row['path'], 'fields': extra})
hash_groups = collections.defaultdict(list)
for row in records:
    hash_groups[row['sha256']].append(row['path'])
summary = {
    'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
    'source_count': len(source), 'target_count': len(target),
    'source_class_counts': dict(collections.Counter(r['class'] for r in source)),
    'source_groups': groups, 'source_channel_counts': dict(channels),
    'source_channel_patterns': dict(patterns),
    'source_rpm_range_in_fields': [min(rpm_values), max(rpm_values)],
    'source_missing_rpm': missing_rpm, 'source_extra_fields': unusual_fields,
    'nonfinite_count': sum(f.get('nonfinite',0) for r in records for f in r['fields'].values()),
    'duplicate_file_groups': [v for v in hash_groups.values() if len(v)>1],
    'target_fields': [{ 'path': r['path'], 'fields': r['fields']} for r in target],
}
out = ROOT / '.analysis-work/problem-reading'
(out / 'data-inventory.json').write_text(json.dumps({'summary': summary, 'files': records}, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
