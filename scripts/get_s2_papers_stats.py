import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict, Counter

total_papers = 0
years = []
s2_fos = []
external_ids = defaultdict(list)
for i in range(0, 1):
    data_file = os.path.join('temp', f'papers-part{i}.jsonl.gz')
    with gzip.open(data_file, 'rt') as f:
        for line in tqdm(f):
            entry = json.loads(line)
            total_papers += 1
            years.append(entry['year'])
            if entry['s2fieldsofstudy']:
                fos = tuple(set([s2fos['category'] for s2fos in entry['s2fieldsofstudy']]))
            else:
                fos = tuple([])
            s2_fos.append(fos)
            for ext_id, ext_val in entry['externalids'].items():
                if ext_val:
                    external_ids[ext_id].append(ext_val)

print('Total papers:', total_papers)

print('External IDs:')
for ext_id, ext_vals in external_ids.items():
    print(ext_id, len(ext_vals))

print('Years:')
print(Counter(years))

print('S2 FOS:')
print(Counter(s2_fos))