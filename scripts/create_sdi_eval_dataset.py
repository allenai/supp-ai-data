"""
Write SDI test set from evaluation dataset
"""

import csv
import ast
import json

LABELED_SDI_DATA = 'eval/sdi_eval_set.tsv'
SDI_TEST_SET = 'eval/sdi_test.jsonl'

sdi_data = []
uid_ind = 0
with open(LABELED_SDI_DATA, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headers = next(reader)
    for _, gold_label, _, _, _, _, _, _, _, paper_id, sentence, span1, arg1, span2, arg2 in reader:
        span1_inds = ast.literal_eval(span1)
        span2_inds = ast.literal_eval(span2)
        sdi_data.append({
            'id': f'{paper_id}-{uid_ind}',
            'sentence_id': 0,
            'sentence': sentence,
            'arg1': {
                "span": span1_inds if len(span1_inds) == 1 else [span1_inds],
                "string": arg1,
                "umls_types": [],
                "id": ""
            },
            'arg2': {
                "span": span2_inds if len(span2_inds) == 1 else [span2_inds],
                "string": arg2,
                "umls_types": [],
                "id": ""
            },
            "label": gold_label
        })
        uid_ind += 1

with open(SDI_TEST_SET, 'w') as outf:
    for entry in sdi_data:
        json.dump(entry, outf)
        outf.write('\n')