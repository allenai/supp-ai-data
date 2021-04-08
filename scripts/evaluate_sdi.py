import os
import json


SDI_TEST_DATA = 'eval/sdi_test.jsonl'
ROBERTA_EVAL_RESULTS = 'eval/bert_ddi_sdi_eval/eval-output.jsonl'

gold_labels = dict()
with open(SDI_TEST_DATA, 'r') as f:
    for line in f:
        entry = json.loads(line)
        gold_labels[entry['id']] = int(entry['label'])

model_labels = dict()
with open(ROBERTA_EVAL_RESULTS, 'r') as f:
    for line in f:
        entry = json.loads(line)
        model_labels[entry['id']] = int(entry['label-model'])

total = 0
tp = 0
fp = 0
tn = 0
fn = 0

for k in gold_labels:
    total += 1
    gold = gold_labels[k]
    model = model_labels[k]
    if gold == 1 and model == 1:
        tp += 1
    elif gold == 0 and model == 1:
        fp += 1
    elif gold == 1 and model == 0:
        fn += 1
    elif gold == 0 and model == 0:
        tn += 1
    else:
        raise Exception

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / total
f1_score = 2. * precision * recall / (precision + recall)

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'Accuracy: {accuracy:.3f}')
print(f'F1-score: {f1_score:.3f}')