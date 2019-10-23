import os
import sys
import glob
import json
import subprocess
import time
import shutil

from s2base2.list_utils import make_chunks


BEAKER_TEMPLATE = """
description: BERT_DDI eval {} (try5 model)
tasks:
- spec:
    image: oyvindt/allennlpDdiV3
    resultPath: /output
    args:
    - python
    - -m
    - allennlp.run
    - evaluate_custom
    - --include-package
    - bert_ddi
    - --evaluation-data-file
    - /dataset/data.jsonl
    - --metadata-fields
    - id,sentence,label,logits,probs
    - --output-file
    - /output/eval-output.jsonl
    - --output-metrics-file
    - /output/metrics.json
    - --cuda-device
    - "0"
    - /model/model.tar.gz
    datasetMounts:
    - datasetId: {}
      containerPath: /dataset/data.jsonl
    - datasetId: {}
      containerPath: /model
    requirements:
      gpuCount: 1
      preemptible: true
"""

LOG_FILE = 'config/log.json'
BEAKER_DIR = 'beaker/'
CHUNK_SIZE = 1000000


if __name__ == '__main__':
    # read preprocessing log file
    with open(LOG_FILE, 'r') as f:
        log_dict = json.load(f)

    supp_sents_dir = log_dict['supp_sents_dir']
    ddi_output_dir = log_dict['ddi_output_dir']
    header_str = log_dict['header_str']
    bert_ddi_model = log_dict['bert_ddi_model']

    # get sentences
    try:
        supp_sents_file = glob.glob(os.path.join(supp_sents_dir, '*.jsonl'))[0]
    except IndexError:
        print('No sentences! Exiting!')
        sys.exit(1)

    # split file
    all_sents = []
    with open(supp_sents_file, 'r') as f:
        for line in f:
            all_sents.append(line)
    num_files = max(len(all_sents) // CHUNK_SIZE, 1)

    # write to batch files
    dataset_files = []
    for i, sents in enumerate(make_chunks(all_sents, num_files)):
        out_file = os.path.join(supp_sents_dir, f'supp_sents.jsonl.{i:02d}')
        with open(out_file, 'w') as outf:
            for line in sents:
                outf.write(line)
        dataset_files.append(out_file)

    # upload each dataset to beaker
    ds_ids = []
    for ds_file in dataset_files:
        print(f'Creating dataset: {ds_file}')
        beaker_args = ['beaker', 'dataset', 'create', ds_file]
        ds_output = subprocess.check_output(' '.join(beaker_args), stderr=subprocess.STDOUT, shell=True)
        ds_sents = [s.strip() for s in ds_output.decode('utf-8').split('\n')]
        ds_identifier = ds_sents[1].split()[-1]
        if ds_identifier.startswith('ds_'):
            ds_ids.append(ds_identifier)
        else:
            ds_ids.append(None)

    # create beaker templates
    exp_ids = []
    for i, (ds_file, ds_id) in enumerate(zip(dataset_files, ds_ids)):
        if not ds_id:
            print('No dataset identifier! Skipping!')
            continue
        print(f'Starting experiment: {ds_id}')
        exp_header = f'supp_ai_exp_{i:02d}'
        yaml_file = os.path.join(BEAKER_DIR, f'{exp_header}.yaml')
        with open(yaml_file, 'w') as yaml_f:
            yaml_f.write(BEAKER_TEMPLATE.format(exp_header, ds_id, bert_ddi_model))
        beaker_args = ['beaker', 'experiment', 'create', '-f', yaml_file]
        exp_output = subprocess.check_output(' '.join(beaker_args), stderr=subprocess.STDOUT, shell=True)
        exp_sents = [s.strip() for s in exp_output.decode('utf-8').split('\n')]
        exp_identifier = exp_sents[1].split()[1]
        if exp_identifier.startswith('ex_'):
            exp_ids.append(exp_identifier)
        else:
            exp_ids.append(None)

    # monitor experiments until done
    start_time = time.time()
    while True:
        now_time = time.time() - start_time
        print(f"Elapsed time: {now_time}", end="")
        all_done = True
        output_datasets = dict()
        for i, exp_id in enumerate(exp_ids):
            if not exp_id:
                continue
            beaker_args = ['beaker', 'experiment', 'inspect', exp_id]
            exp_output = subprocess.check_output(' '.join(beaker_args), stderr=subprocess.STDOUT, shell=True)
            exp_info = json.loads(exp_output.decode('utf-8'))[0]
            status_str = exp_info['nodes'][0]['status']
            if status_str == 'succeeded':
                output_datasets[exp_id] = exp_info['nodes'][0]['resultId']
            else:
                all_done = False
                break
        # check if all experiments done
        if all_done:
            print()
            break
        else:
            # wait 15 min and try all experiments again
            time.sleep(600)

    # download all output datasets
    for i, (exp_id, ds_id) in enumerate(output_datasets.items()):
        output_file = os.path.join(ddi_output_dir, f'output_part{i:02d}')
        beaker_args = ['beaker', 'dataset', 'fetch', '--output={}'.format(output_file), ds_id]
        ds_output = subprocess.run(beaker_args)

    # aggregate outputs into one file
    with open(os.path.join(ddi_output_dir, f'supp_labels_{header_str}.jsonl'), 'wb') as wfd:
        for f in glob.glob(os.path.join(ddi_output_dir, 'output_part*/' 'eval-output.jsonl')):
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            os.remove(f)

    # write beaker dataset ids to log file
    log_dict["bert_ddi_model"] = bert_ddi_model
    log_dict["beaker_dataset_ids"] = ds_ids

    with open(LOG_FILE, 'w') as f:
        json.dump(log_dict, f, indent=4)

    print('done.')