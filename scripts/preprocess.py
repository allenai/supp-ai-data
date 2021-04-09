"""
Download new articles from S2 corpus
Run NER and linking over all article abstracts

"""

import os
import json
import glob
import tqdm
import multiprocessing
from datetime import datetime
from typing import Dict
import itertools
import shutil
import logging

from suppai.data_getter import DataGetter
from suppai.ner_and_linker import DrugSupplementLinker
from suppai.cui_handler import CUIHandler

from s2base2.list_utils import make_chunks

logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)

logger = logging.getLogger("scispacy")
logger.setLevel(logging.ERROR)


def clean_ent(ent: Dict) -> Dict:
    """
    Get top cui from scispacy entity linking output
    Return entity dictionary
    :param ent:
    :return:
    """
    return {
        "span": [[ent['start'], ent['end']]],
        "string": ent['string'],
        "umls_types": ent['linked_cuis'][0][1],
        "id": ent['linked_cuis'][0][0]
    }


def batch_run_ner_linking(batch_dict: Dict):
    """
    Process one batch of paper files (files are jsonl with one paper per line)
    :param batch_dict:
    :return:
    """
    # set environmental variables for spacy multiprocessing
    # NOTE: currently also need to change scispacy/candidate_generation.py:158 to:
    # original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k, num_threads=1)
    # additional num_threads flag prevents scispacy from starting too many threads (keeps each process to 1 CPU)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    input_file = batch_dict["file_to_process"]
    entity_file = batch_dict["entity_file"]
    skipped_file = batch_dict["skipped_file"]

    # fire up scispacy linker
    ds_linker = DrugSupplementLinker()

    with open(input_file, 'r') as f, open(entity_file, 'w+') as outf, open(skipped_file, 'w+') as skip_f:
        # process input file
        for line in tqdm.tqdm(f):
            entry = json.loads(line)

            # skip if no abstract
            if not entry['abstract']:
                skip_f.write(f"{entry['corpus_id']}\n")
                continue

            try:
                ents_per_sentence = ds_linker.get_linked_entities(
                    entry['abstract'],
                    top_k=3
                )
            except Exception as e:
                skip_f.write(f"{entry['corpus_id']}\n")
                continue

            # iterate through sentences
            for sent in ents_per_sentence:
                if not sent["sentence"].strip():
                    continue
                entities = sent["entities"]
                if entities and len(entities) > 0:
                    output_dict = {
                        "id": entry["corpus_id"],
                        "sentence_id": sent["sent_num"],
                        "sentence": sent["sentence"].strip(),
                        "entities": entities
                    }
                    json.dump(output_dict, outf)
                    outf.write('\n')


def batch_filter_sentences(batch_dict: Dict):
    """
    Filter sentences for supp/drug ents
    :param batch_dict:
    :return:
    """
    input_file = batch_dict["input_file"]
    output_file = batch_dict["output_file"]
    handler = batch_dict["cui_handler"]

    with open(input_file, 'r') as in_f, open(output_file, 'w+') as out_f:
        counter = 0
        for line in tqdm.tqdm(in_f):
            sent = json.loads(line.strip())
            entities = sent["entities"]

            # skip sentence if only one detected entity
            if len(entities) < 2:
                continue

            # skip sentence if all entity strings are the same
            if len(set([ent["string"].strip() for ent in entities])) < 2:
                continue

            # clean entities and construct dicts
            keep_ents = [clean_ent(ent) for ent in entities]

            # generate sentence entry for each pair of entities
            for ent1, ent2 in itertools.combinations(keep_ents, 2):

                # skip if same id
                if ent1['id'] == ent2['id']:
                    continue

                # skip if same entity
                if ent1['string'].strip() == ent2['string'].strip():
                    continue

                # skip if not supp-drug or supp-supp
                if not (handler.is_supp_drug(ent1['id'], ent2['id']) or handler.is_supp_supp(ent1['id'], ent2['id'])):
                    continue

                # create sentence entry for DDI model
                output_dict = {
                    "id": str(sent["id"]) + '-' + str(counter),
                    "sentence_id": sent["sentence_id"],
                    "sentence": sent["sentence"],
                    "arg1": ent1,
                    "arg2": ent2
                }
                json.dump(output_dict, out_f)
                out_f.write('\n')
                counter += 1


CONFIG_FILE = 'config/config.json'
NUM_PROCESSES = multiprocessing.cpu_count() // 8

if __name__ == '__main__':
    # load config file
    assert os.path.exists(CONFIG_FILE)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # determine run type
    rerun_ner = config['rerun_ner']
    rerun_ddi = config['rerun_ddi']
    bert_ddi_model = config['bert_ddi_model']

    # get last time is timestamp available, else get all
    if config['timestamp']:
        LAST_TIME = datetime.strptime(config['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        LAST_TIME = None

    if LAST_TIME:
        print(f"Retrieving papers since {LAST_TIME.strftime('%Y-%m-%dT%H:%M:%S.%fZ')}...")
    else:
        print('Retrieving all papers...')

    # get start time
    START_TIME = datetime.now()

    # determine output file name
    header_time = START_TIME.strftime('%Y%m%d')
    addendum_num = 1
    while os.path.exists(f'output/{header_time}_{addendum_num:02d}.tar.gz'):
        addendum_num += 1
    header_str = f'{header_time}_{addendum_num:02d}'
    OUTPUT_FILE = f'output/{header_str}.tar.gz'

    # remaining headers
    DATA_DIR = '/net/nfs.corp/s2-research/suppai-data/'
    BASE_DIR = os.path.join(DATA_DIR, header_str)
    RAW_DATA_DIR = os.path.join(BASE_DIR, 's2_data')
    ENTITY_DIR = os.path.join(BASE_DIR, 's2_entities')
    SUPP_SENTS_DIR = os.path.join(BASE_DIR, 's2_supp_sents')
    DDI_OUTPUT_DIR = os.path.join(BASE_DIR, 'ddi_output')

    # make output directories
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(ENTITY_DIR, exist_ok=True)
    os.makedirs(SUPP_SENTS_DIR, exist_ok=True)
    os.makedirs(DDI_OUTPUT_DIR, exist_ok=True)

    # --- get new data from S2 DB ---
    print('Getting new papers from S2 DB...')
    data_getter = DataGetter(timestamp=LAST_TIME, output_dir=RAW_DATA_DIR)
    data_getter.get_new_data()

    # --- run NER and Linking ---
    print('Run NER and linking...')
    # form batches
    if rerun_ner:
        all_files = []
        for raw_dir in glob.glob(os.path.join(DATA_DIR, '*', 's2_data')):
            all_files += glob.glob(os.path.join(raw_dir, 's2_data_*'))
    else:
        all_files = glob.glob(os.path.join(RAW_DATA_DIR, 's2_data_*'))
    print(f'{len(all_files)} S2 data files for NER and linking.')

    batches = [{
        "batch_num": batch_num,
        "file_to_process": file_name,
        "entity_file": os.path.join(ENTITY_DIR, f'entities.jsonl.{batch_num}'),
        "skipped_file": os.path.join(ENTITY_DIR, f'skipped.txt.{batch_num}')
    } for batch_num, file_name in enumerate(all_files)]
    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        p.map(batch_run_ner_linking, batches)

    # --- filter sentences for supp/drug CUIs ---
    print('Filtering sentences...')
    # create CUI handler
    cui_handler = CUIHandler()
    # form batches
    if rerun_ner or not (rerun_ner or rerun_ddi):
        all_files = glob.glob(os.path.join(ENTITY_DIR, 'entities.jsonl.*'))
    else:
        all_files = []
        for ent_dir in glob.glob(os.path.join(DATA_DIR, '*', 's2_entities')):
            all_files += glob.glob(os.path.join(ent_dir, 'entities.jsonl.*'))
    print(f'{len(all_files)} entity files for filtering.')

    batches = [{
        "input_file": filename,
        "output_file": os.path.join(SUPP_SENTS_DIR, f'sentences.jsonl.{batch_num}'),
        "cui_handler": cui_handler
    } for batch_num, filename in enumerate(
        sorted(all_files)
    )]
    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        p.map(batch_filter_sentences, batches)

    # aggregate into one file
    with open(os.path.join(SUPP_SENTS_DIR, f'supp_sentences_{header_str}.jsonl'), 'wb') as wfd:
        for f in glob.glob(os.path.join(SUPP_SENTS_DIR, 'sentences.jsonl.*')):
            with open(f, 'rb') as fd:
                shutil.copyfileobj(fd, wfd)
            os.remove(f)

    # --- write output log file ---
    print('Logging...')
    log_dict = {
        "timestamp": START_TIME.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "header_str": header_str,
        "bert_ddi_model": bert_ddi_model,
        "raw_data_dir": RAW_DATA_DIR,
        "entity_dir": ENTITY_DIR,
        "supp_sents_dir": SUPP_SENTS_DIR,
        "ddi_output_dir": DDI_OUTPUT_DIR,
        "aggregate": not (rerun_ner or rerun_ddi),
        "output_file": OUTPUT_FILE
    }
    with open('config/log.json', 'w+') as out_f:
        json.dump(log_dict, out_f, indent=4)

    print('done.')