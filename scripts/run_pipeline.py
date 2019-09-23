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

from src.data_getter import DataGetter
from src.ner_and_linker import DrugSupplementLinker

from s2base2.list_utils import make_chunks


def batch_run_ner_linking(batch_dict):
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

    file_list = batch_dict["file_list"]
    entity_file = batch_dict["entity_file"]
    skipped_file = batch_dict["skipped_file"]

    # fire up scispacy linker
    ds_linker = DrugSupplementLinker()

    with open(entity_file, 'w+') as outf, open(skipped_file, 'w+') as skip_f:
        # iterate through files in input list
        for input_file in tqdm.tqdm(file_list):
            with open(input_file, 'r') as f:
                for line in f:
                    entry = json.loads(line)

                    # skip if no abstract
                    if not entry['abstract']:
                        skip_f.write(f"{entry['paper_id']}\n")
                        continue

                    try:
                        ents_per_sentence = ds_linker.get_linked_entities(
                            entry['abstract'],
                            top_k=3
                        )
                    except Exception as e:
                        skip_f.write(f"{entry['paper_id']}\n")
                        continue

                    # iterate through sentences
                    for sent in ents_per_sentence:
                        entities = sent["entities"]
                        if entities:
                            output_dict = {
                                "id": entry["paper_id"],
                                "sentence_id": sent["sent_num"],
                                "sentence": sent["sentence"],
                                "entities": entities
                            }
                            json.dump(output_dict, outf)
                            outf.write('\n')


CONFIG_FILE = 'config/config.json'
NUM_PROCESSES = multiprocessing.cpu_count() // 2

if __name__ == '__main__':
    # load config file
    assert os.path.exists(CONFIG_FILE)
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    # get last time
    LAST_TIME = datetime.strptime(config['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')

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
    BASE_DIR = f'data/{header_str}/'
    RAW_DATA_DIR = os.path.join(BASE_DIR, 's2_data')
    ENTITY_DIR = os.path.join(BASE_DIR, 's2_entities')
    SUPP_SENTS_DIR = os.path.join(BASE_DIR, 's2_supp_sents')
    DDI_OUTPUT_DIR = os.path.join(BASE_DIR, 'ddi_output')

    # make output directories
    os.makedirs(BASE_DIR)
    os.makedirs(RAW_DATA_DIR)
    os.makedirs(ENTITY_DIR)
    os.makedirs(SUPP_SENTS_DIR)
    os.makedirs(DDI_OUTPUT_DIR)

    # get new data from S2 DB
    data_getter = DataGetter(timestamp=LAST_TIME, output_dir=RAW_DATA_DIR)
    data_getter.get_new_data()

    # run NER and Linking
    # form batches
    batches = [{
        "batch_num": batch_num,
        "file_list": file_batch,
        "entity_file": os.path.join(ENTITY_DIR, f'entities.jsonl.{batch_num}'),
        "skipped_file": os.path.join(ENTITY_DIR, f'skipped.txt.{batch_num}')
    } for batch_num, file_batch in enumerate(make_chunks(
        sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*.jsonl'))),
        NUM_PROCESSES
    ))]
    with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
        p.map(batch_run_ner_linking, batches)

    # TODO: filter CUIs for supplements and drugs

    # TODO: compile and run BERT-DDI model on sentences

    # TODO: compile BERT-DDI results

    # TODO: form final dicts

    # TODO: tar and zip output files

    # TODO: write log




