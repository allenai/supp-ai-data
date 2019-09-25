"""
Compile BERT-DDI results
Create final dicts for site

"""

import os
import sys
import json
import glob
import tqdm
import gzip
import tarfile
import multiprocessing
from typing import List, Dict, Tuple
from collections import defaultdict

from suppai.cui_handler import CUIHandler
from suppai.utils.db_utils import get_paper_metadata

from s2base2.list_utils import chunk_iter


def keep_positives(input_dirs: List[str]) -> Dict:
    """
    Read all inputs from input directories and keep only those with
    :param input_dirs:
    :return:
    """
    supp_sentences = dict()

    # iterate through all input data directories
    for in_dir, in input_dirs:
        base_dir, date_str, folder = in_dir.split('/')
        label_dir = os.path.join(base_dir, date_str, 'ddi_output')
        for in_file in glob.glob(os.path.join(in_dir, '*.jsonl')):
            in_file_name = os.path.basename(in_file)
            label_file_name = in_file_name.replace('sentences', 'labels')
            label_file = os.path.join(label_dir, label_file_name)
            if not os.path.exists(label_file):
                print(f"Label file doesn't exist! {label_file}")
            with open(in_file, 'r') as in_f:
                for line_index, line in enumerate(tqdm.tqdm(in_f, desc=f"reading {in_file}")):
                    if 0 < READ_TOP_K_LINES <= line_index:
                        break
                    entry = json.loads(line)
                    supp_sentences[f'{date_str}-{entry["id"]}'] = entry
            with open(label_file, 'r') as lab_f:
                for line_index, line in enumerate(tqdm.tqdm(lab_f, desc=f"reading {label_file}")):
                    if 0 < READ_TOP_K_LINES <= line_index:
                        break
                    entry = json.loads(line)
                    supp_sentences[f'{date_str}-{entry["id"]}']["label-model"] = int(entry["label_model"])

    # filter out sentences with no labels
    positives = {
        k: v for k, v in tqdm.tqdm(supp_sentences.items())
        if "label-model" in v and v["label-model"] == 1
    }

    print(f'{len(positives)} positive sentences out of {len(supp_sentences)}.')
    return positives


def create_interaction_sentence_dicts(positives: Dict, blacklist: List[str]) -> Tuple[Dict, Dict]:
    """
    Create interaction and sentence dicts
    :param positives:
    :param blacklist:
    :return:
    """
    # initialize
    interaction_dict = defaultdict(set)
    sentence_dict = defaultdict(list)

    # create CUI handler
    handler = CUIHandler()

    # keep only unique entity pairs
    uniq_sentence_pairs = set([])

    for k, v in positives.items():
        paper_id, uid = v["id"].split('-')

        # skip if sentence empty
        if not v["sentence"]:
            continue

        # get arguments
        arg1 = v["arg1"]
        arg2 = v["arg2"]

        # get CUIs
        arg1_cui = handler.normalize_cui(arg1["id"])
        arg2_cui = handler.normalize_cui(arg2["id"])

        # if CUIs are the same, skip
        if arg1_cui == arg2_cui:
            continue

        # if either CUI not valid
        if not handler.is_valid_cui(arg1_cui) or not handler.is_valid_cui(arg2_cui):
            continue

        # if not one supplement and one drug or both supplements, skip
        if not handler.is_supp_drug(arg1_cui, arg2_cui) and not handler.is_supp_supp(arg1_cui, arg2_cui):
            continue

        # if any of the spans are in blacklist
        span1_lower = arg1["string"].strip(' .,').lower()
        span2_lower = arg2["string"].strip(' .,').lower()

        if span1_lower in blacklist or span2_lower in blacklist:
            continue

        # if either span starts or ends with compound(s)
        if span1_lower.startswith("compound") or span1_lower.endswith("compound") or span1_lower.endswith("compounds"):
            continue
        if span2_lower.startswith("compound") or span2_lower.endswith("compound") or span2_lower.endswith("compounds"):
            continue

        # put two entities in CUI alphabetical order
        if arg1_cui > arg2_cui:
            arg1, arg2 = arg2, arg1

        # construct sentence representation
        entry = {
            "uid": int(uid),
            "confidence": None,
            "paper_id": paper_id,
            "sentence_id": int(v["sentence_id"]),
            "sentence": v["sentence"].replace("\n", " "),
            "arg1": {
                "id": arg1["id"],
                "span": arg1["span"][0]
            },
            "arg2": {
                "id": arg2["id"],
                "span": arg2["span"][0]
            }
        }

        # add sentence if sentence and entity pair does not already exist
        # (filters for duplicate sentences w/ the same entity mentions)
        if (entry["sentence"], entry["arg1"]["id"], entry["arg2"]["id"]) not in uniq_sentence_pairs:
            # construct interaction id
            interaction_id = f'{arg1["id"]}-{arg2["id"]}'

            # add interaction id to both CUIs
            interaction_dict[arg1["id"]].add(interaction_id)
            interaction_dict[arg2["id"]].add(interaction_id)

            # add interaction sentence to sentence dict
            sentence_dict[interaction_id].append(entry)

            # add to uniq sentence pairs
            uniq_sentence_pairs.add((entry["sentence"], entry["arg1"]["id"], entry["arg2"]["id"]))

    return interaction_dict, sentence_dict


def create_paper_metadata_dict(interaction_dict: Dict, sentence_dict: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Create paper metadata dict
    :param interaction_dict:
    :param sentence_dict:
    :return:
    """
    # get the set of paper ids
    all_paper_ids = []
    for entries in sentence_dict.values():
        for entry in entries:
            all_paper_ids.append(entry["paper_id"])
    all_paper_ids = list(set(all_paper_ids))

    # initialize metadata dict
    paper_metadata_dict = dict()

    # Read relevant publication types and mesh terms.
    with gzip.GzipFile(MEDLINE_METADATA, 'r') as medline_metadata_file:
        json_bytes = medline_metadata_file.read()
    json_str = json_bytes.decode('utf-8')
    medline_metadata = json.loads(json_str)
    human_pmids, animal_pmids = set(), set()
    retraction_pmids, clinical_trial_pmids = set(), set()

    for pmid, metadata in medline_metadata.items():
        if any([mesh_term in ANIMAL_MESH_TERMS for mesh_term in metadata['meshlist']]):
            animal_pmids.add(int(pmid))
        if any([mesh_term in HUMAN_MESH_TERMS for mesh_term in metadata['meshlist']]):
            human_pmids.add(int(pmid))
        if any([pubtype in CLINICAL_TRIAL_PUBTYPES for pubtype in metadata['pubtypeslist']]):
            clinical_trial_pmids.add(int(pmid))
        if any([pubtype in RETRACTION_PUBTYPES for pubtype in metadata['pubtypeslist']]):
            retraction_pmids.add(int(pmid))

    # get paper metadata from DB
    print(f'fetching paper metadata ({len(all_paper_ids)})...')
    chunk_size = 1000
    for paper_chunk_index, paper_chunk in enumerate(
            tqdm.tqdm(chunk_iter(all_paper_ids, chunk_size), total=(len(all_paper_ids) // chunk_size + 1),
                      desc='Fetching paper metadata')):
        if 0 < READ_TOP_K_LINES <= paper_chunk_index:
            continue
        paper_metadata = get_paper_metadata(paper_chunk)
        # TODO: fails to retrieve metadata for some papers, check why, temp solution is to remove missing papers
        for s2_id, metadata_entry in paper_metadata.items():
            metadata_entry["animal_study"] = metadata_entry['pmid'] in animal_pmids
            metadata_entry["human_study"] = metadata_entry['pmid'] in human_pmids
            metadata_entry["clinical_study"] = metadata_entry['pmid'] in clinical_trial_pmids
            metadata_entry["retraction"] = metadata_entry['pmid'] in retraction_pmids
            paper_metadata_dict[s2_id] = metadata_entry

    # temporarily remove interactions and sentences where we can't find paper metadata
    paper_ids_to_remove = set(all_paper_ids).difference(set(paper_metadata_dict.keys()))
    print(f'{len(paper_ids_to_remove)} papers with missing metadata.')

    interactions_to_remove = []
    for interaction_id, sentences in sentence_dict.items():
        new_sents = [sent for sent in sentences if sent["paper_id"] not in paper_ids_to_remove]
        if new_sents:
            sentence_dict[interaction_id] = new_sents
        else:
            interactions_to_remove.append(interaction_id)

    for interaction_id in interactions_to_remove:
        del sentence_dict[interaction_id]
        cui1, cui2 = interaction_id.split('-')
        interaction_dict[cui1].remove(interaction_id)
        interaction_dict[cui2].remove(interaction_id)

    return interaction_dict, sentence_dict, paper_metadata_dict


def create_cui_metadata_dict(interaction_dict: Dict, sentence_dict: Dict):
    """
    Create CUI metadata dict
    :param interaction_dict:
    :param sentence_dict:
    :return:
    """
    # create CUI handler
    handler = CUIHandler()

    # set of CUIs
    all_cuis = set(interaction_dict.keys()) | handler.supps

    # initialize cui metadata dict
    cui_metadata_dict = dict()

    # iterate through all CUIs
    for cui in all_cuis:
        try:
            cui_metadata_dict[cui] = handler.form_cui_entry(cui)
        except KeyError:
            print(f'{cui} missing!')
            sents_to_del = interaction_dict[cui]
            for sent_id in interaction_dict[cui]:
                del sentence_dict[sent_id]
            for k, v in interaction_dict.items():
                for sent_id in sents_to_del:
                    if sent_id in v:
                        v.remove(sent_id)
            del interaction_dict[cui]
            continue

    return interaction_dict, sentence_dict, cui_metadata_dict


def form_dicts(positive_sents: Dict, out_file: str, blacklist_str: List[str]):
    """
    Create final dictionaries for supp.ai
    :param positive_sents:
    :param out_file:
    :param blacklist_str:
    :return:
    """
    # CREATE INTERACTION IDS AND SENTENCE DICT
    interactions, sentences = create_interaction_sentence_dicts(positive_sents, blacklist_str)

    # CREATE PAPER METADATA DICT AND REMOVE MISSING ENTRIES FROM OTHER DICTS
    interactions, sentences, papers = create_paper_metadata_dict(interactions, sentences)

    # CREATE CUI METADATA DICT FROM ENTRIES IN INTERACTIONS
    interactions, sentences, cuis = create_cui_metadata_dict(interactions, sentences)

    # output file names
    interaction_file = 'output/interaction_id_dict.json'
    sentence_file = 'output/sentence_dict.json'
    paper_file = 'output/paper_metadata.json'
    cui_file = 'output/cui_metadata.json'

    # write to output
    with open('output/interaction_id_dict.json', 'w') as out_f:
        json.dump(interactions, out_f, indent=4, sort_keys=True)
    with open('output/sentence_dict.json', 'w') as out_f:
        json.dump(sentences, out_f, indent=4, sort_keys=True)
    with open('output/paper_metadata.json', 'w') as out_f:
        json.dump(papers, out_f, indent=4, sort_keys=True)
    with open('output/cui_metadata.json', 'w') as out_f:
        json.dump(cuis, out_f, indent=4, sort_keys=True)

    # tar and zip dicts together
    with tarfile.open(out_file, "w:gz") as tar:
        tar.add(interaction_file)
        tar.add(sentence_file)
        tar.add(paper_file)
        tar.add(cui_file)


LOG_FILE = 'config/log.json'
BLACKLIST_FILE = 'data/blacklist.txt'
MEDLINE_METADATA = 'data/pmid_metadata.json.gz'

RETRACTION_PUBTYPES = {
    'Retraction of Publication',
    'Retracted Publication'
}
CLINICAL_TRIAL_PUBTYPES = {
    'Adaptive Clinical Trial',
    'Clinical Study',
    'Clinical Trial',
    'Clinical Trial, Phase I',
    'Clinical Trial, Phase II',
    'Clinical Trial, Phase III',
    'Clinical Trial, Phase IV',
    'Controlled Clinical Trial',
    'Pragmatic Clinical Trial'
}
ANIMAL_MESH_TERMS = {'Animals'}
HUMAN_MESH_TERMS = {'Humans'}

READ_TOP_K_LINES = 10000
NUM_PROCESSES = multiprocessing.cpu_count() // 2

if __name__ == '__main__':
    # read preprocessing log file
    with open(LOG_FILE, 'r') as f:
        log_dict = json.load(f)

    output_file = log_dict['output_file']
    aggregate = log_dict['aggregate']

    # determine if need to aggregate results from multiple BERT-DDI runs
    if aggregate:
        all_input_dir = sorted(glob.glob(os.path.join('data', '*', 's2_supp_sents')))
        all_label_dir = sorted(glob.glob(os.path.join('data', '*', 'ddi_output')))
        if len(all_input_dir) != len(all_label_dir):
            print('Not the same number of input and label directories!')
            sys.exit(1)
    else:
        all_input_dir = [log_dict['supp_sents_dir']]
        all_label_dir = [log_dict['ddi_output_dir']]

    # filter and keep only positive interactions labeled by model
    interactions = keep_positives(all_input_dir)

    # load blacklist spans
    blacklist_spans = []
    with open(BLACKLIST_FILE, 'r') as f:
        for line in f:
            blacklist_spans.append(line.strip())

    # form final dictionaries
    form_dicts(interactions, output_file, blacklist_spans)
    print(f'Dicts written to {output_file}')

    print('done.')