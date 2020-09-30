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
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import copy
import re

from suppai.cui_handler import CUIHandler
from suppai.utils.db_utils import get_paper_metadata
from suppai.data import CUIMetadata, PaperAuthor, PaperMetadata, LabeledSpan, EvidenceSentence

from s2base2.list_utils import chunk_iter


def keep_positives(input_dirs: List[str], label_dirs: List[str]) -> List[EvidenceSentence]:
    """
    Read all inputs from input directories and keep only those with
    :param input_dirs:
    :param label_dirs:
    :return:
    """
    positives = []
    total_sents = 0

    # uniq_set (k = paper_id, v = set((sentence, arg1, arg2)))
    uniq_set = defaultdict(set)
    uniq_id = 0

    # create CUI handler
    handler = CUIHandler()

    # iterate through all input data directories
    for in_dir, lab_dir in zip(input_dirs, label_dirs):
        for in_file in glob.glob(os.path.join(in_dir, '*.jsonl')):
            in_file_name = os.path.basename(in_file)
            label_file_name = in_file_name.replace('sentences', 'labels')
            label_file = os.path.join(lab_dir, label_file_name)
            if not os.path.exists(label_file):
                print(f"Label file doesn't exist! {label_file}")

            # read labels and keep only positives
            positive_keys = set([])
            with open(label_file, 'r') as lab_f:
                for line_index, line in enumerate(tqdm.tqdm(lab_f, desc=f"reading {label_file}")):
                    if 0 < READ_TOP_K_LINES <= line_index:
                        break
                    entry = json.loads(line)
                    if int(entry["label-model"]) == 1:
                        positive_keys.add(entry["id"])

            # get sentence info for positive sentences
            with open(in_file, 'r') as in_f:
                for line_index, line in enumerate(tqdm.tqdm(in_f, desc=f"reading {in_file}")):
                    total_sents += 1
                    if 0 < READ_TOP_K_LINES <= line_index:
                        break
                    entry = json.loads(line)
                    pid, _ = entry['id'].split('-')
                    arg1_cui = handler.normalize_cui(entry['arg1']['id'])
                    arg2_cui = handler.normalize_cui(entry['arg2']['id'])
                    span1_inds = entry['arg1']['span'][0]
                    span2_inds = entry['arg2']['span'][0]

                    # reverse cui positions if non-alphabetical
                    if arg1_cui > arg2_cui:
                        arg1_cui, arg2_cui = arg2_cui, arg1_cui
                        span1_inds, span2_inds = span2_inds, span1_inds

                    if entry["id"] in positive_keys:
                        # add sentence if sentence and entity pair does not already exist
                        # (filters for duplicate sentences w/ the same entity mentions)
                        if (entry['sentence'], arg1_cui, arg2_cui) in uniq_set[pid]:
                            continue
                        else:
                            new_evidence = EvidenceSentence(
                                uid=uniq_id,
                                paper_id=pid,
                                sentence_id=entry['sentence_id'],
                                sentence=re.sub(r'\s', ' ', entry['sentence']),
                                confidence=None,
                                arg1=LabeledSpan(id=arg1_cui, span=span1_inds),
                                arg2=LabeledSpan(id=arg2_cui, span=span2_inds)
                            )
                            positives.append(new_evidence)
                            uniq_set[pid].add(
                                (entry['sentence'], arg1_cui, arg2_cui)
                            )
                            uniq_id += 1

    print(f'{len(positives)} positive sentences out of {total_sents}.')
    return positives


def create_interaction_sentence_dicts(positives: List[EvidenceSentence], blacklist: List[str]) -> Tuple[Dict, Dict]:
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

    for pos in tqdm.tqdm(positives):
        # skip if sentence empty
        if not pos.sentence:
            continue

        # if either cui are not normalizable
        if not pos.arg1.id or not pos.arg2.id:
            continue

        # if CUIs are the same, skip
        if pos.arg1.id == pos.arg2.id:
            continue

        # if not one supplement and one drug or both supplements, skip
        if not handler.is_supp_drug(pos.arg1.id, pos.arg2.id) and not handler.is_supp_supp(pos.arg1.id, pos.arg2.id):
            continue

        # if any of the spans are in blacklist
        span1_lower = pos.sentence[pos.arg1.span[0]:pos.arg1.span[1]].strip(' .,').lower()
        span2_lower = pos.sentence[pos.arg2.span[0]:pos.arg2.span[1]].strip(' .,').lower()

        if span1_lower in blacklist or span2_lower in blacklist:
            continue

        # if either span starts or ends with compound(s)
        if span1_lower.startswith("compound") or span1_lower.endswith("compound") or span1_lower.endswith("compounds"):
            continue
        if span2_lower.startswith("compound") or span2_lower.endswith("compound") or span2_lower.endswith("compounds"):
            continue

        # if 'atp' linked to azathioprine
        if pos.arg1.id == 'C0004482' and span1_lower == 'atp':
            continue
        if pos.arg2.id == 'C0004482' and span2_lower == 'atp':
            continue

        # if 'ca2' linked to infliximab
        if pos.arg1.id == 'C0666743' and span1_lower == 'ca2':
            continue
        if pos.arg2.id == 'C0666743' and span2_lower == 'ca2':
            continue

        # construct interaction id
        interaction_id = f'{pos.arg1.id}-{pos.arg2.id}'

        # add interaction id to both CUIs
        interaction_dict[pos.arg1.id].add(interaction_id)
        interaction_dict[pos.arg2.id].add(interaction_id)

        # add interaction sentence to sentence dict
        sentence_dict[interaction_id].append(pos)

    return interaction_dict, sentence_dict


def create_cui_metadata_dict(interaction_dict: Dict[str, Set], sentence_dict: Dict[str, List[EvidenceSentence]]):
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
    cuis_to_delete = set([])

    for cui in all_cuis:
        try:
            cui_metadata_dict[cui] = handler.form_cui_entry(cui)
        except KeyError:
            print(f'{cui} missing!')
            cuis_to_delete.add(cui)

    for cui in cuis_to_delete:
        sents_to_del = copy.copy(interaction_dict[cui])
        for k, v in interaction_dict.items():
            for sent_id in sents_to_del:
                if sent_id in v:
                    v.remove(sent_id)
        for sent_id in sents_to_del:
            del sentence_dict[sent_id]
        del interaction_dict[cui]

    # add empty interaction entries
    for cui in cui_metadata_dict:
        if cui not in interaction_dict:
            interaction_dict[cui] = set([])

    return interaction_dict, sentence_dict, cui_metadata_dict


def create_paper_metadata_dict(
        interaction_dict: Dict[str, Set],
        sentence_dict: Dict[str, List[EvidenceSentence]],
        cui_dict: Dict[str, CUIMetadata]
) -> Tuple[
    Dict[str, Set],
    Dict[str, List[EvidenceSentence]],
    Dict[str, CUIMetadata],
    Dict[str, PaperMetadata]
]:
    """
    Create paper metadata dict
    :param interaction_dict:
    :param sentence_dict:
    :param cui_dict:
    :return:
    """
    # get the set of paper ids
    all_paper_ids = []
    for entries in sentence_dict.values():
        for entry in entries:
            all_paper_ids.append(entry.paper_id)
    all_paper_ids = list(set(all_paper_ids))
    print(f'{len(all_paper_ids)} papers')

    # initialize metadata dict
    paper_metadata_dict = dict()

    # Read relevant publication types and mesh terms.
    # TODO: change this to query the DB for this info so it's up-to-date
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
            paper_metadata_dict[s2_id] = PaperMetadata(
                title=metadata_entry["title"],
                authors=metadata_entry["authors"],
                year=metadata_entry["year"],
                venue=metadata_entry["venue"],
                doi=metadata_entry["doi"],
                pmid=metadata_entry["pmid"],
                fields_of_study=metadata_entry["fields_of_study"],
                retraction=metadata_entry["pmid"] in animal_pmids,
                clinical_study=metadata_entry["pmid"] in clinical_trial_pmids,
                human_study=metadata_entry["pmid"] in human_pmids,
                animal_study=metadata_entry["pmid"] in animal_pmids
            )

    # temporarily remove interactions and sentences where we can't find paper metadata
    paper_ids_to_remove = set(all_paper_ids) - set(paper_metadata_dict.keys())
    print(f'{len(paper_ids_to_remove)} papers with missing metadata.')

    interactions_to_remove = []
    for interaction_id, sentences in sentence_dict.items():
        new_sents = [sent for sent in sentences if sent.paper_id not in paper_ids_to_remove]
        if new_sents:
            sentence_dict[interaction_id] = new_sents
        else:
            interactions_to_remove.append(interaction_id)

    for interaction_id in interactions_to_remove:
        if interaction_id in sentence_dict:
            del sentence_dict[interaction_id]
        cui1, cui2 = interaction_id.split('-')
        if interaction_id in interaction_dict[cui1]:
            interaction_dict[cui1].remove(interaction_id)
        if interaction_id in interaction_dict[cui2]:
            interaction_dict[cui2].remove(interaction_id)

    cuis_to_remove = [cui for cui in interaction_dict if cui not in cui_dict]
    for cui in cuis_to_remove:
        del interaction_dict[cui]

    return interaction_dict, sentence_dict, cui_dict, paper_metadata_dict


def form_dicts(positive_sents: List[EvidenceSentence], out_file: str, blacklist_str: List[str], timestr: str):
    """
    Create final dictionaries for supp.ai
    :param positive_sents:
    :param out_file:
    :param blacklist_str:
    :param timestr:
    :return:
    """
    # CREATE INTERACTION IDS AND SENTENCE DICT
    print('Creating interaction and sentence dicts...')
    interactions, sentences = create_interaction_sentence_dicts(positive_sents, blacklist_str)

    # CREATE CUI METADATA DICT FROM ENTRIES IN INTERACTIONS
    print('Creating CUI metadata dict...')
    interactions, sentences, cuis = create_cui_metadata_dict(interactions, sentences)

    # CREATE PAPER METADATA DICT AND REMOVE MISSING ENTRIES FROM OTHER DICTS
    print('Creating paper metadata dict...')
    interactions, sentences, cuis, papers = create_paper_metadata_dict(interactions, sentences, cuis)

    interactions = {k: list(v) for k, v in interactions.items()}
    sentences = {k: [s.as_json() for s in v] for k, v in sentences.items()}
    cuis = {k: v.as_json() for k, v in cuis.items()}
    papers = {k: v.as_json() for k, v in papers.items()}

    # output file names
    interaction_file = 'output/interaction_id_dict.json'
    sentence_file = 'output/sentence_dict.json'
    paper_file = 'output/paper_metadata.json'
    cui_file = 'output/cui_metadata.json'
    meta_file = 'output/meta.json'

    # write to output
    with open(interaction_file, 'w') as out_f:
        json.dump(interactions, out_f, indent=4, sort_keys=True)
    with open(sentence_file, 'w') as out_f:
        json.dump(sentences, out_f, indent=4, sort_keys=True)
    with open(paper_file, 'w') as out_f:
        json.dump(papers, out_f, indent=4, sort_keys=True)
    with open(cui_file, 'w') as out_f:
        json.dump(cuis, out_f, indent=4, sort_keys=True)
    with open(meta_file, 'w') as out_f:
        json.dump({
            "last_updated_on": timestr
        }, out_f)

    # tar and zip dicts together
    with tarfile.open(out_file, "w:gz") as tar:
        tar.add(interaction_file, arcname=os.path.split(interaction_file)[1])
        tar.add(sentence_file, arcname=os.path.split(sentence_file)[1])
        tar.add(paper_file, arcname=os.path.split(paper_file)[1])
        tar.add(cui_file, arcname=os.path.split(cui_file)[1])
        tar.add(meta_file, arcname=os.path.split(meta_file)[1])


DATA_DIR = '/net/nfs.corp/s2-research/suppai-data/'
LOG_FILE = 'config/log.json'
BLACKLIST_FILE = 'data/blacklist.txt'
MEDLINE_METADATA = os.path.join(DATA_DIR, 'pmid_metadata.json.gz')

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

READ_TOP_K_LINES = 0

if __name__ == '__main__':
    # read preprocessing log file
    with open(LOG_FILE, 'r') as f:
        log_dict = json.load(f)

    output_file = log_dict['output_file']
    aggregate = log_dict['aggregate']

    # determine if need to aggregate results from multiple BERT-DDI runs
    if aggregate:
        all_input_dir = sorted(glob.glob(os.path.join(DATA_DIR, '*', 's2_supp_sents')))
        all_label_dir = sorted(glob.glob(os.path.join(DATA_DIR, '*', 'ddi_output')))
        if len(all_input_dir) != len(all_label_dir):
            print('Not the same number of input and label directories!')
            sys.exit(1)
    else:
        all_input_dir = [log_dict['supp_sents_dir']]
        all_label_dir = [log_dict['ddi_output_dir']]

    # filter and keep only positive interactions labeled by model
    interactions = keep_positives(all_input_dir, all_label_dir)

    # load blacklist spans
    blacklist_spans = []
    with open(BLACKLIST_FILE, 'r') as f:
        for line in f:
            blacklist_spans.append(line.strip())

    # form final dictionaries
    form_dicts(interactions, output_file, blacklist_spans, log_dict['timestamp'])
    print(f'Dicts written to {output_file}')

    # write config file
    with open('config/config.json', 'w') as config_f:
        json.dump({
            "timestamp": log_dict["timestamp"],
            "rerun_ner": False,
            "rerun_ddi": False,
            "bert_ddi_model": log_dict['bert_ddi_model']
        }, config_f, indent=4)

    print('done.')