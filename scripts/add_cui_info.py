import os
import json
import tqdm
from collections import defaultdict
import unicodedata


def process_synonyms(all_names):
    """
    Get list of synonyms
    :param all_names:
    :return:
    """
    all_names.sort(key=lambda x: x[0])

    done_set = set([])
    syn = []
    for order, name_type, name in all_names:
        norm_name = unicodedata.normalize('NFC', name)
        if norm_name and norm_name.lower() not in done_set:
            if order <= 4 or (order > 4 and not syn):
                syn.append(norm_name)
                done_set.add(norm_name.lower())
    return syn


# CUI cluster file
CUI_FILE = 'data/cui_clusters.json'

# MRCONSO file from UMLS (current version 2018AB)
MRCONSO_FILE = '/Users/lucyw/git/hack19_supplements/data/MRCONSO.RRF'

# MRDEF file from UMLS (current version 2018AB)
MRDEF_FILE = '/Users/lucyw/git/hack19_supplements/data/MRDEF.RRF'

# Drug tradename file
DRUG_TRADENAME_FILE = '/Users/lucyw/git/hack19_supplements/data/tradename_mapping.json'

# Preferred name file
PREFERRED_NAME_FILE = 'data/preferred_names.json'

# MRCONSO sort order
TTY_sort_order = {"MH": 0,                      # main heading, supplementary concept name
                  "PEP": 1, "PT": 1,            # preferred terms
                  "ET": 2, "CE": 2, "SY": 2, "SYN": 2, "NA": 2,  # entry terms, aliases, synonyms
                  "AB": 3, "ACR": 3,            # abbreviations and acronyms
                  "NM": 4, "PCE": 4}            # supplementary concepts
TTY_sort_order = defaultdict(lambda: 5, TTY_sort_order)


# read CUI dict
with open(CUI_FILE, 'r') as f:
    cui_dict = json.load(f)

all_cuis = set(cui_dict['supplements'].keys()) | set(cui_dict['drugs'].keys())

# read tradenames
with open(DRUG_TRADENAME_FILE, 'r') as f:
    tradename_dict = json.load(f)

# read preferred names
with open(PREFERRED_NAME_FILE, 'r') as f:
    pref_name_dict = json.load(f)

# read UMLS names and aliases from MRCONSO.RRF, and populate cui_to_names
cui_to_all_names = defaultdict(list)
with open(MRCONSO_FILE, mode='r') as mrconso_file:
    for line in tqdm.tqdm(mrconso_file, desc=f"reading {MRCONSO_FILE}"):
        splits = line.strip().split('|')
        cui, lang, tty, name = splits[0], splits[1], splits[12], splits[14]
        # keep only english language names
        if cui in all_cuis and lang == 'ENG':
            cui_to_all_names[cui].append((TTY_sort_order[tty], tty, name))

# read UMLS definitions from MRDEF.RRF, and populate cui_to_def
cui_to_def = defaultdict(str)
with open(MRDEF_FILE, mode='r') as mrdef_file:
    for line in tqdm.tqdm(mrdef_file, desc=f"reading {MRDEF_FILE}"):
        splits = line.strip().split('|')
        cui, def_text = splits[0], splits[5]
        if cui in all_cuis and def_text and not cui_to_def[cui]:
            cui_to_def[cui] = def_text

# add metadata to supplement CUIs
for k, v in cui_dict['supplements'].items():
    synonyms = process_synonyms(cui_to_all_names[k])
    if k in pref_name_dict:
        pref_name = pref_name_dict[k]
    else:
        pref_name = synonyms[0].title() if len(synonyms[0]) > 4 else synonyms[0]
        synonyms = synonyms[1:]
    v['preferred_name'] = pref_name
    v['synonyms'] = synonyms
    v['definition'] = cui_to_def[k]

# add metadata to drug CUIs
for k, v in cui_dict['drugs'].items():
    synonyms = process_synonyms(cui_to_all_names[k])
    if k in pref_name_dict:
        pref_name = pref_name_dict[k]
    else:
        pref_name = synonyms[0].title() if len(synonyms[0]) > 4 else synonyms[0]
        synonyms = synonyms[1:]
    v['preferred_name'] = pref_name
    v['synonyms'] = synonyms
    v['definition'] = cui_to_def[k]
    v['tradenames'] = [trade[1] for trade in tradename_dict[k]][:20] if k in tradename_dict else []

with open(CUI_FILE, 'w') as outf:
    json.dump(cui_dict, outf, indent=4, sort_keys=True)