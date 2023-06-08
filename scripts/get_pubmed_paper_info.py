from joblib import Parallel, delayed
from xml.etree import ElementTree as etree
from urllib.error import URLError
from urllib.request import urlopen
import os, sys
import gzip
import re
import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

title_sub = re.compile("[^\w\.']+")

today = datetime.today()
datey = today.year
MEDLINE_S3_URLS = [f's3://ai2-s2-data/medline/{datey}/baseline/', f's3://ai2-s2-data/medline/{datey}/update/']
DATA_DIR = '/net/s3/s2-research/lucyw/pubmed/'
OUTPUT_PMID_FILE = os.path.join(DATA_DIR, 'pmid_metadata.json.gz')


def process_s3_file(file_name):
    try:
        return (get_links_from_file(file_name), file_name)
    except Exception as e:
        return (e, file_name)


def get_links_from_file(file_name):
    """ Go through a XML file and find the pubmed, pmc, and various database ids
        And also find all the article types
    """
    rows = []
    with open(file_name, 'r') as f:
        for row in f:
            rows.append(row)

    root = etree.fromstringlist(rows)
    paperinfos = []
    for child in root:
        databaseids = []
        pubtypelist = []
        meshlist = []
        try:
            pmid = child[0][0].text
        except:
            continue
        pmcid = None
        for elem in child.iterfind('PubmedData/ArticleIdList/ArticleId'):
            if elem.text is not None and elem.text.lower().startswith('pmc'):
                pmcid = elem.text  # should only be one of these
        for elem in child.iterfind('MedlineCitation/Article/DataBankList/DataBank/AccessionNumberList/AccessionNumber'):
            if elem.text is not None:
                databaseids.append(elem.text)
        for elem in child.iterfind('MedlineCitation/Article/PublicationTypeList/'):
            if elem.text is not None:
                pubtypelist.append(elem.text)
        for elem in child.iterfind('MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName'):
            if elem.text is not None:
                meshlist.append(elem.text)
        paperinfo = {'pmid': pmid, 'databaseids': databaseids, 'pmcid': pmcid, 'pubtypelist': pubtypelist,
                     'meshlist': meshlist}
        paperinfos.append(paperinfo)
    return paperinfos


if __name__ == '__main__':
    files = [file for s3_url in MEDLINE_S3_URLS for file in s3_url]
    list_of_results = Parallel(n_jobs=32, verbose=25)(delayed(process_s3_file)(file) for file in files)

    # some errored out - we catch those here and redo
    not_done_files = [i[1] for i in list_of_results if type(i[0]) is not list]
    for file_name in not_done_files:
        print('Processing:', file_name)
        result = process_s3_file(file_name)
        if type(result[0]) is list:
            list_of_results.append(result)
            print('Success!')
        else:
            print('Failed!')

    # combine results from list_of_results
    results = []
    for result, file_name in list_of_results:
        if type(result) is list:
            results.extend(result)

    pmid_metadata = defaultdict(dict)
    for result in tqdm(results):
        pmid = result['pmid']
        if 'pubtypeslist' not in pmid_metadata[pmid]:
            pmid_metadata[pmid]['pubtypeslist'] = []
        pmid_metadata[pmid]['pubtypeslist'].extend(result['pubtypelist'])
        if 'meshlist' not in pmid_metadata[pmid]:
            pmid_metadata[pmid]['meshlist'] = []
        pmid_metadata[pmid]['meshlist'].extend(result['meshlist'])

    with gzip.open(OUTPUT_PMID_FILE, mode='wt') as zipfile:
        json.dump(pmid_metadata, zipfile)
        '''
        The output file consists of one json object (dict). The key is a pubmed paper ID
        and the value is a dict with metadata about the paper (meshlist, pubtypeslist).
        In "meshlist", we include *all* mesh term descriptor names, including the terms 
        "Animals" and "Humans". In "pubtypeslist", we include *all* publication types listed
        in pubmed, including 'Clinical Study' and 'Retracted Publication'. 
        The following is the output for paper ID 12484:
        "12484": {
          "pubtypeslist": [
            "Clinical Trial",
            "English Abstract",
            "Journal Article",
            "Randomized Controlled Trial"
          ],
          "meshlist": [
            "Adult",
            "Anxiety Disorders",
            "Capsules",
            "Clinical Trials as Topic",
            "Drug Evaluation",
            "Female",
            "Humans",
            "Indoles",
            "Middle Aged",
            "Piperazines",
            "Placebos"
          ]
        },
        '''