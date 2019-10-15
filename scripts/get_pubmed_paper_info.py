from s2base import file_util
from s2base.elastic import default_es_client
from joblib import Parallel, delayed
from xml.etree import ElementTree as etree
from urllib.error import URLError
from urllib.request import urlopen
import re
import json
from collections import defaultdict

title_sub = re.compile("[^\w\.']+")

ES_URL = 'es5.development.s2.dev.ai2'
MEDLINE_S3_URLS = ['s3://ai2-s2-data/medline/2019/baseline/', 's3://ai2-s2-data/medline/2019/update/']
OUTPUT_PMID_FILE = 'data/pmid_metadata.json'


def process_s3_file(file_name):
    try:
        return (get_links_from_file(file_name), file_name)
    except Exception as e:
        import pdb;
        pdb.set_trace()
        return (e, file_name)


def get_links_from_file(file_name):
    """ Go through a XML file and find the pubmed, pmc, and various database ids
        And also find all the article types
    """

    root = etree.fromstringlist(list(file_util.read_lines(file_name, streaming=False)))
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

    files = [file for s3_url in MEDLINE_S3_URLS for file in file_util.iterate_files(s3_url)]
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
    for result in results:
        pmid = result['pmid']
        if 'pubtypeslist' not in pmid_metadata[pmid]:
            pmid_metadata[pmid]['pubtypeslist'] = []
        pmid_metadata[pmid]['pubtypeslist'].extend(result['pubtypelist'])
        if 'meshlist' not in pmid_metadata[pmid]:
            pmid_metadata[pmid]['meshlist'] = []
        pmid_metadata[pmid]['meshlist'].extend(result['meshlist'])

    with open(OUTPUT_PMID_FILE, mode='wt') as outfile:
        json.dump(pmid_metadata, outfile)
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