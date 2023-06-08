"""
Retrieve papers from S2 database along with abstracts
For use in SDI identification...
"""

import os
import csv
import json
import tqdm
from datetime import datetime
import requests
import urllib
import subprocess
from easy_entrez import EntrezAPI

entrez_api = EntrezAPI(
    'supp.ai',
    'lucylw@uw.edu',
    # optional
    return_type='json'
)

TEMP_DIR = 'temp/'
DONE_PAPERS_DIR = 'temp/done_papers/'
DONE_ABSTRACT_DIR = 'temp/done_abstracts/'

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DONE_PAPERS_DIR, exist_ok=True)
os.makedirs(DONE_ABSTRACT_DIR, exist_ok=True)

# switch to getting data out of the Entrez API
class DataGetterAPI:
    def __init__(self, timestamp: datetime, output_dir: str):
        """
        Create data getter object and load timestamp from last run
        :param output_dir:
        """
        self.last_time = timestamp
        self.output_dir = output_dir

    def get_s2ids(self):
        """
        Fetch good S2 ids from S2 Datasets API
        """
        output_file = os.path.join(self.output_dir, f's2ids.txt')

        # Get info about the latest release
        latest_release = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest").json()
        print(latest_release['README'])
        print(latest_release['release_id'])

        # Get info about the papers dataset
        papers = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers",
                            headers={'x-api-key':os.getenv("S2_API_KEY")}).json()

        # Download each part of the dataset
        for i, fname in enumerate(papers['files']):
            # check if already done, if so, skip
            done_file = os.path.join(DONE_PAPERS_DIR, f"papers-part{i}.done")
            if os.path.exists(done_file):
                print(f'Already downloaded papers chunk {i}, skipping...')
                continue

            # else download
            print(f'Downloading papers chunk {i}...')
            paper_temp_file = os.path.join(TEMP_DIR, f"papers-part{i}.jsonl.gz")
            urllib.request.urlretrieve(fname, paper_temp_file)
            
            # get pubmed papers
            keep_corpus_ids = []
            with open(paper_temp_file, 'r') as f:
                for line in tqdm.tqdm(f):
                    entry = json.loads(line)
                    # check if pubmed or pmc id
                    if entry['externalids']['PubMed']:
                        keep_corpus_ids.append((entry['corpusid'], entry['externalids']['PubMed'], entry['title']))
                        continue

            # write to output file
            with open(output_file, 'a+') as outf:
                writer = csv.writer(outf, delimiter='|', quotechar='"')
                for row in keep_corpus_ids:
                    writer.writerow(row)

            # write done file
            open(done_file, 'w').close()

            # delete temp file
            os.remove(paper_temp_file)

        return output_file

    def get_abstracts(self, id_file):
        """
        Fetch abstracts of good S2 ids from S2 Datasets API
        """
        abstract_file = os.path.join(self.output_dir, f'abstracts.txt')

        # corpus ids to fetch abstracts of
        corpus_ids = []
        with open(id_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                corpus_ids.append(row[0])
        corpus_ids = set(corpus_ids)

        # Download abstract dataset
        abstracts = requests.get("http://api.semanticscholar.org/datasets/v1/release/latest/dataset/abstracts",
                            headers={'x-api-key':os.getenv("S2_API_KEY")}).json()
        
        # Download each part of the dataset
        for i, fname in enumerate(abstracts['files']):
            done_file = os.path.join(DONE_ABSTRACT_DIR, f"abstracts-part{i}.done")
            if os.path.exists(done_file):
                print(f'Already downloaded abstract chunk {i}, skipping...')
                continue

            print(f'Downloading abstract chunk {i}...')
            abstract_temp_file = os.path.join(TEMP_DIR, f"abstracts-part{i}.jsonl.gz")
            urllib.request.urlretrieve(fname, abstract_temp_file)

            # iterate through an keep abstracts with matching corpus ids
            keep_abstracts = []
            with open(abstract_temp_file, 'r') as f:
                for line in tqdm.tqdm(f):
                    entry = json.loads(line)
                    if entry['corpusid'] in corpus_ids:
                        keep_abstracts.append((entry['corpusid'], entry['abstract']))
            
            # write to output file
            with open(abstract_file, 'a+') as outf:
                writer = csv.writer(outf, delimiter='|', quotechar='"')
                for entry in keep_abstracts:
                    writer.writerow(entry)

            # write done file
            open(done_file, 'w').close()

            # delete temp file
            os.remove(abstract_temp_file)

        return abstract_file
    
    def get_mesh(self, id_file):
        """
        Query S2 API for Pubmed identifiers 
        """
        # corpus ids to fetch abstracts and mesh terms for
        pmids = []
        with open(id_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                pmids.append(row[1])
        pmids = set(pmids)

        # results from pubmed
        pm_result = (
            entrez.api
            .in_batches_of(1_000)
            .fetch(pmids, max_results=5_000, database='pubmed')
        )

        # get abstract and mesh terms




    def get_new_data(self):
        """
        Get papers from S2 API
        """
        id_file = self.get_s2ids()
        # abs_file = self.get_abstracts(id_file)
        mesh_file = self.get_mesh(id_file)

        # split file into chunks
        subprocess.run(["split", "-l", "100000", output_file, os.path.join(self.output_dir, f's2_data_')])