"""
Retrieve Pubmed papers from S2 database along with metadata, abstracts, and full text
For use in SDI identification...
"""

import os
import csv
import json
import tqdm
from datetime import datetime
from typing import List, Dict
import multiprocessing
from time import sleep

from s2base2.config import DB_REDSHIFT, SOURCE_DATA_SERVICE
from s2base2.list_utils import make_chunks
from s2base2.db_utils import S2DBIterator
from s2base2.sds_utils import SdsClient, PaperElementType


"""
Redshift content.papers example entry

paper_sha               712fa8239d5705290225221fcadedece50bf46d7
corpus_paper_id         2
doi                     10.1016/j.ejor.2015.05.030
title                   Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research
year                    2015
journal_name            Eur. J. Oper. Res.
venue                   Eur. J. Oper. Res.
fields_of_study         ["Computer Science"]
pdf_processed           True
has_abstract            True
from_medline            False
from_dblp               True
sources_json            "Unpaywall","ScienceParseMerged","Anansi","DBLP","MAG"
arxiv_id                None
open_access_url         https://eprints.soton.ac.uk/377196/1/Lessmann_Benchmarking.pdf
open_access_url_is_pdf  True
abstract
pdf_sources             "ScienceParseMerged","Anansi"
mag_id                  2131816657
pubmed_id               None
publication_date        2015-11-16
earliest_acquisition_date    2018-04-03 00:38:39.023237
inserted                2018-04-27 19:19:31.775522
updated                 2020-12-19 09:24:23.984289
cs_paper                True
medical_paper           False
springer_paper          False
normalized_title        benchmarking stateoftheart classification algorithms for credit scoring an update of research
science_journal         False
"""


NUM_PROCESSES = 4

class DataGetter:
    def __init__(self, timestamp: datetime, output_dir: str, num_processes = NUM_PROCESSES):
        """
        Create data getter object and load timestamp from last run
        :param output_dir:
        """
        self.last_time = timestamp
        self.output_dir = output_dir
        self.num_processes = num_processes

    def get_new_data(self):
        """
        Fetch new data from S2 corpus DB
        :return:
        """
        output_file = os.path.join(self.output_dir, f'redshift_data.csv')

        # check if file exists; if so, read from, else make query
        new_data = []
        if os.path.exists(output_file):
            print('Reading from file...')
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f, quotechar='|')
                for line in tqdm.tqdm(reader):
                    new_data.append(line)
        else:
            if self.last_time:
                psql_date_format = self.last_time.strftime('%Y-%m-%d %H:%M:%S.%f+00')
                query = f"""
                    SELECT paper_sha, corpus_paper_id, 
                           title, doi, pubmed_id,
                           year, venue, abstract, 
                           fields_of_study 
                    FROM content.papers
                    WHERE from_medline  
                        AND title IS NOT NULL
                        AND title <> ''
                        AND pubmed_id
                        AND updated > '{psql_date_format}';
                """
            else:
                query = """
                    SELECT paper_sha, corpus_paper_id, 
                           title, doi, pubmed_id,
                           year, venue, abstract, 
                           fields_of_study 
                    FROM content.papers
                    WHERE from_medline
                        AND title IS NOT NULL
                        AND title <> ''
                        AND pubmed_id;
                """
            db_iterator = S2DBIterator(query, db_config=DB_REDSHIFT)

            headers = [
                "paper_id", "corpus_id", "title", "doi", "pmid",
                "year", "venue", "abstract", "body_text", "section_headers", "mag_fos"
            ]
            seen_pmid = set([])
            with open(output_file, 'w') as outf:
                writer = csv.writer(outf, quotechar='|')
                writer.writerow(headers)
                for sha, corpus_id, title, doi, pmid, year, venue, abstract, fos in tqdm.tqdm(db_iterator):
                    if pmid and pmid not in seen_pmid:
                        entry = [
                            sha, corpus_id, title, doi, pmid, year, venue, abstract, "", [], fos
                        ]
                        writer.writerow(entry)
                        new_data.append({
                            "paper_id": sha,
                            "corpus_id": corpus_id,
                            "title": title,
                            "doi": doi,
                            "pmid": pmid,
                            "year": year,
                            "venue": venue,
                            "abstract": abstract,
                            "body_text": "",
                            "section_headers": [],
                            "mag_fos": fos
                        })
                        seen_pmid.add(pmid)

        # get file abstracts and save to jsonl chunks
        self.get_abstracts(new_data)

    @staticmethod
    def get_abstracts_for_batch(batch: Dict):
        """
        Fetch abstracts for one batch
        :param batch:
        :return:
        """
        data_chunk = batch['data_chunk']
        out_file_for_chunk = batch['out_file']
        skip_file_for_chunk = batch['skip_file']

        sds = SdsClient(SOURCE_DATA_SERVICE, 'suppai-data-getter')

        with open(out_file_for_chunk, 'w') as outf, open(skip_file_for_chunk, 'w') as skipf:
            for data_entry in tqdm.tqdm(data_chunk):
                try:
                    res = sds.find_by_pmid(data_entry['pmid'], els=[PaperElementType.Abstract])
                    try:
                        abstract = res[0]['elements'][0]['text']
                        data_entry['abstract'] = abstract
                        json.dump(data_entry, outf)
                        outf.write('\n')
                    except IndexError:
                        skipf.write(data_entry['paper_id'])
                        skipf.write('\n')
                except Exception:
                    sleep(1)

    def get_abstracts(self, data: List):
        """
        Fetch abstracts of all papers from SDS
        :param data:
        :return:
        """
        batches = [{
            "data_chunk": chunk,
            "out_file": os.path.join(self.output_dir, f'data_{ind:02d}.jsonl'),
            "skip_file": os.path.join(self.output_dir, f'skipped_{ind:02d}.skip')
        } for ind, chunk in enumerate(make_chunks(data, self.num_processes))]

        with multiprocessing.Pool(processes=NUM_PROCESSES) as p:
            p.map(self.get_abstracts_for_batch, batches)