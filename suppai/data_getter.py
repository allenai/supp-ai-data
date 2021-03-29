"""
Retrieve Pubmed papers from S2 database along with metadata, abstracts, and full text
For use in SDI identification...
"""

import os
import json
import tqdm
from datetime import datetime
import subprocess

from s2base2.config import DB_REDSHIFT
from s2base2.db_utils import S2DBIterator


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


class DataGetter:
    def __init__(self, timestamp: datetime, output_dir: str):
        """
        Create data getter object and load timestamp from last run
        :param output_dir:
        """
        self.last_time = timestamp
        self.output_dir = output_dir

    def get_new_data(self):
        """
        Fetch new data from S2 corpus DB
        :return:
        """
        output_file = os.path.join(self.output_dir, f'redshift_data.jsonl')

        if self.last_time:
            psql_date_format = self.last_time.strftime('%Y-%m-%d %H:%M:%S.%f+00')
            query = f"""
                SELECT corpus_paper_id, 
                       title, doi, pmid,
                       year, venue, abstract, 
                       fields_of_study 
                FROM content_ext.papers
                WHERE pmid IS NOT NULL  
                    AND title IS NOT NULL
                    AND title <> ''
                    AND abstract IS NOT NULL
                    AND abstract <> ''
                    AND updated > '{psql_date_format}';
            """
        else:
            query = """
                SELECT corpus_paper_id, 
                       title, doi, pmid,
                       year, venue, abstract, 
                       fields_of_study 
                FROM content_ext.papers
                WHERE pmid IS NOT NULL  
                    AND title IS NOT NULL
                    AND title <> ''
                    AND abstract IS NOT NULL
                    AND abstract <> '';
            """
        print("=== Query ===")
        print(query)
        db_iterator = S2DBIterator(query, db_config=DB_REDSHIFT)

        seen_pmid = set([])
        with open(output_file, 'w') as outf:
            for corpus_id, title, doi, pmid, year, venue, abstract, fos in tqdm.tqdm(db_iterator):
                try:
                    if pmid and pmid not in seen_pmid:
                        json.dump({
                            "paper_id": None,
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
                        }, outf)
                        outf.write('\n')
                        seen_pmid.add(pmid)
                except Exception:
                    print(f'Issue with {corpus_id}')
                    pass

        # split file into chunks
        subprocess.run(["split", "-l", "100000", output_file, os.path.join(self.output_dir, f's2_data_')])