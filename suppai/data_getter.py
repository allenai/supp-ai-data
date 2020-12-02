"""
Retrieve Pubmed papers from S2 database along with metadata, abstracts, and full text
For use in SDI identification...
"""

import os
import json
import tqdm
from datetime import datetime

from s2base2.db_utils import S2DBIterator


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
        if self.last_time:
            # TODO: this query no longer executes properly, psql kicks me off
            psql_date_format = self.last_time.strftime('%Y-%m-%d %H:%M:%S.%f+00')
            query = f"""
                SELECT t2.sha, t1.title, t1.doi, t1.pmid, 
                       t1.year, t1.venue, t1.abstract, 
                       t1.section_body, t1.section_header, 
                       t1.fields_of_study 
                FROM papers t1 
                INNER JOIN legacy_paper_ids t2 ON t1.id = t2.paper_id 
                WHERE t1.pmid IS NOT NULL 
                    AND t1.title IS NOT NULL 
                    AND t1.abstract IS NOT NULL 
                    AND t2.id_type='Canonical'
                    AND t1.inserted > '{psql_date_format}';
            """
        else:
            query = """
                SELECT t2.sha, t1.title, t1.doi, t1.pmid, 
                       t1.year, t1.venue, t1.abstract, 
                       t1.section_body, t1.section_header, 
                       t1.fields_of_study 
                FROM papers t1 
                INNER JOIN legacy_paper_ids t2 ON t1.id = t2.paper_id 
                WHERE t1.pmid IS NOT NULL 
                    AND t1.title IS NOT NULL 
                    AND t1.abstract IS NOT NULL 
                    AND t2.id_type='Canonical';
            """
        db_iterator = S2DBIterator(query)

        for paper_id, title, doi, pmid, year, venue, abstract, body_text, section_headers, fos in tqdm.tqdm(db_iterator):
            output_file = os.path.join(self.output_dir, f'{paper_id[:4]}.jsonl')
            paper_blob = {
                "paper_id": paper_id,
                "title": title,
                "doi": doi,
                "pmid": pmid,
                "year": year,
                "venue": venue,
                "abstract": abstract,
                "body_text": body_text,
                "section_headers": section_headers,
                "mag_fos": fos
            }
            with open(output_file, 'a+') as outf:
                json.dump(paper_blob, outf)
                outf.write('\n')