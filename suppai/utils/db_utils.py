from collections import defaultdict
from typing import List, Dict

from suppai.data import PaperAuthor


def get_paper_metadata_no_sha(pids: List[str]) -> Dict[str, Dict]:
    """
    Get metadata entries from papers table without SHA
    """

    raise NotImplementedError("This function is not implemented yet.")
    # reimplement with API
    # metadata_query = """
    #     SELECT p.id, p.title, p.year, p.venue, p.doi, p.pmid, p.fields_of_study
    #     FROM papers p
    #     WHERE p.id in ({});
    # """.format(','.join([f"'{pid}'" for pid in pids]))

    # s2_id_to_metadata = dict()
    # for pid, title, year, venue, doi, pmid, fos in S2DBIterator(query_text=metadata_query, db_config=DB_S2_CORPUS):
    #     s2_id_to_metadata[str(pid)] = {
    #         "title": title,
    #         "authors": [],
    #         "year": year,
    #         "venue": venue,
    #         "doi": doi,
    #         "pmid": pmid,
    #         "fields_of_study": fos
    #     }
    # return s2_id_to_metadata